from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from examples.nanogpt.model import GPT, GPTConfig


REGIMES = ("HEAD", "MID", "TAIL")


def load_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def effective_c_proj_weight(model: GPT, layer: int) -> torch.Tensor:
    module = model.transformer.h[layer].mlp.c_proj
    if hasattr(module, "heads"):
        parts = [head.weight.detach().float() for head in module.heads]
        if module.__class__.__name__ == "GroupedInputLinear":
            return torch.cat(parts, dim=1)
        return torch.cat(parts, dim=0)
    return module.weight.detach().float()


def spectrum_metrics(weight: torch.Tensor) -> dict[str, float]:
    singular = torch.linalg.svdvals(weight.float())
    energy = singular.square()
    total = energy.sum().clamp_min(1e-30)
    p = energy / total
    return {
        "soft_rank": float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())),
        "hard_rank": float(1.0 / p.square().sum()),
        "stable_rank": float(total / singular.max().square().clamp_min(1e-30)),
        "top1_energy": float(p[0]),
        "top10_energy": float(p[: min(10, p.numel())].sum()),
        "fro_norm": float(torch.linalg.matrix_norm(weight)),
        "spectral_norm": float(singular.max()),
        "std": float(weight.std()),
    }


def norm_metrics(weight: torch.Tensor) -> dict[str, float]:
    col = weight.norm(dim=0)
    row = weight.norm(dim=1)
    return {
        "col_norm_mean": float(col.mean()),
        "col_norm_cv": float(col.std() / col.mean().clamp_min(1e-30)),
        "col_norm_p95_p05": float(torch.quantile(col, 0.95) / torch.quantile(col, 0.05).clamp_min(1e-30)),
        "row_norm_mean": float(row.mean()),
        "row_norm_cv": float(row.std() / row.mean().clamp_min(1e-30)),
        "row_norm_p95_p05": float(torch.quantile(row, 0.95) / torch.quantile(row, 0.05).clamp_min(1e-30)),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    return float(torch.dot(af, bf) / (af.norm() * bf.norm()).clamp_min(1e-30))


def parse_manifest(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in csv.DictReader(path.open("r", encoding="utf-8")):
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parameter_flow(args: argparse.Namespace) -> None:
    manifest = parse_manifest(Path(args.manifest))
    layers = [int(part) for part in args.layers.split(",")]
    runs: dict[str, dict[int, dict[int, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    meta: dict[tuple[str, int], str] = {}
    rows = []
    for item in manifest:
        run = item["run"]
        step = int(item["step"])
        ckpt = Path(item["checkpoint"])
        model = load_model(ckpt, args.param_device)
        for layer in layers:
            weight = effective_c_proj_weight(model, layer)
            runs[run][layer][step] = weight
            metrics = {**spectrum_metrics(weight), **norm_metrics(weight)}
            row: dict[str, object] = {
                "run": run,
                "kind": item.get("kind", ""),
                "step": step,
                "layer": layer,
                **metrics,
            }
            rows.append(row)
        meta[(run, step)] = item.get("kind", "")
        del model
        if "cuda" in args.param_device:
            torch.cuda.empty_cache()

    flow_rows = []
    control = args.control_run
    for run, by_layer in runs.items():
        for layer, by_step in by_layer.items():
            if 0 not in by_step:
                continue
            w0 = by_step[0]
            dense0 = runs.get(control, {}).get(layer, {}).get(0)
            for step, weight in sorted(by_step.items()):
                delta = weight - w0
                row = {
                    "run": run,
                    "kind": meta.get((run, step), ""),
                    "step": step,
                    "layer": layer,
                    "delta_fro": float(torch.linalg.matrix_norm(delta)),
                    "delta_rel_to_w0": float(torch.linalg.matrix_norm(delta) / torch.linalg.matrix_norm(w0).clamp_min(1e-30)),
                    "weight_cos_to_init": cosine(weight, w0),
                }
                dense_by_step = runs.get(control, {}).get(layer, {})
                if step in dense_by_step:
                    dense_w = dense_by_step[step]
                    row["cos_to_control_weight"] = cosine(weight, dense_w)
                    row["fro_dist_to_control"] = float(torch.linalg.matrix_norm(weight - dense_w))
                    if dense0 is not None:
                        dense_delta = dense_w - dense0
                        row["cos_delta_to_control_delta"] = cosine(delta, dense_delta) if delta.norm() > 0 and dense_delta.norm() > 0 else float("nan")
                flow_rows.append(row)

    write_csv(Path(args.output) / "cproj_param_metrics.csv", rows)
    write_csv(Path(args.output) / "cproj_flow_metrics.csv", flow_rows)
    summarize(Path(args.output) / "cproj_param_summary.csv", rows, ["run", "kind", "step"])
    summarize(Path(args.output) / "cproj_flow_summary.csv", flow_rows, ["run", "kind", "step"])


def summarize(path: Path, rows: list[dict[str, object]], keys: list[str]) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        entry = {k: v for k, v in zip(keys, key, strict=True)}
        numeric = [k for k, v in items[0].items() if k not in entry and isinstance(v, (int, float))]
        for name in numeric:
            values = [float(item[name]) for item in items if np.isfinite(float(item[name]))]
            if values:
                entry[f"{name}_mean"] = float(np.mean(values))
        out.append(entry)
    write_csv(path, out)


def token_regimes(data_dir: Path, vocab_size: int) -> np.ndarray:
    train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(train), minlength=vocab_size).astype(np.int64)
    order = np.argsort(-counts)
    total = int(counts.sum())
    cumsum = np.cumsum(counts[order])
    regime = np.full(vocab_size, 2, dtype=np.int64)
    regime[order[cumsum <= 2.0 * total / 3.0]] = 1
    regime[order[cumsum <= total / 3.0]] = 0
    return regime


def get_batch(data_dir: Path, split: str, batch_size: int, block_size: int, device: str) -> torch.Tensor:
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    return x.to(device)


def effective_ranks(values: torch.Tensor) -> dict[str, float]:
    if values.shape[0] < 2:
        return {"soft_rank": float("nan"), "hard_rank": float("nan"), "samples": int(values.shape[0])}
    x = values.float() - values.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x.cpu())
    eig = singular.square()
    total = eig.sum().clamp_min(1e-30)
    p = eig / total
    return {
        "soft_rank": float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())),
        "hard_rank": float(1.0 / p.square().sum()),
        "samples": int(values.shape[0]),
    }


class ActivationCollector:
    def __init__(self, model: GPT, regimes: np.ndarray, layers: list[int], sample_cap: int) -> None:
        self.model = model
        self.regimes = regimes
        self.layers = set(layers)
        self.sample_cap = int(sample_cap)
        self.current_tokens: torch.Tensor | None = None
        self.samples: dict[tuple[int, str, str], list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[tuple[int, str, str], int] = defaultdict(int)
        self.handles = []
        for idx, block in enumerate(model.transformer.h):
            if idx not in self.layers:
                continue
            self.handles.append(block.mlp.c_fc.register_forward_hook(self._hook(idx, "pre_gelu")))
            self.handles.append(block.mlp.gelu.register_forward_hook(self._hook(idx, "post_gelu")))
            self.handles.append(block.mlp.c_proj.register_forward_hook(self._hook(idx, "mlp_out")))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer: int, point: str):
        def hook(_module, _inputs, output):
            if self.current_tokens is None:
                return
            acts = output.detach().float().reshape(-1, output.shape[-1]).cpu()
            tokens = self.current_tokens.reshape(-1).detach().cpu().numpy()
            regimes = self.regimes[tokens]
            for idx, name in enumerate(REGIMES):
                mask = np.flatnonzero(regimes == idx)
                if mask.size == 0:
                    continue
                key = (layer, point, name)
                remaining = self.sample_cap - self.counts[key]
                if remaining <= 0:
                    continue
                if mask.size > remaining:
                    mask = np.random.choice(mask, size=remaining, replace=False)
                selected = acts[torch.from_numpy(mask)]
                self.samples[key].append(selected)
                self.counts[key] += int(selected.shape[0])
        return hook

    def full(self) -> bool:
        for layer in self.layers:
            for point in ("pre_gelu", "post_gelu", "mlp_out"):
                for regime in REGIMES:
                    if self.counts[(layer, point, regime)] < self.sample_cap:
                        return False
        return True

    def rows(self) -> list[dict[str, object]]:
        rows = []
        for (layer, point, regime), pieces in sorted(self.samples.items()):
            values = torch.cat(pieces, dim=0)[: self.sample_cap]
            ranks = effective_ranks(values)
            rows.append({"layer": layer, "point": point, "regime": regime, **ranks})
        return rows


def activation_ranks(args: argparse.Namespace) -> None:
    manifest = parse_manifest(Path(args.manifest))
    selected_steps = {int(part) for part in args.activation_steps.split(",")}
    selected_runs = set(args.activation_runs.split(","))
    layers = [int(part) for part in args.layers.split(",")]
    data_dir = Path(args.data_dir)
    rows = []
    for item in manifest:
        run = item["run"]
        step = int(item["step"])
        if run not in selected_runs or step not in selected_steps:
            continue
        model = load_model(Path(item["checkpoint"]), args.device)
        regimes = token_regimes(data_dir, model.config.vocab_size)
        collector = ActivationCollector(model, regimes, layers, args.sample_cap)
        try:
            with torch.no_grad():
                for _ in range(args.max_batches):
                    x = get_batch(data_dir, args.split, args.batch_size, args.block_size, args.device)
                    collector.current_tokens = x
                    model(x, None)
                    if collector.full():
                        break
        finally:
            collector.close()
        for row in collector.rows():
            rows.append({"run": run, "kind": item.get("kind", ""), "step": step, **row})
    write_csv(Path(args.output) / "cproj_activation_ranks.csv", rows)
    summarize(Path(args.output) / "cproj_activation_rank_summary.csv", rows, ["run", "kind", "step", "point", "regime"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-run", default="attn_only")
    parser.add_argument("--layers", default="3,6,9")
    parser.add_argument("--data-dir", default="/root/userdata/MappingNetworks/data/finewebedu_2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--param-device", default="cuda")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--sample-cap", type=int, default=512)
    parser.add_argument("--activation-runs", default="attn_only,cproj_group12,cproj_outg12,g12_pregain_cproj_group12,g12_pregain_cproj_outg12")
    parser.add_argument("--activation-steps", default="0,250,750")
    parser.add_argument("--skip-activations", action="store_true")
    args = parser.parse_args()
    parameter_flow(args)
    if not args.skip_activations:
        activation_ranks(args)
    print(json.dumps({"output": args.output}, indent=2))


if __name__ == "__main__":
    main()
