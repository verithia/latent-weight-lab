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


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() == "all":
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def load_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


class ActivationStats:
    def __init__(self, model: GPT, regimes: np.ndarray, layers: list[int] | None) -> None:
        self.model = model
        self.regimes = regimes
        self.layers = set(layers) if layers is not None else None
        self.current_tokens: torch.Tensor | None = None
        self.pre_cache: dict[int, torch.Tensor] = {}
        self.stats: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.handles = []
        for idx, block in enumerate(model.transformer.h):
            if self.layers is not None and idx not in self.layers:
                continue
            self.handles.append(block.mlp.c_fc.register_forward_hook(self._pre_hook(idx)))
            self.handles.append(block.mlp.gelu.register_forward_hook(self._post_hook(idx)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _regime_indices(self) -> dict[str, torch.Tensor]:
        if self.current_tokens is None:
            return {}
        tokens = self.current_tokens.reshape(-1).detach().cpu().numpy()
        regimes = self.regimes[tokens]
        return {
            name: torch.from_numpy(np.flatnonzero(regimes == idx))
            for idx, name in enumerate(REGIMES)
            if np.any(regimes == idx)
        }

    def _pre_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.pre_cache[layer] = output.detach().float().reshape(-1, output.shape[-1]).cpu()
        return hook

    def _post_hook(self, layer: int):
        def hook(_module, _inputs, output):
            if layer not in self.pre_cache:
                return
            pre = self.pre_cache.pop(layer)
            post = output.detach().float().reshape(-1, output.shape[-1]).cpu()
            for regime, idx in self._regime_indices().items():
                pre_r = pre[idx]
                post_r = post[idx]
                if pre_r.numel() == 0:
                    continue
                key = (layer, regime)
                row = self.stats[key]
                count = float(pre_r.numel())
                token_count = float(pre_r.shape[0])
                row["tokens"] += token_count
                row["values"] += count
                row["pre_mean_sum"] += float(pre_r.sum().item())
                row["pre_abs_sum"] += float(pre_r.abs().sum().item())
                row["pre_sq_sum"] += float(pre_r.square().sum().item())
                row["post_mean_sum"] += float(post_r.sum().item())
                row["post_abs_sum"] += float(post_r.abs().sum().item())
                row["post_sq_sum"] += float(post_r.square().sum().item())
                row["positive"] += float((pre_r > 0).sum().item())
                row["near_zero_pre"] += float((pre_r.abs() < 0.1).sum().item())
                row["near_zero_post"] += float((post_r.abs() < 0.1).sum().item())
                row["strong_negative"] += float((pre_r < -1.0).sum().item())
                row["strong_positive"] += float((pre_r > 1.0).sum().item())
                row["gelu_shrink"] += float((post_r.abs() < 0.5 * pre_r.abs()).sum().item())
                row["gelu_amplify"] += float((post_r.abs() > pre_r.abs()).sum().item())
        return hook

    def rows(self) -> list[dict[str, object]]:
        rows = []
        for (layer, regime), s in sorted(self.stats.items()):
            values = max(s["values"], 1.0)
            rows.append(
                {
                    "layer": layer,
                    "regime": regime,
                    "tokens": int(s["tokens"]),
                    "pre_mean": s["pre_mean_sum"] / values,
                    "pre_abs_mean": s["pre_abs_sum"] / values,
                    "pre_rms": (s["pre_sq_sum"] / values) ** 0.5,
                    "post_mean": s["post_mean_sum"] / values,
                    "post_abs_mean": s["post_abs_sum"] / values,
                    "post_rms": (s["post_sq_sum"] / values) ** 0.5,
                    "positive_frac": s["positive"] / values,
                    "near_zero_pre_frac": s["near_zero_pre"] / values,
                    "near_zero_post_frac": s["near_zero_post"] / values,
                    "strong_negative_frac": s["strong_negative"] / values,
                    "strong_positive_frac": s["strong_positive"] / values,
                    "gelu_shrink_frac": s["gelu_shrink"] / values,
                    "gelu_amplify_frac": s["gelu_amplify"] / values,
                    "post_pre_rms_ratio": ((s["post_sq_sum"] / values) ** 0.5) / max((s["pre_sq_sum"] / values) ** 0.5, 1e-12),
                }
            )
        return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["regime"])].append(row)
    result = []
    metrics = [
        "pre_abs_mean",
        "pre_rms",
        "post_abs_mean",
        "post_rms",
        "positive_frac",
        "near_zero_pre_frac",
        "near_zero_post_frac",
        "strong_negative_frac",
        "strong_positive_frac",
        "gelu_shrink_frac",
        "post_pre_rms_ratio",
    ]
    for regime, group in sorted(grouped.items()):
        item: dict[str, object] = {"regime": regime, "layers": len(group)}
        for metric in metrics:
            item[metric] = float(np.mean([float(row[metric]) for row in group]))
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--layers", default="3,6,9")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model = load_model(Path(args.checkpoint), args.device)
    regimes = token_regimes(data_dir, model.config.vocab_size)
    collector = ActivationStats(model, regimes, parse_layers(args.layers))
    try:
        with torch.no_grad():
            for _ in range(args.max_batches):
                x = get_batch(data_dir, args.split, args.batch_size, args.block_size, args.device)
                collector.current_tokens = x
                model(x, None)
    finally:
        collector.close()
    rows = collector.rows()
    write_csv(Path(args.output), rows)
    write_csv(Path(args.summary), summarize(rows))
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
