"""Compare MLP residual updates on fixed validation tokens.

This is deliberately an inference-time diagnostic.  It does not infer a weight
manifold from endpoint checkpoints; it asks whether a generated FFN produces a
residual update with a different scale, alignment, or covariance from a matched
dense FFN.  The probe uses identical validation windows for every checkpoint.
"""

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
POINTS = ("residual_in", "ln2", "pre_gelu", "post_gelu", "mlp_out", "residual_out")


def token_regimes_from_validation(data_dir: Path, vocab_size: int) -> np.ndarray:
    """Classify tokens by validation-corpus frequency without scanning train.bin."""
    values = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(values), minlength=vocab_size).astype(np.int64)
    order = np.argsort(-counts)
    total = int(counts.sum())
    cumulative = np.cumsum(counts[order])
    regime = np.full(vocab_size, 2, dtype=np.int64)
    regime[order[cumulative <= 2 * total / 3]] = 1
    regime[order[cumulative <= total / 3]] = 0
    return regime


def fixed_validation_batches(
    data_dir: Path,
    batch_size: int,
    block_size: int,
    batches: int,
    seed: int,
) -> list[torch.Tensor]:
    values = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    if len(values) <= block_size:
        raise ValueError("validation data is shorter than block_size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(len(values) - block_size, (int(batches), int(batch_size)), generator=generator)
    return [
        torch.stack([torch.from_numpy(np.asarray(values[int(index) : int(index) + block_size], dtype=np.int64)) for index in row])
        for row in indices
    ]


def effective_ranks(values: torch.Tensor) -> tuple[float, float]:
    if values.shape[0] < 2:
        return float("nan"), float("nan")
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if total <= 0:
        return 0.0, 0.0
    probability = energy / total
    soft = torch.exp(-(probability * torch.log(probability.clamp_min(1e-30))).sum())
    hard = 1.0 / probability.square().sum()
    return float(soft), float(hard)


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape[0] < 2 or right.shape[0] < 2:
        return float("nan")
    left_centered = left.float() - left.float().mean(dim=0, keepdim=True)
    right_centered = right.float() - right.float().mean(dim=0, keepdim=True)
    numerator = torch.linalg.matrix_norm(left_centered.transpose(0, 1).matmul(right_centered)).square()
    left_norm = torch.linalg.matrix_norm(left_centered.transpose(0, 1).matmul(left_centered))
    right_norm = torch.linalg.matrix_norm(right_centered.transpose(0, 1).matmul(right_centered))
    return float(numerator / (left_norm * right_norm).clamp_min(1e-30))


def compatibility_metrics(residual: torch.Tensor, update: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    """Return scale/alignment/covariance quantities for ``output = residual + update``."""
    residual = residual.float()
    update = update.float()
    output = output.float()
    residual_rms = residual.square().mean(dim=-1).sqrt()
    update_rms = update.square().mean(dim=-1).sqrt()
    output_rms = output.square().mean(dim=-1).sqrt()
    cosine = (residual * update).sum(dim=-1) / (residual.norm(dim=-1) * update.norm(dim=-1)).clamp_min(1e-30)
    reconstruction = output - residual - update
    residual_soft, residual_hard = effective_ranks(residual)
    update_soft, update_hard = effective_ranks(update)
    output_soft, output_hard = effective_ranks(output)
    return {
        "samples": float(residual.shape[0]),
        "residual_rms_mean": float(residual_rms.mean()),
        "update_rms_mean": float(update_rms.mean()),
        "output_rms_mean": float(output_rms.mean()),
        "update_to_residual_rms": float(update_rms.mean() / residual_rms.mean().clamp_min(1e-30)),
        "residual_update_cos_mean": float(cosine.mean()),
        "residual_update_cos_p05": float(torch.quantile(cosine, 0.05)),
        "residual_update_cos_p50": float(torch.quantile(cosine, 0.50)),
        "residual_update_cos_p95": float(torch.quantile(cosine, 0.95)),
        "residual_update_parallel_energy": float(cosine.square().mean()),
        "residual_update_cka": linear_cka(residual, update),
        "residual_soft_rank": residual_soft,
        "residual_hard_rank": residual_hard,
        "update_soft_rank": update_soft,
        "update_hard_rank": update_hard,
        "output_soft_rank": output_soft,
        "output_hard_rank": output_hard,
        "residual_add_reconstruction_max_abs": float(reconstruction.abs().max()),
    }


class ResidualCollector:
    def __init__(self, model: GPT, token_regimes: np.ndarray, layers: list[int], sample_cap: int) -> None:
        self.model = model
        self.token_regimes = token_regimes
        self.layers = set(layers)
        self.sample_cap = int(sample_cap)
        self.current_indices: dict[str, torch.Tensor] = {}
        self.samples: dict[tuple[int, str, str], list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[tuple[int, str, str], int] = defaultdict(int)
        self.handles = []
        for layer_index, block in enumerate(model.transformer.h):
            if layer_index not in self.layers:
                continue
            self.handles.append(block.ln_2.register_forward_pre_hook(self._pre_hook(layer_index, "residual_in")))
            self.handles.append(block.ln_2.register_forward_hook(self._hook(layer_index, "ln2")))
            self.handles.append(block.mlp.c_fc.register_forward_hook(self._hook(layer_index, "pre_gelu")))
            self.handles.append(block.mlp.gelu.register_forward_hook(self._hook(layer_index, "post_gelu")))
            self.handles.append(block.mlp.c_proj.register_forward_hook(self._hook(layer_index, "mlp_out")))
            self.handles.append(block.register_forward_hook(self._hook(layer_index, "residual_out")))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def set_tokens(self, tokens: torch.Tensor) -> None:
        flattened = tokens.reshape(-1).detach().cpu().numpy()
        labels = self.token_regimes[flattened]
        self.current_indices = {
            regime: torch.from_numpy(np.flatnonzero(labels == regime_index)).long()
            for regime_index, regime in enumerate(REGIMES)
        }

    def _pre_hook(self, layer: int, point: str):
        def hook(_module, inputs):
            self._collect(layer, point, inputs[0])

        return hook

    def _hook(self, layer: int, point: str):
        def hook(_module, _inputs, output):
            self._collect(layer, point, output)

        return hook

    def _collect(self, layer: int, point: str, output: torch.Tensor) -> None:
        values = output.detach().float().reshape(-1, output.shape[-1]).cpu()
        for regime, indices in self.current_indices.items():
            key = (layer, point, regime)
            remaining = self.sample_cap - self.counts[key]
            if remaining <= 0 or indices.numel() == 0:
                continue
            selected = indices[:remaining]
            self.samples[key].append(values[selected])
            self.counts[key] += int(selected.numel())

    def values(self, layer: int, point: str, regime: str) -> torch.Tensor | None:
        pieces = self.samples.get((layer, point, regime), [])
        if not pieces:
            return None
        return torch.cat(pieces, dim=0)[: self.sample_cap]


def load_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # GPT initializes every dense parameter before the checkpoint replaces it.
    # Constructing a 350M model on CPU makes that otherwise-discarded work take
    # tens of minutes on a single core.  Build on the requested accelerator so
    # both initialization and the eventual checkpoint residency are local.
    with torch.device(device):
        model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_checkpoint(
    name: str,
    checkpoint: Path,
    data_dir: Path,
    device: str,
    layers: list[int],
    batches: list[torch.Tensor],
    sample_cap: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    print(f"loading {name}: {checkpoint}", flush=True)
    model = load_model(checkpoint, device)
    if model.config.block_fht:
        # Inference has no parameter updates.  Materialize each generated
        # matrix once instead of re-running the Python fallback FHT for every
        # validation batch when a CUDA compiler/extension is unavailable.
        print(f"materializing inference cache for {name}", flush=True)
        model.prepare_block_fht_cache(dtype=next(model.parameters()).dtype)
        print(f"inference cache ready for {name}", flush=True)
    print(f"collecting activations for {name}", flush=True)
    regimes = token_regimes_from_validation(data_dir, model.config.vocab_size)
    collector = ResidualCollector(model, regimes, layers, sample_cap)
    try:
        with torch.no_grad():
            for batch in batches:
                collector.set_tokens(batch)
                model(batch.to(device), None)
    finally:
        collector.close()

    print(f"computing compatibility metrics for {name}", flush=True)

    rows: list[dict[str, object]] = []
    for layer in layers:
        for regime in REGIMES:
            residual = collector.values(layer, "residual_in", regime)
            update = collector.values(layer, "mlp_out", regime)
            output = collector.values(layer, "residual_out", regime)
            pre_gelu = collector.values(layer, "pre_gelu", regime)
            post_gelu = collector.values(layer, "post_gelu", regime)
            ln2 = collector.values(layer, "ln2", regime)
            if any(value is None for value in (residual, update, output, pre_gelu, post_gelu, ln2)):
                continue
            assert residual is not None and update is not None and output is not None
            assert pre_gelu is not None and post_gelu is not None and ln2 is not None
            # Samples are collected on CPU so hook storage remains bounded, but
            # the covariance/SVD metrics are the expensive part of this probe.
            # Move only the capped samples back to the requested device here.
            # On a 97GB PRO6 this avoids serial CPU SVDs for 512 x 4096 FFN
            # activations while preserving exactly the same samples and math.
            residual, update, output, pre_gelu, post_gelu, ln2 = (
                value.to(device) for value in (residual, update, output, pre_gelu, post_gelu, ln2)
            )
            metrics = compatibility_metrics(residual, update, output)
            pre_soft, pre_hard = effective_ranks(pre_gelu)
            post_soft, post_hard = effective_ranks(post_gelu)
            ln2_soft, ln2_hard = effective_ranks(ln2)
            rows.append(
                {
                    "run": name,
                    "checkpoint": str(checkpoint),
                    "layer": layer,
                    "regime": regime,
                    **metrics,
                    "ln2_soft_rank": ln2_soft,
                    "ln2_hard_rank": ln2_hard,
                    "pregelu_soft_rank": pre_soft,
                    "pregelu_hard_rank": pre_hard,
                    "postgelu_soft_rank": post_soft,
                    "postgelu_hard_rank": post_hard,
                    "postgelu_to_pregelu_hard_rank": post_hard / max(pre_hard, 1e-30),
                }
            )
    metadata = {
        "checkpoint": str(checkpoint),
        "model_config": model.config.__dict__,
        "rows": len(rows),
    }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="3,6,9")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--sample-cap", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=20260716)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    layers = [int(value) for value in args.layers.split(",") if value]
    batches = fixed_validation_batches(data_dir, args.batch_size, args.block_size, args.batches, args.sample_seed)
    rows: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "layers": layers,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "batches": args.batches,
        "sample_cap": args.sample_cap,
        "sample_seed": args.sample_seed,
        "data_dir": str(data_dir),
        "checkpoints": {},
    }
    for name, checkpoint in args.checkpoint:
        run_rows, run_metadata = collect_checkpoint(
            name, checkpoint, data_dir, args.device, layers, batches, args.sample_cap
        )
        rows.extend(run_rows)
        metadata["checkpoints"][name] = run_metadata
    write_csv(output / "residual_compatibility.csv", rows)
    (output / "residual_compatibility.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
