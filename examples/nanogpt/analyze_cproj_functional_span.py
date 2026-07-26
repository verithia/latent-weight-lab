from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from examples.nanogpt.analyze_cproj_manifold import (
    load_model,
    spectral_residual_weight,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches


def solve_paired_diagonal(
    hidden: torch.Tensor,
    target_update: torch.Tensor,
    in_basis: torch.Tensor,
    out_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = hidden.float() @ in_basis.float()
    out_basis = out_basis.float()
    atom_gram = (
        features.transpose(0, 1) @ features
    ) * (out_basis.transpose(0, 1) @ out_basis)
    rhs = torch.diagonal(
        features.transpose(0, 1) @ target_update.float() @ out_basis
    )
    coefficients = torch.linalg.pinv(atom_gram, rtol=1e-7, atol=1e-10) @ rhs
    prediction = (features * coefficients.unsqueeze(0)) @ out_basis.transpose(0, 1)
    return coefficients, prediction


def solve_full_core(
    hidden: torch.Tensor,
    target_update: torch.Tensor,
    in_basis: torch.Tensor,
    out_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = hidden.float() @ in_basis.float()
    out_basis = out_basis.float()
    feature_gram = features.transpose(0, 1) @ features
    output_gram = out_basis.transpose(0, 1) @ out_basis
    core = (
        torch.linalg.pinv(feature_gram, rtol=1e-7, atol=1e-10)
        @ features.transpose(0, 1)
        @ target_update.float()
        @ out_basis
        @ torch.linalg.pinv(output_gram, rtol=1e-7, atol=1e-10)
    )
    prediction = features @ core @ out_basis.transpose(0, 1)
    return core, prediction


def explained_energy(target: torch.Tensor, prediction: torch.Tensor) -> float:
    denominator = target.float().square().sum().clamp_min(1e-30)
    return float(1.0 - (target.float() - prediction.float()).square().sum() / denominator)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1).float()
    right = right.reshape(-1).float()
    denominator = left.norm() * right.norm()
    if denominator <= 0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def functional_span_metrics(
    hidden: torch.Tensor,
    target_update: torch.Tensor,
    in_basis: torch.Tensor,
    out_basis: torch.Tensor,
    learned_diagonal: torch.Tensor,
    learned_scale: torch.Tensor,
) -> dict[str, float]:
    diagonal, diagonal_prediction = solve_paired_diagonal(
        hidden, target_update, in_basis, out_basis
    )
    core, core_prediction = solve_full_core(
        hidden, target_update, in_basis, out_basis
    )
    features = hidden.float() @ in_basis.float()
    learned_prediction = (
        features
        * learned_diagonal.detach().float().reshape(-1).unsqueeze(0)
        * learned_scale.detach().float()
    ) @ out_basis.detach().float().transpose(0, 1)
    return {
        "samples": float(hidden.shape[0]),
        "target_rms": float(target_update.float().square().mean().sqrt()),
        "paired_optimal_explained_energy": explained_energy(
            target_update, diagonal_prediction
        ),
        "full_core_optimal_explained_energy": explained_energy(
            target_update, core_prediction
        ),
        "paired_optimal_coeff_l2": float(diagonal.norm()),
        "full_core_optimal_fro": float(core.norm()),
        "learned_explained_energy": explained_energy(
            target_update, learned_prediction
        ),
        "learned_target_cosine": cosine(learned_prediction, target_update),
        "learned_paired_optimal_cosine": cosine(
            learned_prediction, diagonal_prediction
        ),
        "learned_to_paired_optimal_norm": float(
            learned_prediction.norm() / diagonal_prediction.norm().clamp_min(1e-30)
        ),
    }


class LayerIOCollector:
    def __init__(self, model: torch.nn.Module, layers: list[int], sample_cap: int) -> None:
        self.layers = set(layers)
        self.sample_cap = int(sample_cap)
        self.values: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[tuple[int, str], int] = defaultdict(int)
        self.handles = []
        for index, block in enumerate(model.transformer.h):
            if index not in self.layers:
                continue
            self.handles.append(
                block.mlp.gelu.register_forward_hook(self._hook(index, "post_gelu"))
            )
            self.handles.append(
                block.mlp.register_forward_hook(self._hook(index, "mlp_out"))
            )

    def _hook(self, layer: int, point: str):
        def hook(_module, _inputs, output):
            key = (layer, point)
            remaining = self.sample_cap - self.counts[key]
            if remaining <= 0:
                return
            values = output.detach().float().reshape(-1, output.shape[-1])
            values = values[:remaining].cpu()
            self.values[key].append(values)
            self.counts[key] += int(values.shape[0])

        return hook

    def complete(self) -> bool:
        return all(
            self.counts[(layer, point)] >= self.sample_cap
            for layer in self.layers
            for point in ("post_gelu", "mlp_out")
        )

    def tensor(self, layer: int, point: str) -> torch.Tensor:
        return torch.cat(self.values[(layer, point)], dim=0)[: self.sample_cap]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def collect_layer_io(
    checkpoint: Path,
    batches: list[torch.Tensor],
    layers: list[int],
    sample_cap: int,
    device: str,
) -> dict[tuple[int, str], torch.Tensor]:
    model = load_model(checkpoint, device)
    if model.config.block_fht:
        model.prepare_block_fht_cache(dtype=next(model.parameters()).dtype)
    collector = LayerIOCollector(model, layers, sample_cap)
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch.to(device), None)
                if collector.complete():
                    break
        return {
            (layer, point): collector.tensor(layer, point)
            for layer in layers
            for point in ("post_gelu", "mlp_out")
        }
    finally:
        collector.close()
        del model
        if "cuda" in device:
            torch.cuda.empty_cache()


def candidate_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=CHECKPOINT")
    name, checkpoint = value.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("candidate must be NAME=CHECKPOINT")
    return name, Path(checkpoint)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for candidate in sorted({str(row["candidate"]) for row in rows}):
        selected = [row for row in rows if row["candidate"] == candidate]
        output[candidate] = {
            f"{key}_mean": float(np.mean([float(row[key]) for row in selected]))
            for key, value in selected[0].items()
            if key not in {"candidate", "layer"} and isinstance(value, (int, float))
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", required=True, type=Path)
    parser.add_argument("--plain-cproj", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=candidate_arg)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--layers", default="3,6,9")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--sample-cap", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",")]
    batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.sample_seed,
    )
    print("collecting matched attention-only activations", flush=True)
    attention = collect_layer_io(
        args.attention_only, batches, layers, args.sample_cap, args.device
    )
    print("collecting matched plain-cproj activations", flush=True)
    plain = collect_layer_io(
        args.plain_cproj, batches, layers, args.sample_cap, args.device
    )

    rows: list[dict[str, object]] = []
    for candidate, checkpoint in args.candidate:
        print(f"loading correction bases for {candidate}", flush=True)
        model = load_model(checkpoint, args.device)
        for layer in layers:
            mlp = model.transformer.h[layer].mlp
            if spectral_residual_weight(model, layer) is None:
                raise ValueError(f"{candidate} layer {layer} has no spectral residual")
            hidden = plain[(layer, "post_gelu")].to(args.device)
            target = (
                attention[(layer, "mlp_out")] - plain[(layer, "mlp_out")]
            ).to(args.device)
            metrics = functional_span_metrics(
                hidden,
                target,
                mlp.cproj_spectral_resid_in_basis,
                mlp.cproj_spectral_resid_out_basis,
                mlp.cproj_spectral_resid_diag,
                mlp.cproj_spectral_resid_scale,
            )
            rows.append({"candidate": candidate, "layer": layer, **metrics})
        del model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cproj_functional_span.csv", rows)
    summary = summarize(rows)
    (args.output / "cproj_functional_span_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
