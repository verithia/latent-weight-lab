"""Measure an MLP weight trajectory from immutable exact-resume checkpoints.

This probe deliberately separates what two checkpoints can establish from a
manifold claim.  It records full-matrix displacement/norm geometry immediately,
but reports a trajectory-subspace dimension only after at least three snapshots
from the *same run* are supplied.  The sampled-entry trajectory is deterministic
and is retained so later exact snapshots can be appended and re-analysed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import torch


def parse_checkpoint(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be STEP=PATH")
    step, path = value.split("=", 1)
    try:
        parsed_step = int(step)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint STEP must be an integer") from exc
    if parsed_step < 0 or not path:
        raise argparse.ArgumentTypeError("checkpoint must be non-negative STEP=PATH")
    return parsed_step, Path(path)


def parse_layers(value: str) -> list[int]:
    layers = [int(item) for item in value.split(",") if item]
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("--layers must contain non-negative indices")
    return layers


def weight_key(layer: int, target: str) -> str:
    return f"transformer.h.{layer}.mlp.{target}.weight"


def sample_indices(size: int, sample_elements: int, seed: int, layer: int) -> torch.Tensor:
    if sample_elements <= 0:
        raise ValueError("sample_elements must be > 0")
    count = min(int(size), int(sample_elements))
    # A fixed per-layer CPU permutation is exact without allocating a GPU-sized
    # random stream.  The same entry indices are used at every trajectory step.
    digest = hashlib.sha256(f"{seed}:{layer}".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
    return torch.randperm(size, generator=generator)[:count]


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left.reshape(-1), right.reshape(-1)) / (left.norm() * right.norm()).clamp_min(1e-30))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trajectory_summary(samples: dict[int, list[tuple[int, torch.Tensor]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer, by_step in sorted(samples.items()):
        by_step.sort(key=lambda item: item[0])
        steps = [step for step, _ in by_step]
        vectors = torch.stack([value.float() for _, value in by_step])
        displacements = vectors - vectors[0]
        row: dict[str, object] = {
            "layer": layer,
            "steps": ";".join(str(step) for step in steps),
            "snapshots": len(steps),
            "sample_elements": int(vectors.shape[1]),
        }
        if len(steps) < 3:
            row.update(
                {
                    "trajectory_subspace_status": "deferred_requires_at_least_3_same_run_snapshots",
                    "trajectory_linear_rank": "",
                    "pc1_energy": "",
                    "pc2_energy": "",
                }
            )
        else:
            singular = torch.linalg.svdvals(displacements[1:])
            energy = singular.square()
            probability = energy / energy.sum().clamp_min(1e-30)
            row.update(
                {
                    "trajectory_subspace_status": "measured_sampled_entry_displacements",
                    "trajectory_linear_rank": int((singular > singular.max() * 1e-6).sum()),
                    "pc1_energy": float(probability[0]),
                    "pc2_energy": float(probability[1]) if probability.numel() > 1 else 0.0,
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", choices=("c_fc", "c_proj"), default="c_fc")
    parser.add_argument("--layers", default="3,11,19")
    parser.add_argument("--sample-elements", type=int, default=65536)
    parser.add_argument("--sample-seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoints = sorted(args.checkpoint)
    if len({step for step, _ in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint steps must be unique")
    layers = parse_layers(args.layers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    base_weights: dict[int, torch.Tensor] = {}
    indices: dict[int, torch.Tensor] = {}
    samples: dict[int, list[tuple[int, torch.Tensor]]] = {layer: [] for layer in layers}
    rows: list[dict[str, object]] = []
    config_digest: str | None = None
    for step, path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        this_digest = hashlib.sha256(json.dumps(checkpoint["model_config"], sort_keys=True).encode()).hexdigest()
        if config_digest is None:
            config_digest = this_digest
        elif this_digest != config_digest:
            raise ValueError("checkpoint model_config differs; these are not one trajectory")
        state = checkpoint["model"]
        for layer in layers:
            key = weight_key(layer, args.target)
            if key not in state:
                raise KeyError(f"missing {key} in {path}")
            weight = state[key].detach().to(device=args.device, dtype=torch.float32)
            if layer not in indices:
                indices[layer] = sample_indices(weight.numel(), args.sample_elements, args.sample_seed, layer)
            sample = weight.reshape(-1)[indices[layer].to(args.device)].cpu()
            samples[layer].append((step, sample))
            if layer not in base_weights:
                base_weights[layer] = weight.clone()
            base = base_weights[layer]
            delta = weight - base
            weight_norm = weight.norm()
            delta_norm = delta.norm()
            rows.append(
                {
                    "step": step,
                    "checkpoint": str(path),
                    "target": args.target,
                    "layer": layer,
                    "matrix_rows": weight.shape[0],
                    "matrix_cols": weight.shape[1],
                    "weight_fro": float(weight_norm),
                    "weight_mean": float(weight.mean()),
                    "weight_std": float(weight.std()),
                    "delta_fro_from_step0": float(delta_norm),
                    "delta_rel_to_step0": float(delta_norm / base.norm().clamp_min(1e-30)),
                    "weight_cos_to_step0": cosine(weight, base),
                    "sample_delta_rms_from_step0": float((sample - samples[layer][0][1]).square().mean().sqrt()),
                }
            )
            del weight, delta
        del checkpoint
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(output / "mlp_trajectory_matrix_metrics.csv", rows)
    summary = trajectory_summary(samples)
    write_csv(output / "mlp_trajectory_subspace.csv", summary)
    (output / "mlp_trajectory_metadata.json").write_text(
        json.dumps(
            {
                "target": args.target,
                "layers": layers,
                "checkpoints": [{"step": step, "path": str(path)} for step, path in checkpoints],
                "sample_elements_requested": args.sample_elements,
                "sample_seed": args.sample_seed,
                "model_config_sha256": config_digest,
                "interpretation": (
                    "Full-matrix displacement metrics are exact. The trajectory subspace is estimated from fixed "
                    "sampled entries and is intentionally deferred until >=3 immutable snapshots from one run."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "matrix_rows": len(rows), "subspace_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
