#!/usr/bin/env python3
"""Audit a learned three-bit ambient partition with private values/gains.

H4a stores one shared three-bit category per ambient coordinate.  Each MLP
matrix uses eight private category values, private row/column gains, and a
procedural coordinate residual.  The shared partition is learned from the
discovery PCs only; all evaluation targets refit only private coordinates.
No language-model update is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.nanogpt.analyze_mlp_disjoint_data_state_transfer import load_weight_run
from examples.nanogpt.analyze_mlp_global_sign_bank import (
    git_commit,
    load_node_bases,
    orient,
    procedural_support,
    stable_seed,
    weighted_summary,
)
from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


def partition_state_accounting(
    *,
    rows: int,
    columns: int,
    category_bits: int,
    deployment_matrix_count: int,
    maximum_fraction: float,
) -> dict[str, int | float]:
    ambient = rows * columns
    categories = 1 << category_bits
    denominator_bytes = ambient * deployment_matrix_count * 2
    maximum_bytes = math.floor(maximum_fraction * denominator_bytes)
    partition_bits = category_bits * ambient
    partition_bytes = math.ceil(partition_bits / 8)
    gain_scalars_per_matrix = rows + columns
    value_scalars_per_matrix = categories
    private_scalars_per_matrix = gain_scalars_per_matrix + value_scalars_per_matrix
    private_bytes = private_scalars_per_matrix * deployment_matrix_count * 2
    remaining_bytes = maximum_bytes - partition_bytes - private_bytes
    if remaining_bytes < 0:
        raise ValueError("partition and private values exceed the budget")
    residual_per_matrix = remaining_bytes // (2 * deployment_matrix_count)
    residual_bytes = residual_per_matrix * deployment_matrix_count * 2
    total_bytes = partition_bytes + private_bytes + residual_bytes
    return {
        "ambient_scalars_per_matrix": ambient,
        "deployment_matrix_count": deployment_matrix_count,
        "denominator_fp16_bytes": denominator_bytes,
        "maximum_checkpoint_bytes": maximum_bytes,
        "category_bits": category_bits,
        "category_count": categories,
        "partition_bits": partition_bits,
        "partition_bytes": partition_bytes,
        "partition_byte_fraction": partition_bytes / denominator_bytes,
        "gain_scalars_per_matrix": gain_scalars_per_matrix,
        "value_scalars_per_matrix": value_scalars_per_matrix,
        "private_scalars_total": private_scalars_per_matrix * deployment_matrix_count,
        "private_bytes": private_bytes,
        "private_byte_fraction": private_bytes / denominator_bytes,
        "residual_coordinates_per_matrix": residual_per_matrix,
        "residual_bytes": residual_bytes,
        "total_checkpoint_bytes": total_bytes,
        "total_checkpoint_byte_fraction": total_bytes / denominator_bytes,
    }


def learn_partition(
    feature_rows: torch.Tensor,
    *,
    category_count: int,
    iterations: int,
    coordinate_batch_size: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Cluster ambient coordinates by their vector across discovery targets."""
    if feature_rows.ndim != 2:
        raise ValueError("feature_rows must be [features, ambient]")
    features, ambient = feature_rows.shape
    covariance = feature_rows.double() @ feature_rows.double().T
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    vectors = eigenvectors[:, torch.argsort(eigenvalues, descending=True)[:3]].to(
        feature_rows.dtype
    )
    scores = vectors.T @ feature_rows
    assignments = (
        (scores[0] >= 0).long()
        | ((scores[1] >= 0).long() << 1)
        | ((scores[2] >= 0).long() << 2)
    )
    history: list[dict[str, float | int]] = []
    data = feature_rows.T.contiguous()
    for iteration in range(iterations):
        centroids = torch.zeros(
            category_count,
            features,
            device=data.device,
            dtype=data.dtype,
        )
        centroids.index_add_(0, assignments, data)
        counts = torch.bincount(assignments, minlength=category_count).to(data.dtype)
        if bool((counts == 0).any()):
            raise ValueError("learned partition has an empty category")
        centroids /= counts.unsqueeze(1)
        centroid_norm = centroids.square().sum(dim=1)
        updated_parts: list[torch.Tensor] = []
        objective = 0.0
        for start in range(0, ambient, coordinate_batch_size):
            batch = data[start : start + coordinate_batch_size]
            distance = (
                batch.square().sum(dim=1, keepdim=True)
                - 2 * (batch @ centroids.T)
                + centroid_norm.unsqueeze(0)
            )
            values, chosen = distance.min(dim=1)
            updated_parts.append(chosen)
            objective += float(values.double().sum())
        updated = torch.cat(updated_parts)
        history.append(
            {
                "iteration": iteration,
                "changed_fraction": float((updated != assignments).float().mean()),
                "mean_squared_distance": objective / ambient,
                "minimum_category_count": int(counts.min()),
                "maximum_category_count": int(counts.max()),
            }
        )
        assignments = updated
    return assignments, history


def _fit_batch(
    targets: torch.Tensor,
    *,
    assignments: torch.Tensor,
    matrix_rows: int,
    matrix_columns: int,
    bilateral: bool,
    iterations: int,
) -> torch.Tensor:
    """Return fitted matrices for a batch using exact alternating LS steps."""
    batch = targets.shape[0]
    y = targets.reshape(batch, matrix_rows, matrix_columns)
    group = assignments.reshape(matrix_rows, matrix_columns)
    group_flat = assignments.unsqueeze(0).expand(batch, -1)
    left = torch.ones(batch, matrix_rows, device=y.device, dtype=y.dtype)
    right = torch.ones(batch, matrix_columns, device=y.device, dtype=y.dtype)
    values = torch.zeros(batch, 8, device=y.device, dtype=y.dtype)
    rounds = iterations if bilateral else 1
    for _ in range(rounds):
        scale = left.unsqueeze(2) * right.unsqueeze(1)
        numerator = torch.zeros_like(values)
        denominator = torch.zeros_like(values)
        numerator.scatter_add_(1, group_flat, (y * scale).flatten(1))
        denominator.scatter_add_(1, group_flat, scale.square().flatten(1))
        values = numerator / denominator.clamp_min(1e-20)
        if not bilateral:
            break

        code = values[:, assignments].reshape(batch, matrix_rows, matrix_columns)
        right_basis = code * right.unsqueeze(1)
        left = (y * right_basis).sum(dim=2) / right_basis.square().sum(dim=2).clamp_min(1e-20)
        left_norm = left.square().mean(dim=1).sqrt().clamp_min(1e-8)
        left /= left_norm.unsqueeze(1)
        values *= left_norm.unsqueeze(1)

        code = values[:, assignments].reshape(batch, matrix_rows, matrix_columns)
        left_basis = code * left.unsqueeze(2)
        right = (y * left_basis).sum(dim=1) / left_basis.square().sum(dim=1).clamp_min(1e-20)
        right_norm = right.square().mean(dim=1).sqrt().clamp_min(1e-8)
        right /= right_norm.unsqueeze(1)
        values *= right_norm.unsqueeze(1)

    code = values[:, group].reshape(batch, matrix_rows, matrix_columns)
    return left.unsqueeze(2) * code * right.unsqueeze(1)


def partition_capture(
    rows_flat: torch.Tensor,
    *,
    assignments: torch.Tensor,
    support: torch.Tensor,
    matrix_rows: int,
    matrix_columns: int,
    bilateral: bool,
    iterations: int,
    batch_size: int,
) -> torch.Tensor:
    captures: list[torch.Tensor] = []
    for start in range(0, rows_flat.shape[0], batch_size):
        batch = rows_flat[start : start + batch_size]
        fitted = _fit_batch(
            batch,
            assignments=assignments,
            matrix_rows=matrix_rows,
            matrix_columns=matrix_columns,
            bilateral=bilateral,
            iterations=iterations,
        ).flatten(1)
        error = batch - fitted
        error[:, support] = 0
        total = batch.double().square().sum(dim=1).clamp_min(1e-30)
        unexplained = error.double().square().sum(dim=1)
        captures.append((1 - unexplained / total).clamp(0, 1))
    return torch.cat(captures)


def aggregate_capture(rows_flat: torch.Tensor, **kwargs: Any) -> float:
    captures = partition_capture(rows_flat, **kwargs)
    energy = rows_flat.double().square().sum(dim=1)
    return float((captures * energy).sum() / energy.sum().clamp_min(1e-30))


def synthetic_self_check(device: str) -> float:
    generator = torch.Generator(device="cpu").manual_seed(163)
    rows, columns = 32, 24
    assignments = torch.randint(0, 8, (rows * columns,), generator=generator).to(device)
    left = (0.8 + 0.4 * torch.rand(5, rows, generator=generator)).to(device)
    right = (0.8 + 0.4 * torch.rand(5, columns, generator=generator)).to(device)
    values = torch.randn(5, 8, generator=generator).to(device)
    targets = left.unsqueeze(2) * values[:, assignments].reshape(5, rows, columns) * right.unsqueeze(1)
    captures = partition_capture(
        targets.flatten(1),
        assignments=assignments,
        support=torch.empty(0, dtype=torch.long, device=device),
        matrix_rows=rows,
        matrix_columns=columns,
        bilateral=True,
        iterations=12,
        batch_size=5,
    )
    return float(captures.min())


def pack_partition(path: Path, assignments: torch.Tensor) -> dict[str, Any]:
    values = assignments.cpu().numpy().astype(np.uint8)
    payload: dict[str, np.ndarray] = {}
    bit_hashes: dict[str, str] = {}
    for bit in range(3):
        plane = ((values >> bit) & 1).astype(np.uint8)
        payload[f"bit_{bit}"] = np.packbits(plane, bitorder="little")
        bit_hashes[f"bit_{bit}"] = hashlib.sha256(plane.tobytes()).hexdigest()
    np.savez_compressed(path, **payload)
    return {
        "logical_bits": int(values.size * 3),
        "packed_bytes": int(sum(array.size for array in payload.values())),
        "assignment_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "bitplane_sha256": bit_hashes,
        "category_counts": np.bincount(values, minlength=8).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--run-a-probe-dir", required=True, type=Path)
    parser.add_argument("--run-b-probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,6,11")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--matrix-rows", type=int, default=3072)
    parser.add_argument("--matrix-columns", type=int, default=768)
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--category-bits", type=int, default=3)
    parser.add_argument("--partition-iterations", type=int, default=6)
    parser.add_argument("--fit-iterations", type=int, default=8)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--late-start", type=int, default=180)
    parser.add_argument("--deployment-matrix-count", type=int, default=24)
    parser.add_argument("--maximum-byte-fraction", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coordinate-batch-size", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {value for value in args.targets.split(",") if value}
    if (args.category_bits, args.partition_iterations, args.fit_iterations) != (3, 6, 8):
        raise ValueError("the frozen H4a oracle requires 3 bits, 6 partition and 8 ALS rounds")
    if args.maximum_byte_fraction != 0.01:
        raise ValueError("the frozen H4a oracle requires a one-percent byte gate")

    accounting = partition_state_accounting(
        rows=args.matrix_rows,
        columns=args.matrix_columns,
        category_bits=args.category_bits,
        deployment_matrix_count=args.deployment_matrix_count,
        maximum_fraction=args.maximum_byte_fraction,
    )
    ambient = args.matrix_rows * args.matrix_columns
    steps, nodes, snapshot_metadata = load_node_bases(
        args.snapshot_dir,
        layers=layers,
        targets=targets,
        rank=args.basis_rank,
        discovery_stop=args.discovery_stop,
        device=args.device,
    )
    if len(nodes) != len(layers) * len(targets):
        raise ValueError("the frozen six-node inventory is incomplete")
    feature_rows = torch.cat(
        [
            node.discovery_rows.to(args.device)
            * (node.discovery_weights.to(args.device) / len(nodes)).sqrt().unsqueeze(1)
            for node in nodes
        ],
        dim=0,
    )
    learned, partition_history = learn_partition(
        feature_rows,
        category_count=8,
        iterations=args.partition_iterations,
        coordinate_batch_size=args.coordinate_batch_size,
    )
    del feature_rows
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_assignments = torch.randint(0, 8, (ambient,), generator=generator).to(args.device)
    families = {
        "learned_partition_bilateral": (learned, True),
        "learned_partition_values_only": (learned, False),
        "random_partition_bilateral": (random_assignments, True),
    }
    support_count = int(accounting["residual_coordinates_per_matrix"])
    supports = {
        node.parameter: procedural_support(
            ambient,
            support_count,
            seed=stable_seed(f"h4a:{node.parameter}", args.seed),
            device=args.device,
        )
        for node in nodes
    }
    self_check = synthetic_self_check(args.device)
    if self_check < 0.999:
        raise ValueError(f"synthetic partition reconstruction failed: {self_check}")

    pc_rows: list[dict[str, Any]] = []
    for node in nodes:
        for family, (assignments, bilateral) in families.items():
            captures = partition_capture(
                node.full_rows.to(args.device),
                assignments=assignments,
                support=supports[node.parameter],
                matrix_rows=args.matrix_rows,
                matrix_columns=args.matrix_columns,
                bilateral=bilateral,
                iterations=args.fit_iterations,
                batch_size=args.batch_size,
            )
            weighted, minimum, maximum = weighted_summary(captures, node.full_weights.to(args.device))
            pc_rows.append(
                {
                    "family": family,
                    "parameter": node.parameter,
                    "target": node.target,
                    "weighted_top16_pc_capture": weighted,
                    "minimum_pc_capture": minimum,
                    "maximum_pc_capture": maximum,
                    "top16_retained_path_energy": node.retained_full_energy,
                }
            )

    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    late_rows: list[dict[str, Any]] = []
    for layer in sorted(layers):
        node_steps, values, _ = load_snapshots(paths, layers={layer}, targets=targets)
        late_indices = [i for i, step in enumerate(node_steps[1:]) if step >= args.late_start]
        for parameter, tensors in sorted(values.items()):
            match = PARAMETER_PATTERN.match(parameter)
            assert match is not None
            target = match.group("target")
            positions = orient(target, torch.stack(tensors).to(args.device, torch.float32))
            updates = (positions[1:] - positions[:-1]).flatten(1)[late_indices]
            centered = updates - updates.mean(dim=0, keepdim=True)
            for family, (assignments, bilateral) in families.items():
                kwargs = {
                    "assignments": assignments,
                    "support": supports[parameter],
                    "matrix_rows": args.matrix_rows,
                    "matrix_columns": args.matrix_columns,
                    "bilateral": bilateral,
                    "iterations": args.fit_iterations,
                    "batch_size": args.batch_size,
                }
                late_rows.append(
                    {
                        "family": family,
                        "parameter": parameter,
                        "target": target,
                        "late_update_count": updates.shape[0],
                        "uncentered_late_update_capture": aggregate_capture(updates, **kwargs),
                        "centered_late_update_capture": aggregate_capture(centered, **kwargs),
                    }
                )
            del positions, updates, centered
            torch.cuda.empty_cache()

    steps_a, run_a, metadata_a = load_weight_run(args.run_a_probe_dir, layer=6, targets=targets)
    steps_b, run_b, metadata_b = load_weight_run(args.run_b_probe_dir, layer=6, targets=targets)
    if steps_a != steps_b or set(run_a) != set(run_b):
        raise ValueError("A/B probe inventories do not match")
    multi_rows: list[dict[str, Any]] = []
    for parameter in sorted(run_a):
        match = PARAMETER_PATTERN.match(parameter)
        assert match is not None
        target = match.group("target")
        a = orient(target, torch.stack(run_a[parameter]).to(args.device, torch.float32))
        b = orient(target, torch.stack(run_b[parameter]).to(args.device, torch.float32))
        if not torch.equal(a[0], b[0]):
            raise ValueError(f"A/B step-zero mismatch: {parameter}")
        da = (a - a[0:1]).flatten(1)
        db = (b - b[0:1]).flatten(1)
        common = 0.5 * (da + db)
        innovation = 0.5 * (db - da)
        common -= common.mean(dim=0, keepdim=True)
        innovation -= innovation.mean(dim=0, keepdim=True)
        for family, (assignments, bilateral) in families.items():
            kwargs = {
                "assignments": assignments,
                "support": supports[parameter],
                "matrix_rows": args.matrix_rows,
                "matrix_columns": args.matrix_columns,
                "bilateral": bilateral,
                "iterations": args.fit_iterations,
                "batch_size": args.batch_size,
            }
            multi_rows.append(
                {
                    "family": family,
                    "parameter": parameter,
                    "target": target,
                    "common_centered_capture": aggregate_capture(common, **kwargs),
                    "heldout_b_innovation_centered_capture": aggregate_capture(innovation, **kwargs),
                }
            )

    gates: dict[str, Any] = {}
    for family in families:
        pc = [row for row in pc_rows if row["family"] == family]
        late = [row for row in late_rows if row["family"] == family]
        multi = [row for row in multi_rows if row["family"] == family]
        gates[family] = {
            "minimum_weighted_top16_pc_capture": min(float(row["weighted_top16_pc_capture"]) for row in pc),
            "minimum_pc_capture": min(float(row["minimum_pc_capture"]) for row in pc),
            "minimum_common_centered_capture": min(float(row["common_centered_capture"]) for row in multi),
            "minimum_heldout_innovation_capture": min(float(row["heldout_b_innovation_centered_capture"]) for row in multi),
            "minimum_centered_late_update_capture": min(float(row["centered_late_update_capture"]) for row in late),
        }
    learned_gate = gates["learned_partition_bilateral"]
    random_gate = gates["random_partition_bilateral"]
    margin = float(learned_gate["minimum_weighted_top16_pc_capture"]) - float(
        random_gate["minimum_weighted_top16_pc_capture"]
    )
    learned_gate["random_control_margin"] = margin
    learned_gate["retained"] = bool(
        float(learned_gate["minimum_weighted_top16_pc_capture"]) >= 0.30
        and float(learned_gate["minimum_pc_capture"]) >= 0.05
        and float(learned_gate["minimum_common_centered_capture"]) >= 0.30
        and float(learned_gate["minimum_heldout_innovation_capture"]) >= 0.10
        and float(learned_gate["minimum_centered_late_update_capture"]) >= 0.10
        and margin >= 0.10
    )

    args.output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "pc": args.output / "pc_capture.csv",
        "late": args.output / "late_update_capture.csv",
        "multi": args.output / "multimanifold_capture.csv",
        "accounting": args.output / "accounting.json",
        "fit": args.output / "partition_fit.json",
        "partition": args.output / "learned_partition.npz",
    }
    write_csv(outputs["pc"], pc_rows)
    write_csv(outputs["late"], late_rows)
    write_csv(outputs["multi"], multi_rows)
    outputs["accounting"].write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    outputs["fit"].write_text(json.dumps(partition_history, indent=2, sort_keys=True) + "\n")
    partition_manifest = pack_partition(outputs["partition"], learned)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_boolean_partition_v1",
        "candidate": "H4a shared three-bit partition with private bilateral values",
        "optimism": "noncausal per-target private-coordinate representation ceiling",
        "steps": steps,
        "layers": sorted(layers),
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "discovery_stop": args.discovery_stop,
        "late_start": args.late_start,
        "partition_iterations": args.partition_iterations,
        "fit_iterations": args.fit_iterations,
        "accounting": accounting,
        "synthetic_capture": self_check,
        "partition_history": partition_history,
        "partition_manifest": partition_manifest,
        "family_gates": gates,
        "snapshot_metadata": snapshot_metadata,
        "run_a_metadata": metadata_a,
        "run_b_metadata": metadata_b,
        "analysis_execution": {
            "source_commit": git_commit(script.parents[2]),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": [str(script), *sys.argv[1:]],
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "runtime_seconds": time.time() - started,
            "device": args.device,
        },
        "outputs": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in outputs.items()},
        "limitations": [
            "The shared partition sees only discovery PCs through step 119.",
            "Private values and gains are refit to every target, so this is an optimistic representation ceiling.",
            "Three packed bitplanes are checkpoint state even though they are not trainable real scalars.",
            "A pass authorizes a causal assignment-learning/JVP test, never CE directly.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "family_gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
