#!/usr/bin/env python3
"""Audit a procedural W0-quantile field against MLP residual trajectories.

H10c derives a node-private categorical coordinate field from the reproducible
initial weight W0.  It stores no ambient assignment or empirical basis.  Each
target may optimistically refit private category values, row/column gains, and
a procedural coordinate residual.  This is an offline representation ceiling;
it does not update a language model.
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
from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION


def field_state_accounting(
    *,
    rows: int,
    columns: int,
    category_count: int,
    deployment_matrix_count: int,
    maximum_fraction: float,
) -> dict[str, int | float]:
    ambient = rows * columns
    denominator_bytes = ambient * deployment_matrix_count * 2
    maximum_bytes = math.floor(maximum_fraction * denominator_bytes)
    private_scalars_per_matrix = rows + columns + category_count
    private_bytes = private_scalars_per_matrix * deployment_matrix_count * 2
    remaining_bytes = maximum_bytes - private_bytes
    if remaining_bytes < 0:
        raise ValueError("private field coordinates exceed the budget")
    residual_per_matrix = remaining_bytes // (2 * deployment_matrix_count)
    residual_bytes = residual_per_matrix * deployment_matrix_count * 2
    total_bytes = private_bytes + residual_bytes
    return {
        "ambient_scalars_per_matrix": ambient,
        "deployment_matrix_count": deployment_matrix_count,
        "denominator_fp16_bytes": denominator_bytes,
        "maximum_checkpoint_bytes": maximum_bytes,
        "category_count": category_count,
        "gain_scalars_per_matrix": rows + columns,
        "value_scalars_per_matrix": category_count,
        "private_scalars_total": private_scalars_per_matrix * deployment_matrix_count,
        "private_bytes": private_bytes,
        "private_byte_fraction": private_bytes / denominator_bytes,
        "residual_coordinates_per_matrix": residual_per_matrix,
        "residual_bytes": residual_bytes,
        "total_checkpoint_bytes": total_bytes,
        "total_checkpoint_byte_fraction": total_bytes / denominator_bytes,
    }


def w0_quantile_assignment(w0: torch.Tensor, *, category_count: int) -> torch.Tensor:
    """Derive balanced Gaussian-quantile categories from a reproducible W0."""
    if w0.ndim != 2:
        raise ValueError("W0 must be a matrix")
    rms = w0.double().square().mean().sqrt().clamp_min(1e-30).to(w0.dtype)
    probability = 0.5 * (1.0 + torch.erf((w0 / rms) / math.sqrt(2.0)))
    return torch.floor(probability.flatten() * category_count).long().clamp_(0, category_count - 1)


def procedural_hash_assignment(
    ambient: int,
    *,
    category_count: int,
    parameter: str,
    seed: int,
    device: str,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(f"h10c-hash:{parameter}", seed))
    return torch.randint(0, category_count, (ambient,), generator=generator).to(device)


def _fit_batch(
    targets: torch.Tensor,
    *,
    assignments: torch.Tensor,
    category_count: int,
    matrix_rows: int,
    matrix_columns: int,
    bilateral: bool,
    iterations: int,
) -> torch.Tensor:
    batch = targets.shape[0]
    y = targets.reshape(batch, matrix_rows, matrix_columns)
    group = assignments.reshape(matrix_rows, matrix_columns)
    group_flat = assignments.unsqueeze(0).expand(batch, -1)
    left = torch.ones(batch, matrix_rows, device=y.device, dtype=y.dtype)
    right = torch.ones(batch, matrix_columns, device=y.device, dtype=y.dtype)
    values = torch.zeros(batch, category_count, device=y.device, dtype=y.dtype)
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


def field_capture(
    rows_flat: torch.Tensor,
    *,
    assignments: torch.Tensor,
    category_count: int,
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
            category_count=category_count,
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
    captures = field_capture(rows_flat, **kwargs)
    energy = rows_flat.double().square().sum(dim=1)
    return float((captures * energy).sum() / energy.sum().clamp_min(1e-30))


def synthetic_self_check(device: str) -> float:
    generator = torch.Generator(device="cpu").manual_seed(163)
    rows, columns, category_count = 32, 24, 32
    assignments = torch.randint(0, category_count, (rows * columns,), generator=generator).to(device)
    left = (0.8 + 0.4 * torch.rand(5, rows, generator=generator)).to(device)
    right = (0.8 + 0.4 * torch.rand(5, columns, generator=generator)).to(device)
    values = torch.randn(5, category_count, generator=generator).to(device)
    targets = left.unsqueeze(2) * values[:, assignments].reshape(5, rows, columns) * right.unsqueeze(1)
    captures = field_capture(
        targets.flatten(1),
        assignments=assignments,
        category_count=category_count,
        support=torch.empty(0, dtype=torch.long, device=device),
        matrix_rows=rows,
        matrix_columns=columns,
        bilateral=True,
        iterations=12,
        batch_size=5,
    )
    return float(captures.min())


def assignment_record(assignments: torch.Tensor, *, category_count: int) -> dict[str, Any]:
    values = assignments.detach().cpu().numpy().astype(np.uint16)
    counts = np.bincount(values.astype(np.int64), minlength=category_count)
    return {
        "assignment_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "minimum_category_count": int(counts.min()),
        "maximum_category_count": int(counts.max()),
        "empty_category_count": int((counts == 0).sum()),
    }


def load_w0_snapshot(
    path: Path,
    *,
    layers: set[int],
    targets: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load the exact step-zero tensors without the >=3-state path guard."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trajectory snapshot: {path}")
    if int(payload.get("step", -1)) != 0:
        raise ValueError(f"the W0 field requires step zero, got {payload.get('step')}")
    selected: dict[str, torch.Tensor] = {}
    for name, tensor in payload["parameters"].items():
        match = PARAMETER_PATTERN.match(name)
        if match is None:
            continue
        if int(match.group("layer")) not in layers or match.group("target") not in targets:
            continue
        selected[name] = tensor.detach().float().contiguous()
    if not selected:
        raise ValueError("step-zero filter selected no parameters")
    metadata = {
        "schema_version": payload["schema_version"],
        "step": int(payload["step"]),
        "run_identity": payload["run_identity"],
        "run_identity_sha256": payload["run_identity_sha256"],
        "model_config": payload["model_config"],
        "storage_dtype": payload["storage_dtype"],
        "execution_provenance": payload.get("execution_provenance"),
        "snapshot_path": str(path),
        "snapshot_sha256": file_sha256(path),
    }
    return selected, metadata


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
    parser.add_argument("--category-count", type=int, default=1024)
    parser.add_argument("--fit-iterations", type=int, default=8)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--late-start", type=int, default=180)
    parser.add_argument("--deployment-matrix-count", type=int, default=24)
    parser.add_argument("--maximum-byte-fraction", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {value for value in args.targets.split(",") if value}
    if (args.category_count, args.fit_iterations) != (1024, 8):
        raise ValueError("the frozen H10c oracle requires 1024 cells and 8 ALS rounds")
    if args.maximum_byte_fraction != 0.01:
        raise ValueError("the frozen H10c oracle requires a one-percent byte gate")

    accounting = field_state_accounting(
        rows=args.matrix_rows,
        columns=args.matrix_columns,
        category_count=args.category_count,
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
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    first_values, first_metadata = load_w0_snapshot(paths[0], layers=layers, targets=targets)

    quantile_assignments: dict[str, torch.Tensor] = {}
    hash_assignments: dict[str, torch.Tensor] = {}
    field_manifest: dict[str, Any] = {}
    for parameter, tensor in sorted(first_values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        assert match is not None
        target = match.group("target")
        w0 = orient(target, tensor.unsqueeze(0).to(args.device, torch.float32))[0]
        quantile = w0_quantile_assignment(w0, category_count=args.category_count)
        hashed = procedural_hash_assignment(
            ambient,
            category_count=args.category_count,
            parameter=parameter,
            seed=args.seed,
            device=args.device,
        )
        quantile_assignments[parameter] = quantile
        hash_assignments[parameter] = hashed
        field_manifest[parameter] = {
            "w0_quantile": assignment_record(quantile, category_count=args.category_count),
            "procedural_hash": assignment_record(hashed, category_count=args.category_count),
        }
        del w0

    families = {
        "w0_quantile_bilateral": (quantile_assignments, True),
        "w0_quantile_values_only": (quantile_assignments, False),
        "procedural_hash_bilateral": (hash_assignments, True),
    }
    support_count = int(accounting["residual_coordinates_per_matrix"])
    supports = {
        node.parameter: procedural_support(
            ambient,
            support_count,
            seed=stable_seed(f"h10c-residual:{node.parameter}", args.seed),
            device=args.device,
        )
        for node in nodes
    }
    self_check = synthetic_self_check(args.device)
    if self_check < 0.999:
        raise ValueError(f"synthetic field reconstruction failed: {self_check}")

    pc_rows: list[dict[str, Any]] = []
    for node in nodes:
        for family, (assignment_bank, bilateral) in families.items():
            captures = field_capture(
                node.full_rows.to(args.device),
                assignments=assignment_bank[node.parameter],
                category_count=args.category_count,
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
            for family, (assignment_bank, bilateral) in families.items():
                kwargs = {
                    "assignments": assignment_bank[parameter],
                    "category_count": args.category_count,
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
        for family, (assignment_bank, bilateral) in families.items():
            kwargs = {
                "assignments": assignment_bank[parameter],
                "category_count": args.category_count,
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
    candidate = gates["w0_quantile_bilateral"]
    control = gates["procedural_hash_bilateral"]
    margin = float(candidate["minimum_weighted_top16_pc_capture"]) - float(
        control["minimum_weighted_top16_pc_capture"]
    )
    candidate["hash_control_margin"] = margin
    candidate["retained"] = bool(
        float(candidate["minimum_weighted_top16_pc_capture"]) >= 0.20
        and float(candidate["minimum_pc_capture"]) >= 0.05
        and float(candidate["minimum_common_centered_capture"]) >= 0.20
        and float(candidate["minimum_heldout_innovation_capture"]) >= 0.05
        and float(candidate["minimum_centered_late_update_capture"]) >= 0.05
        and margin >= 0.05
    )

    args.output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "pc": args.output / "pc_capture.csv",
        "late": args.output / "late_update_capture.csv",
        "multi": args.output / "multimanifold_capture.csv",
        "accounting": args.output / "accounting.json",
        "field_manifest": args.output / "field_manifest.json",
    }
    write_csv(outputs["pc"], pc_rows)
    write_csv(outputs["late"], late_rows)
    write_csv(outputs["multi"], multi_rows)
    outputs["accounting"].write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    outputs["field_manifest"].write_text(json.dumps(field_manifest, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_w0_quantile_field_v1",
        "candidate": "H10c procedural W0-quantile field with private bilateral values",
        "optimism": "noncausal per-target private-coordinate representation ceiling",
        "steps": steps,
        "layers": sorted(layers),
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "discovery_stop": args.discovery_stop,
        "late_start": args.late_start,
        "category_count": args.category_count,
        "fit_iterations": args.fit_iterations,
        "accounting": accounting,
        "synthetic_capture": self_check,
        "field_manifest": field_manifest,
        "family_gates": gates,
        "snapshot_metadata": snapshot_metadata,
        "first_snapshot_metadata": first_metadata,
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
            "The field is a pointwise function of reproducible W0 and stores no empirical assignment.",
            "Private values, gains, and residuals are refit to every target, so this is an optimistic ceiling.",
            "A pass authorizes a causal compact-native JVP test, never CE directly.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "family_gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
