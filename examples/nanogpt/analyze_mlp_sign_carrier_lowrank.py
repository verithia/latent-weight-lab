#!/usr/bin/env python3
"""Audit a shared binary carrier with private low-rank MLP envelopes.

This is the frozen H5a optimistic representation discriminator.  It fits one
global sign carrier on discovery-window residual PCs, then scores the best
rank-four private envelope plus a procedural coordinate residual on untouched
PCs, late updates, and disjoint-data paths.  It performs no LM update.
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


def state_accounting(
    *,
    rows: int,
    columns: int,
    rank: int,
    deployment_matrix_count: int,
    maximum_fraction: float,
) -> dict[str, int | float]:
    ambient = rows * columns
    denominator_bytes = ambient * deployment_matrix_count * 2
    maximum_bytes = math.floor(maximum_fraction * denominator_bytes)
    carrier_bits = ambient
    carrier_bytes = math.ceil(carrier_bits / 8)
    factor_scalars_per_matrix = rank * (rows + columns)
    factor_bytes = factor_scalars_per_matrix * deployment_matrix_count * 2
    remaining_bytes = maximum_bytes - carrier_bytes - factor_bytes
    if remaining_bytes < 0:
        raise ValueError("carrier and factors exceed the checkpoint budget")
    residual_per_matrix = remaining_bytes // (2 * deployment_matrix_count)
    residual_bytes = residual_per_matrix * deployment_matrix_count * 2
    total_bytes = carrier_bytes + factor_bytes + residual_bytes
    return {
        "ambient_scalars_per_matrix": ambient,
        "deployment_matrix_count": deployment_matrix_count,
        "denominator_fp16_bytes": denominator_bytes,
        "maximum_checkpoint_bytes": maximum_bytes,
        "carrier_bits": carrier_bits,
        "carrier_bytes": carrier_bytes,
        "carrier_byte_fraction": carrier_bytes / denominator_bytes,
        "factor_rank": rank,
        "factor_scalars_per_matrix": factor_scalars_per_matrix,
        "factor_scalars_total": factor_scalars_per_matrix
        * deployment_matrix_count,
        "factor_bytes": factor_bytes,
        "factor_byte_fraction": factor_bytes / denominator_bytes,
        "residual_coordinates_per_matrix": residual_per_matrix,
        "residual_bytes": residual_bytes,
        "total_checkpoint_bytes": total_bytes,
        "total_checkpoint_byte_fraction": total_bytes / denominator_bytes,
    }


def randomized_lowrank(
    matrices: torch.Tensor,
    *,
    rank: int,
    seed: int,
    oversample: int,
    power_iterations: int,
    reconstruction: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return deterministic randomized-SVD values and optional rank-r fit."""
    if matrices.ndim != 3:
        raise ValueError("matrices must have shape [batch, rows, columns]")
    batch, _rows, columns = matrices.shape
    width = min(columns, rank + oversample)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    omega = torch.randn(batch, columns, width, generator=generator).to(
        device=matrices.device,
        dtype=matrices.dtype,
    )
    work = torch.bmm(matrices, omega)
    for _ in range(power_iterations):
        q = torch.linalg.qr(work, mode="reduced").Q
        work = torch.bmm(matrices, torch.bmm(matrices.transpose(1, 2), q))
    q = torch.linalg.qr(work, mode="reduced").Q
    small = torch.bmm(q.transpose(1, 2), matrices)
    u_small, singular_values, vh = torch.linalg.svd(small, full_matrices=False)
    singular_values = singular_values[:, :rank]
    if not reconstruction:
        return singular_values, None
    u = torch.bmm(q, u_small[:, :, :rank])
    fitted = torch.bmm(
        u * singular_values.unsqueeze(1),
        vh[:, :rank, :],
    )
    return singular_values, fitted


def carrier_capture(
    rows_flat: torch.Tensor,
    *,
    carrier: torch.Tensor,
    support: torch.Tensor,
    matrix_rows: int,
    matrix_columns: int,
    rank: int,
    seed: int,
    batch_size: int,
    oversample: int = 12,
    power_iterations: int = 4,
) -> torch.Tensor:
    """Conservative capture by coordinates plus sign-masked rank-r factors."""
    captures: list[torch.Tensor] = []
    carrier_matrix = carrier.reshape(matrix_rows, matrix_columns)
    for start in range(0, rows_flat.shape[0], batch_size):
        batch = rows_flat[start : start + batch_size].clone()
        total = batch.double().square().sum(dim=1).clamp_min(1e-30)
        coordinate_energy = batch[:, support].double().square().sum(dim=1)
        batch[:, support] = 0
        transformed = batch.reshape(-1, matrix_rows, matrix_columns)
        transformed = transformed * carrier_matrix
        singular_values, _ = randomized_lowrank(
            transformed,
            rank=rank,
            seed=stable_seed(f"capture:{seed}:{start}", seed),
            oversample=oversample,
            power_iterations=power_iterations,
            reconstruction=False,
        )
        factor_energy = singular_values.double().square().sum(dim=1)
        captures.append(((coordinate_energy + factor_energy) / total).clamp(0, 1))
    return torch.cat(captures)


def aggregate_capture(
    rows_flat: torch.Tensor,
    **kwargs: Any,
) -> float:
    captures = carrier_capture(rows_flat, **kwargs)
    energy = rows_flat.double().square().sum(dim=1)
    return float((captures * energy).sum() / energy.sum().clamp_min(1e-30))


def initialize_carrier(
    nodes: list[Any],
    supports: dict[str, torch.Tensor],
    *,
    ambient: int,
    device: str,
) -> torch.Tensor:
    accumulator = torch.zeros(ambient, device=device)
    for node in nodes:
        batch = node.discovery_rows.to(device).clone()
        batch[:, supports[node.parameter]] = 0
        weights = node.discovery_weights.to(device)
        weights = weights / weights.sum().clamp_min(1e-30) / len(nodes)
        accumulator.add_((batch * weights.unsqueeze(1)).sum(dim=0))
    return torch.where(accumulator >= 0, 1.0, -1.0)


def fit_carrier(
    nodes: list[Any],
    supports: dict[str, torch.Tensor],
    *,
    matrix_rows: int,
    matrix_columns: int,
    rank: int,
    iterations: int,
    seed: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    ambient = matrix_rows * matrix_columns
    carrier = initialize_carrier(
        nodes,
        supports,
        ambient=ambient,
        device=device,
    )
    history: list[dict[str, float | int]] = []
    for iteration in range(iterations):
        score = torch.zeros(ambient, device=device)
        captured = 0.0
        total = 0.0
        carrier_matrix = carrier.reshape(matrix_rows, matrix_columns)
        for node_index, node in enumerate(nodes):
            node_rows = node.discovery_rows.to(device).clone()
            node_rows[:, supports[node.parameter]] = 0
            node_weights = node.discovery_weights.to(device)
            node_weights = (
                node_weights
                / node_weights.sum().clamp_min(1e-30)
                / len(nodes)
            )
            for start in range(0, node_rows.shape[0], batch_size):
                batch = node_rows[start : start + batch_size]
                weights = node_weights[start : start + batch_size]
                transformed = (
                    batch.reshape(-1, matrix_rows, matrix_columns)
                    * carrier_matrix
                )
                singular_values, fitted = randomized_lowrank(
                    transformed,
                    rank=rank,
                    seed=stable_seed(
                        f"fit:{seed}:{iteration}:{node_index}:{start}", seed
                    ),
                    oversample=8,
                    power_iterations=2,
                    reconstruction=True,
                )
                assert fitted is not None
                score.add_(
                    (
                        batch
                        * fitted.flatten(1)
                        * weights.unsqueeze(1)
                    ).sum(dim=0)
                )
                captured += float(
                    (singular_values.double().square().sum(dim=1) * weights).sum()
                )
                total += float(
                    (batch.double().square().sum(dim=1) * weights).sum()
                )
        updated = torch.where(score >= 0, 1.0, -1.0)
        history.append(
            {
                "iteration": iteration,
                "discovery_weighted_capture": captured / max(total, 1e-30),
                "carrier_flip_fraction": float((updated != carrier).float().mean()),
            }
        )
        carrier = updated
    return carrier.cpu(), history


def synthetic_self_check(device: str) -> float:
    generator = torch.Generator(device="cpu").manual_seed(101)
    carrier = torch.where(
        torch.randn(32 * 24, generator=generator) >= 0,
        1.0,
        -1.0,
    ).to(device)
    left = torch.randn(5, 32, 3, generator=generator).to(device)
    right = torch.randn(5, 24, 3, generator=generator).to(device)
    targets = carrier.reshape(32, 24) * torch.bmm(left, right.transpose(1, 2))
    captures = carrier_capture(
        targets.flatten(1),
        carrier=carrier,
        support=torch.empty(0, dtype=torch.long, device=device),
        matrix_rows=32,
        matrix_columns=24,
        rank=3,
        seed=31,
        batch_size=5,
        oversample=8,
        power_iterations=3,
    )
    return float(captures.min())


def pack_carrier(path: Path, carrier: torch.Tensor) -> dict[str, Any]:
    bits = (carrier.numpy() > 0).astype(np.uint8)
    packed = np.packbits(bits, bitorder="little")
    np.savez_compressed(path, learned_common_carrier=packed)
    return {
        "shape": list(carrier.shape),
        "logical_bits": int(bits.size),
        "packed_bytes": int(packed.size),
        "unpacked_sign_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
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
    parser.add_argument("--envelope-rank", type=int, default=4)
    parser.add_argument("--fit-iterations", type=int, default=6)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--late-start", type=int, default=180)
    parser.add_argument("--deployment-matrix-count", type=int, default=24)
    parser.add_argument("--maximum-byte-fraction", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {value for value in args.targets.split(",") if value}
    if args.envelope_rank != 4 or args.fit_iterations != 6:
        raise ValueError("the frozen H5a oracle requires rank 4 and six fits")
    if args.maximum_byte_fraction != 0.01:
        raise ValueError("the frozen H5a oracle requires a one-percent byte gate")

    accounting = state_accounting(
        rows=args.matrix_rows,
        columns=args.matrix_columns,
        rank=args.envelope_rank,
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
    if any(node.full_rows.shape[1] != ambient for node in nodes):
        raise ValueError("oriented matrix dimensions do not match the plan")

    support_count = int(accounting["residual_coordinates_per_matrix"])
    supports = {
        node.parameter: procedural_support(
            ambient,
            support_count,
            seed=stable_seed(f"h5a:{node.parameter}", args.seed),
            device=args.device,
        )
        for node in nodes
    }
    self_check = synthetic_self_check(args.device)
    if self_check < 0.999:
        raise ValueError(f"synthetic carrier reconstruction failed: {self_check}")

    learned, fit_history = fit_carrier(
        nodes,
        supports,
        matrix_rows=args.matrix_rows,
        matrix_columns=args.matrix_columns,
        rank=args.envelope_rank,
        iterations=args.fit_iterations,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_carrier = torch.where(
        torch.randn(ambient, generator=generator) >= 0,
        1.0,
        -1.0,
    )
    families = {
        "ordinary_rank4": torch.ones(ambient),
        "random_sign_rank4": random_carrier,
        "learned_sign_rank4": learned,
    }

    summary_rows: list[dict[str, Any]] = []
    for family, carrier_cpu in families.items():
        carrier = carrier_cpu.to(args.device)
        for node_index, node in enumerate(nodes):
            captures = carrier_capture(
                node.full_rows.to(args.device),
                carrier=carrier,
                support=supports[node.parameter],
                matrix_rows=args.matrix_rows,
                matrix_columns=args.matrix_columns,
                rank=args.envelope_rank,
                seed=stable_seed(f"pc:{family}:{node_index}", args.seed),
                batch_size=args.batch_size,
            )
            weighted, minimum, maximum = weighted_summary(
                captures,
                node.full_weights.to(args.device),
            )
            summary_rows.append(
                {
                    "family": family,
                    "parameter": node.parameter,
                    "target": node.target,
                    "weighted_top16_pc_capture": weighted,
                    "minimum_pc_capture": minimum,
                    "maximum_pc_capture": maximum,
                    "top16_retained_path_energy": node.retained_full_energy,
                    "total_checkpoint_byte_fraction": accounting[
                        "total_checkpoint_byte_fraction"
                    ],
                }
            )

    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    late_rows: list[dict[str, Any]] = []
    for layer in sorted(layers):
        node_steps, values, _ = load_snapshots(paths, layers={layer}, targets=targets)
        late_indices = [
            index
            for index, step in enumerate(node_steps[1:])
            if step >= args.late_start
        ]
        for parameter, tensors in sorted(values.items()):
            match = PARAMETER_PATTERN.match(parameter)
            assert match is not None
            target = match.group("target")
            positions = orient(
                target,
                torch.stack(tensors).to(args.device, torch.float32),
            )
            updates = (positions[1:] - positions[:-1]).flatten(1)[late_indices]
            centered = updates - updates.mean(dim=0, keepdim=True)
            for family, carrier_cpu in families.items():
                kwargs = {
                    "carrier": carrier_cpu.to(args.device),
                    "support": supports[parameter],
                    "matrix_rows": args.matrix_rows,
                    "matrix_columns": args.matrix_columns,
                    "rank": args.envelope_rank,
                    "seed": stable_seed(f"late:{family}:{parameter}", args.seed),
                    "batch_size": args.batch_size,
                }
                late_rows.append(
                    {
                        "family": family,
                        "parameter": parameter,
                        "target": target,
                        "late_update_count": updates.shape[0],
                        "uncentered_late_update_capture": aggregate_capture(
                            updates, **kwargs
                        ),
                        "centered_late_update_capture": aggregate_capture(
                            centered, **kwargs
                        ),
                    }
                )
            del positions, updates, centered
            torch.cuda.empty_cache()

    steps_a, run_a, metadata_a = load_weight_run(
        args.run_a_probe_dir, layer=6, targets=targets
    )
    steps_b, run_b, metadata_b = load_weight_run(
        args.run_b_probe_dir, layer=6, targets=targets
    )
    if steps_a != steps_b or set(run_a) != set(run_b):
        raise ValueError("A/B probe inventories do not match")
    multimanifold_rows: list[dict[str, Any]] = []
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
        for family, carrier_cpu in families.items():
            kwargs = {
                "carrier": carrier_cpu.to(args.device),
                "support": supports[parameter],
                "matrix_rows": args.matrix_rows,
                "matrix_columns": args.matrix_columns,
                "rank": args.envelope_rank,
                "seed": stable_seed(f"multi:{family}:{parameter}", args.seed),
                "batch_size": args.batch_size,
            }
            multimanifold_rows.append(
                {
                    "family": family,
                    "parameter": parameter,
                    "target": target,
                    "common_centered_capture": aggregate_capture(common, **kwargs),
                    "heldout_b_innovation_centered_capture": aggregate_capture(
                        innovation, **kwargs
                    ),
                }
            )
        del a, b, da, db, common, innovation
        torch.cuda.empty_cache()

    family_gates: dict[str, Any] = {}
    for family in families:
        pc = [row for row in summary_rows if row["family"] == family]
        late = [row for row in late_rows if row["family"] == family]
        multi = [row for row in multimanifold_rows if row["family"] == family]
        minimum_weighted = min(float(row["weighted_top16_pc_capture"]) for row in pc)
        family_gates[family] = {
            "minimum_weighted_top16_pc_capture": minimum_weighted,
            "minimum_pc_capture": min(float(row["minimum_pc_capture"]) for row in pc),
            "minimum_common_centered_capture": min(
                float(row["common_centered_capture"]) for row in multi
            ),
            "minimum_heldout_innovation_capture": min(
                float(row["heldout_b_innovation_centered_capture"])
                for row in multi
            ),
            "minimum_centered_late_update_capture": min(
                float(row["centered_late_update_capture"]) for row in late
            ),
            "base_gates_pass": (
                minimum_weighted >= 0.50
                and min(float(row["minimum_pc_capture"]) for row in pc) >= 0.20
                and min(float(row["common_centered_capture"]) for row in multi)
                >= 0.70
                and min(
                    float(row["heldout_b_innovation_centered_capture"])
                    for row in multi
                )
                >= 0.40
                and min(float(row["centered_late_update_capture"]) for row in late)
                >= 0.20
            ),
        }
    learned_score = family_gates["learned_sign_rank4"][
        "minimum_weighted_top16_pc_capture"
    ]
    control_score = max(
        family_gates["ordinary_rank4"]["minimum_weighted_top16_pc_capture"],
        family_gates["random_sign_rank4"]["minimum_weighted_top16_pc_capture"],
    )
    family_gates["learned_sign_rank4"]["control_margin"] = (
        learned_score - control_score
    )
    family_gates["learned_sign_rank4"]["retained"] = bool(
        family_gates["learned_sign_rank4"]["base_gates_pass"]
        and learned_score - control_score >= 0.10
    )

    args.output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "pc": args.output / "pc_capture.csv",
        "late": args.output / "late_update_capture.csv",
        "multi": args.output / "multimanifold_capture.csv",
        "accounting": args.output / "accounting.json",
        "fit": args.output / "carrier_fit.json",
        "carrier": args.output / "learned_carrier.npz",
    }
    write_csv(outputs["pc"], summary_rows)
    write_csv(outputs["late"], late_rows)
    write_csv(outputs["multi"], multimanifold_rows)
    outputs["accounting"].write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n"
    )
    outputs["fit"].write_text(
        json.dumps(fit_history, indent=2, sort_keys=True) + "\n"
    )
    carrier_manifest = pack_carrier(outputs["carrier"], learned)

    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_sign_carrier_lowrank_v1",
        "candidate": "H5a shared learned binary carrier times private rank-four envelopes",
        "optimism": "noncausal per-target best-factor representation ceiling",
        "steps": steps,
        "layers": sorted(layers),
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "envelope_rank": args.envelope_rank,
        "fit_iterations": args.fit_iterations,
        "discovery_stop": args.discovery_stop,
        "late_start": args.late_start,
        "synthetic_capture": self_check,
        "accounting": accounting,
        "carrier_manifest": carrier_manifest,
        "fit_history": fit_history,
        "family_gates": family_gates,
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
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in outputs.items()
        },
        "limitations": [
            "The carrier is fit on discovery PCs and is persistent global state.",
            "Evaluation refits private factors to each target and is therefore an optimistic representation ceiling, not a causal optimizer result.",
            "The coordinate-plus-low-rank score is a conservative lower bound because the randomized SVD penalizes fitted values on coordinate-support entries that the residual can cancel.",
            "A representation pass requires a later causal tangent/JVP and performance gate before CE.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"metadata": str(metadata_path), "family_gates": family_gates},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
