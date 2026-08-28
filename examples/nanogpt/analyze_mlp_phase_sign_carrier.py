#!/usr/bin/env python3
"""Audit two chronological sign-carrier charts under one percent.

H12a fits one binary carrier on early stream-A residual PCs and a second on
middle residual PCs.  A fixed chronological route extrapolates the middle
chart to untouched late updates and stream B.  Per-target private rank-two
factors and procedural coordinates are refit optimistically; no LM update is
performed.
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
    NodeBasis,
    basis_from_positions,
    git_commit,
    orient,
    procedural_support,
    stable_seed,
    weighted_summary,
)
from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_sign_carrier_lowrank import (
    aggregate_capture,
    carrier_capture,
    fit_carrier,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


def phase_state_accounting(
    *,
    rows: int,
    columns: int,
    rank: int,
    carrier_count: int,
    deployment_matrix_count: int,
    maximum_fraction: float,
) -> dict[str, int | float]:
    ambient = rows * columns
    denominator_bytes = ambient * deployment_matrix_count * 2
    maximum_bytes = math.floor(maximum_fraction * denominator_bytes)
    carrier_bits = carrier_count * ambient
    carrier_bytes = math.ceil(carrier_bits / 8)
    factor_scalars_per_matrix = rank * (rows + columns)
    factor_bytes = factor_scalars_per_matrix * deployment_matrix_count * 2
    remaining_bytes = maximum_bytes - carrier_bytes - factor_bytes
    if remaining_bytes < 0:
        raise ValueError("phase carriers and factors exceed the checkpoint budget")
    residual_per_matrix = remaining_bytes // (2 * deployment_matrix_count)
    residual_bytes = residual_per_matrix * deployment_matrix_count * 2
    total_bytes = carrier_bytes + factor_bytes + residual_bytes
    return {
        "ambient_scalars_per_matrix": ambient,
        "deployment_matrix_count": deployment_matrix_count,
        "denominator_fp16_bytes": denominator_bytes,
        "maximum_checkpoint_bytes": maximum_bytes,
        "carrier_count": carrier_count,
        "carrier_bits": carrier_bits,
        "carrier_bytes": carrier_bytes,
        "carrier_byte_fraction": carrier_bytes / denominator_bytes,
        "factor_rank": rank,
        "factor_scalars_per_matrix": factor_scalars_per_matrix,
        "factor_scalars_total": factor_scalars_per_matrix * deployment_matrix_count,
        "factor_bytes": factor_bytes,
        "factor_byte_fraction": factor_bytes / denominator_bytes,
        "residual_coordinates_per_matrix": residual_per_matrix,
        "residual_bytes": residual_bytes,
        "total_checkpoint_bytes": total_bytes,
        "total_checkpoint_byte_fraction": total_bytes / denominator_bytes,
    }


def make_node(
    parameter: str,
    target: str,
    positions: torch.Tensor,
    *,
    rank: int,
) -> NodeBasis:
    rows, weights, retained = basis_from_positions(
        positions,
        rank=rank,
        target=target,
    )
    rows = rows.cpu()
    weights = weights.cpu()
    return NodeBasis(
        parameter=parameter,
        target=target,
        discovery_rows=rows,
        discovery_weights=weights,
        full_rows=rows,
        full_weights=weights,
        retained_full_energy=retained,
    )


def routed_aggregate_capture(
    rows: torch.Tensor,
    routes: torch.Tensor,
    *,
    carriers: tuple[torch.Tensor, torch.Tensor],
    support: torch.Tensor,
    matrix_rows: int,
    matrix_columns: int,
    rank: int,
    seed: int,
    batch_size: int,
) -> float:
    if rows.shape[0] != routes.shape[0]:
        raise ValueError("rows and routes must have the same length")
    captured = 0.0
    total = 0.0
    for phase in (0, 1):
        chosen = rows[routes == phase]
        if chosen.numel() == 0:
            continue
        energy = float(chosen.double().square().sum())
        capture = aggregate_capture(
            chosen,
            carrier=carriers[phase],
            support=support,
            matrix_rows=matrix_rows,
            matrix_columns=matrix_columns,
            rank=rank,
            seed=stable_seed(f"route:{seed}:{phase}", seed),
            batch_size=batch_size,
        )
        captured += capture * energy
        total += energy
    return captured / max(total, 1e-30)


def synthetic_self_check(device: str) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(127)
    rows, columns, rank = 40, 24, 2
    signs = tuple(
        torch.where(torch.randn(rows * columns, generator=generator) >= 0, 1.0, -1.0).to(device)
        for _ in range(2)
    )
    targets: list[torch.Tensor] = []
    routes: list[int] = []
    for phase in (0, 1):
        left = torch.randn(5, rows, rank, generator=generator).to(device)
        right = torch.randn(5, columns, rank, generator=generator).to(device)
        targets.append(signs[phase].reshape(rows, columns) * torch.bmm(left, right.transpose(1, 2)))
        routes.extend([phase] * 5)
    flat = torch.cat(targets).flatten(1)
    route_tensor = torch.tensor(routes, device=device)
    empty = torch.empty(0, dtype=torch.long, device=device)
    correct = routed_aggregate_capture(
        flat,
        route_tensor,
        carriers=signs,
        support=empty,
        matrix_rows=rows,
        matrix_columns=columns,
        rank=rank,
        seed=17,
        batch_size=5,
    )
    swapped = routed_aggregate_capture(
        flat,
        route_tensor,
        carriers=(signs[1], signs[0]),
        support=empty,
        matrix_rows=rows,
        matrix_columns=columns,
        rank=rank,
        seed=17,
        batch_size=5,
    )
    return {"correct": correct, "swapped": swapped}


def pack_carriers(path: Path, carriers: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
    payload: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {}
    for phase, carrier in enumerate(carriers):
        bits = (carrier.numpy() > 0).astype(np.uint8)
        packed = np.packbits(bits, bitorder="little")
        key = f"phase_{phase}"
        payload[key] = packed
        manifest[key] = {
            "shape": list(carrier.shape),
            "logical_bits": int(bits.size),
            "packed_bytes": int(packed.size),
            "unpacked_sign_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
        }
    np.savez_compressed(path, **payload)
    return manifest


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
    parser.add_argument("--envelope-rank", type=int, default=2)
    parser.add_argument("--fit-iterations", type=int, default=6)
    parser.add_argument("--early-stop", type=int, default=119)
    parser.add_argument("--middle-stop", type=int, default=179)
    parser.add_argument("--late-start", type=int, default=180)
    parser.add_argument("--deployment-matrix-count", type=int, default=24)
    parser.add_argument("--maximum-byte-fraction", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {value for value in args.targets.split(",") if value}
    if args.envelope_rank != 2 or args.fit_iterations != 6:
        raise ValueError("the frozen H12a oracle requires rank two and six fits")
    if (args.early_stop, args.middle_stop, args.late_start) != (119, 179, 180):
        raise ValueError("the frozen H12a phase boundaries are 119/179/180")
    if args.maximum_byte_fraction != 0.01:
        raise ValueError("the frozen H12a oracle requires a one-percent byte gate")

    accounting = phase_state_accounting(
        rows=args.matrix_rows,
        columns=args.matrix_columns,
        rank=args.envelope_rank,
        carrier_count=2,
        deployment_matrix_count=args.deployment_matrix_count,
        maximum_fraction=args.maximum_byte_fraction,
    )
    ambient = args.matrix_rows * args.matrix_columns
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(paths, layers=layers, targets=targets)
    windows = {
        "early": [i for i, step in enumerate(steps) if step <= args.early_stop],
        "middle": [i for i, step in enumerate(steps) if args.early_stop < step <= args.middle_stop],
        "late": [i for i, step in enumerate(steps) if step >= args.late_start],
    }
    if any(len(indices) < args.basis_rank + 1 for indices in windows.values()):
        raise ValueError("a frozen phase window has too few states")

    phase_nodes: dict[str, list[NodeBasis]] = {name: [] for name in windows}
    positions_by_parameter: dict[str, torch.Tensor] = {}
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter: {parameter}")
        target = match.group("target")
        positions = torch.stack(tensors).to(args.device, torch.float32)
        positions_by_parameter[parameter] = positions
        for name, indices in windows.items():
            phase_nodes[name].append(
                make_node(parameter, target, positions[indices], rank=args.basis_rank)
            )
    if any(len(nodes) != len(layers) * len(targets) for nodes in phase_nodes.values()):
        raise ValueError("the frozen six-node phase inventory is incomplete")

    support_count = int(accounting["residual_coordinates_per_matrix"])
    supports = {
        parameter: procedural_support(
            ambient,
            support_count,
            seed=stable_seed(f"h12a:{parameter}", args.seed),
            device=args.device,
        )
        for parameter in positions_by_parameter
    }
    self_check = synthetic_self_check(args.device)
    if self_check["correct"] < 0.999 or self_check["correct"] - self_check["swapped"] < 0.30:
        raise ValueError(f"synthetic phase routing failed: {self_check}")

    learned_early, history_early = fit_carrier(
        phase_nodes["early"],
        supports,
        matrix_rows=args.matrix_rows,
        matrix_columns=args.matrix_columns,
        rank=args.envelope_rank,
        iterations=args.fit_iterations,
        seed=stable_seed("early", args.seed),
        batch_size=args.batch_size,
        device=args.device,
    )
    learned_middle, history_middle = fit_carrier(
        phase_nodes["middle"],
        supports,
        matrix_rows=args.matrix_rows,
        matrix_columns=args.matrix_columns,
        rank=args.envelope_rank,
        iterations=args.fit_iterations,
        seed=stable_seed("middle", args.seed),
        batch_size=args.batch_size,
        device=args.device,
    )
    learned = (learned_early.to(args.device), learned_middle.to(args.device))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random = tuple(
        torch.where(torch.randn(ambient, generator=generator) >= 0, 1.0, -1.0).to(args.device)
        for _ in range(2)
    )
    families = {
        "learned_correct_route": learned,
        "learned_swapped_route": (learned[1], learned[0]),
        "random_correct_route": random,
    }

    pc_rows: list[dict[str, Any]] = []
    phase_route = {"early": 0, "middle": 1, "late": 1}
    for phase, nodes in phase_nodes.items():
        correct_phase = phase_route[phase]
        for node_index, node in enumerate(nodes):
            for family, carriers in families.items():
                captures = carrier_capture(
                    node.full_rows.to(args.device),
                    carrier=carriers[correct_phase],
                    support=supports[node.parameter],
                    matrix_rows=args.matrix_rows,
                    matrix_columns=args.matrix_columns,
                    rank=args.envelope_rank,
                    seed=stable_seed(f"pc:{phase}:{family}:{node_index}", args.seed),
                    batch_size=args.batch_size,
                )
                weighted, minimum, maximum = weighted_summary(
                    captures, node.full_weights.to(args.device)
                )
                pc_rows.append(
                    {
                        "family": family,
                        "phase": phase,
                        "route": correct_phase,
                        "parameter": node.parameter,
                        "target": node.target,
                        "weighted_top16_pc_capture": weighted,
                        "minimum_pc_capture": minimum,
                        "maximum_pc_capture": maximum,
                        "top16_retained_phase_energy": node.retained_full_energy,
                    }
                )

    late_rows: list[dict[str, Any]] = []
    late_update_indices = [i for i, step in enumerate(steps[1:]) if step >= args.late_start]
    for parameter, positions in sorted(positions_by_parameter.items()):
        match = PARAMETER_PATTERN.match(parameter)
        assert match is not None
        target = match.group("target")
        oriented = orient(target, positions)
        updates = (oriented[1:] - oriented[:-1]).flatten(1)[late_update_indices]
        centered = updates - updates.mean(dim=0, keepdim=True)
        for family, carriers in families.items():
            kwargs = {
                "carrier": carriers[1],
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
                    "uncentered_late_update_capture": aggregate_capture(updates, **kwargs),
                    "centered_late_update_capture": aggregate_capture(centered, **kwargs),
                }
            )

    steps_a, run_a, metadata_a = load_weight_run(args.run_a_probe_dir, layer=6, targets=targets)
    steps_b, run_b, metadata_b = load_weight_run(args.run_b_probe_dir, layer=6, targets=targets)
    if steps_a != steps_b or set(run_a) != set(run_b):
        raise ValueError("A/B probe inventories do not match")
    route_tensor = torch.tensor(
        [0 if step <= args.early_stop else 1 for step in steps_a],
        device=args.device,
    )
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
        for family, carriers in families.items():
            kwargs = {
                "routes": route_tensor,
                "carriers": carriers,
                "support": supports[parameter],
                "matrix_rows": args.matrix_rows,
                "matrix_columns": args.matrix_columns,
                "rank": args.envelope_rank,
                "seed": stable_seed(f"multi:{family}:{parameter}", args.seed),
                "batch_size": args.batch_size,
            }
            multi_rows.append(
                {
                    "family": family,
                    "parameter": parameter,
                    "target": target,
                    "common_centered_capture": routed_aggregate_capture(common, **kwargs),
                    "heldout_b_innovation_centered_capture": routed_aggregate_capture(innovation, **kwargs),
                }
            )

    gates: dict[str, Any] = {}
    for family in families:
        pc = [row for row in pc_rows if row["family"] == family]
        late = [row for row in late_rows if row["family"] == family]
        multi = [row for row in multi_rows if row["family"] == family]
        gates[family] = {
            "minimum_phase_weighted_top16_pc_capture": min(float(row["weighted_top16_pc_capture"]) for row in pc),
            "minimum_pc_capture": min(float(row["minimum_pc_capture"]) for row in pc),
            "minimum_common_centered_capture": min(float(row["common_centered_capture"]) for row in multi),
            "minimum_heldout_innovation_capture": min(float(row["heldout_b_innovation_centered_capture"]) for row in multi),
            "minimum_centered_late_update_capture": min(float(row["centered_late_update_capture"]) for row in late),
        }
    learned_gate = gates["learned_correct_route"]
    random_gate = gates["random_correct_route"]
    swapped_gate = gates["learned_swapped_route"]
    late_margin = float(learned_gate["minimum_centered_late_update_capture"]) - max(
        float(random_gate["minimum_centered_late_update_capture"]),
        float(swapped_gate["minimum_centered_late_update_capture"]),
    )
    learned_gate["late_control_margin"] = late_margin
    learned_gate["retained_discriminator"] = bool(
        float(learned_gate["minimum_phase_weighted_top16_pc_capture"]) >= 0.30
        and float(learned_gate["minimum_pc_capture"]) >= 0.05
        and float(learned_gate["minimum_common_centered_capture"]) >= 0.30
        and float(learned_gate["minimum_heldout_innovation_capture"]) >= 0.10
        and float(learned_gate["minimum_centered_late_update_capture"]) >= 0.10
        and late_margin >= 0.05
    )

    args.output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "pc": args.output / "phase_pc_capture.csv",
        "late": args.output / "late_update_capture.csv",
        "multi": args.output / "multimanifold_capture.csv",
        "accounting": args.output / "accounting.json",
        "fit": args.output / "carrier_fit.json",
        "carriers": args.output / "learned_phase_carriers.npz",
    }
    write_csv(outputs["pc"], pc_rows)
    write_csv(outputs["late"], late_rows)
    write_csv(outputs["multi"], multi_rows)
    outputs["accounting"].write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    outputs["fit"].write_text(
        json.dumps({"early": history_early, "middle": history_middle}, indent=2, sort_keys=True) + "\n"
    )
    carrier_manifest = pack_carriers(outputs["carriers"], (learned_early, learned_middle))
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_phase_sign_carrier_v1",
        "candidate": "H12a two phase sign carriers with private rank-two envelope",
        "optimism": "noncausal per-target private-factor ceiling with causal carrier split",
        "steps": steps,
        "phase_windows": {name: [steps[i] for i in indices] for name, indices in windows.items()},
        "layers": sorted(layers),
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "envelope_rank": args.envelope_rank,
        "fit_iterations": args.fit_iterations,
        "accounting": accounting,
        "synthetic_self_check": self_check,
        "fit_history": {"early": history_early, "middle": history_middle},
        "family_gates": gates,
        "carrier_manifest": carrier_manifest,
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
            "Both learned carriers see only stream A; the late carrier sees steps 120 through 179 only.",
            "Private factors are refit separately to every target, so this is an optimistic representation ceiling.",
            "The coordinate-plus-low-rank score is conservative on residual-support entries.",
            "A discriminator pass authorizes only a three-chart proof, never CE directly.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "family_gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
