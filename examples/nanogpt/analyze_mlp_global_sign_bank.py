#!/usr/bin/env python3
"""Audit a bitpacked global MLP orientation bank plus procedural residual.

The global atoms are learned only from discovery-window residual PCs.  They
are stored as one-bit signs and shared across all shape-compatible MLP
matrices.  Each matrix receives scalar mixture coefficients and an optional
procedurally addressed coordinate residual.  No empirical per-node PCA vector
or dense floating-point shadow is stored.

This is an offline representation discriminator.  It does not train a
language model and must not be interpreted as a CE result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.nanogpt.analyze_mlp_disjoint_data_state_transfer import load_weight_run
from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_residual_qtt_basis import residual_temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    load_snapshots,
    parse_int_list,
    write_csv,
)


@dataclass(frozen=True)
class NodeBasis:
    parameter: str
    target: str
    discovery_rows: torch.Tensor
    discovery_weights: torch.Tensor
    full_rows: torch.Tensor
    full_weights: torch.Tensor
    retained_full_energy: float


@dataclass(frozen=True)
class Family:
    name: str
    atoms: torch.Tensor
    active: dict[str, tuple[int, ...]]
    residual: bool
    fixed_bits: int
    local_coefficients_per_matrix: int
    premise_control: bool = False


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def orient(target: str, matrices: torch.Tensor) -> torch.Tensor:
    if target == "mlp.c_fc":
        return matrices
    if target == "mlp.c_proj":
        return matrices.transpose(-1, -2)
    raise ValueError(f"unsupported target: {target}")


def normalized_sign(rows: torch.Tensor) -> torch.Tensor:
    signs = torch.where(rows >= 0, torch.ones_like(rows), -torch.ones_like(rows))
    return signs / math.sqrt(rows.shape[1])


def learned_sign_atoms(
    rows: torch.Tensor,
    weights: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Return signs of the top target-covariance directions.

    The covariance is solved through the small sample Gram matrix, so no
    ambient P by P matrix is formed.
    """
    if rows.ndim != 2 or weights.shape != (rows.shape[0],):
        raise ValueError("rows and weights have incompatible shapes")
    if not 1 <= count <= rows.shape[0]:
        raise ValueError("atom count must fit the sample count")
    weights = weights.double()
    weights = weights / weights.sum().clamp_min(1e-30)
    weighted = rows.double() * weights.sqrt().unsqueeze(1)
    gram = weighted @ weighted.T
    gram = (gram + gram.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:count]
    values = eigenvalues[order].clamp_min(1e-30)
    vectors = eigenvectors[:, order]
    real_atoms = (vectors.T @ weighted) / values.sqrt().unsqueeze(1)
    return normalized_sign(real_atoms.float())


def project_rows(rows: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Squared-energy capture of rows by the span of atom rows."""
    if atoms.shape[0] == 0:
        return torch.zeros(rows.shape[0], dtype=torch.float64, device=rows.device)
    gram = atoms.double() @ atoms.double().T
    inverse = torch.linalg.pinv(gram, hermitian=True, rtol=1e-10)
    dots = rows.double() @ atoms.double().T
    projected = torch.einsum("bi,ij,bj->b", dots, inverse, dots)
    total = rows.double().square().sum(dim=1).clamp_min(1e-30)
    return (projected / total).clamp(0.0, 1.0)


def procedural_support(size: int, count: int, *, seed: int, device: str) -> torch.Tensor:
    if not 0 <= count <= size:
        raise ValueError("support count is outside the ambient dimension")
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[:count].to(device)


def union_capture(
    rows: torch.Tensor,
    atoms: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Exact capture by coordinate axes union residualized global atoms."""
    if support.numel() == 0:
        return project_rows(rows, atoms)
    selected = rows[:, support]
    selected_energy = selected.double().square().sum(dim=1)
    total = rows.double().square().sum(dim=1).clamp_min(1e-30)
    if atoms.shape[0] == 0:
        return (selected_energy / total).clamp(0.0, 1.0)
    residual_atoms = atoms.clone()
    residual_atoms[:, support] = 0
    gram = residual_atoms.double() @ residual_atoms.double().T
    inverse = torch.linalg.pinv(gram, hermitian=True, rtol=1e-10)
    dots = rows.double() @ residual_atoms.double().T
    atom_energy = torch.einsum("bi,ij,bj->b", dots, inverse, dots)
    return ((selected_energy + atom_energy) / total).clamp(0.0, 1.0)


def weighted_summary(captures: torch.Tensor, weights: torch.Tensor) -> tuple[float, float, float]:
    weights = weights.double() / weights.double().sum().clamp_min(1e-30)
    return (
        float((captures.double() * weights).sum()),
        float(captures.min()),
        float(captures.max()),
    )


def aggregate_rows_capture(rows: torch.Tensor, atoms: torch.Tensor, support: torch.Tensor) -> float:
    per_row = union_capture(rows, atoms, support)
    energy = rows.double().square().sum(dim=1)
    return float((per_row * energy).sum() / energy.sum().clamp_min(1e-30))


def residual_budget_per_matrix(
    *,
    dense_scalars_per_matrix: int,
    deployment_matrix_count: int,
    maximum_fraction: float,
    fixed_bits: int,
    coefficients_per_matrix: int,
    residual: bool,
) -> tuple[int, dict[str, float | int]]:
    denominator_scalars = dense_scalars_per_matrix * deployment_matrix_count
    maximum_fp16_bytes = math.floor(maximum_fraction * denominator_scalars * 2)
    fixed_bytes = math.ceil(fixed_bits / 8)
    coefficient_bytes = coefficients_per_matrix * deployment_matrix_count * 2
    remaining_bytes = maximum_fp16_bytes - fixed_bytes - coefficient_bytes
    if remaining_bytes < 0:
        raise ValueError("fixed bank and coefficients exceed the checkpoint budget")
    residual_coordinates = (
        remaining_bytes // (2 * deployment_matrix_count) if residual else 0
    )
    used_bytes = (
        fixed_bytes
        + coefficient_bytes
        + residual_coordinates * deployment_matrix_count * 2
    )
    return residual_coordinates, {
        "denominator_fp16_bytes": denominator_scalars * 2,
        "maximum_checkpoint_bytes": maximum_fp16_bytes,
        "fixed_orientation_bits": fixed_bits,
        "fixed_orientation_bytes": fixed_bytes,
        "local_coefficient_scalars": coefficients_per_matrix * deployment_matrix_count,
        "local_coefficient_bytes": coefficient_bytes,
        "residual_coordinates_per_matrix": residual_coordinates,
        "residual_coordinate_bytes": residual_coordinates * deployment_matrix_count * 2,
        "total_checkpoint_bytes": used_bytes,
        "total_checkpoint_byte_fraction": used_bytes / (denominator_scalars * 2),
        "local_real_scalar_fraction": (
            coefficients_per_matrix * deployment_matrix_count
            + residual_coordinates * deployment_matrix_count
        )
        / denominator_scalars,
    }


def basis_from_positions(
    positions: torch.Tensor,
    *,
    rank: int,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    _residuals, eigenvalues, basis = residual_temporal_basis(
        positions, maximum_rank=rank
    )
    retained = eigenvalues[: basis.shape[1]]
    weights = retained / retained.sum().clamp_min(1e-30)
    matrices = basis.T.reshape(basis.shape[1], *positions.shape[1:])
    rows = orient(target, matrices).flatten(1).contiguous()
    return rows, weights.float(), float(retained.sum() / eigenvalues.sum().clamp_min(1e-30))


def load_node_bases(
    snapshot_dir: Path,
    *,
    layers: set[int],
    targets: set[str],
    rank: int,
    discovery_stop: int,
    device: str,
) -> tuple[list[int], list[NodeBasis], dict[str, Any]]:
    paths = sorted(snapshot_dir.glob("step_*.pt"))
    steps, values, metadata = load_snapshots(paths, layers=layers, targets=targets)
    discovery = [index for index, step in enumerate(steps) if step <= discovery_stop]
    if len(discovery) < rank + 1:
        raise ValueError("discovery window has too few states for the requested rank")
    result: list[NodeBasis] = []
    for parameter, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        if match is None:
            raise ValueError(f"unsupported parameter: {parameter}")
        target = match.group("target")
        positions = torch.stack(tensors).to(device=device, dtype=torch.float32)
        discovery_rows, discovery_weights, _ = basis_from_positions(
            positions[discovery], rank=rank, target=target
        )
        full_rows, full_weights, retained = basis_from_positions(
            positions, rank=rank, target=target
        )
        result.append(
            NodeBasis(
                parameter=parameter,
                target=target,
                discovery_rows=discovery_rows.cpu(),
                discovery_weights=discovery_weights.cpu(),
                full_rows=full_rows.cpu(),
                full_weights=full_weights.cpu(),
                retained_full_energy=retained,
            )
        )
        del positions
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return steps, result, metadata


def concatenate_learning_rows(nodes: list[NodeBasis], *, target: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [node for node in nodes if target is None or node.target == target]
    rows = torch.cat([node.discovery_rows for node in selected], dim=0)
    weights = torch.cat(
        [node.discovery_weights / len(selected) for node in selected], dim=0
    )
    return rows, weights


def role_private_atoms(nodes: list[NodeBasis], *, device: str) -> torch.Tensor:
    rows, weights = concatenate_learning_rows(nodes)
    common = learned_sign_atoms(rows.to(device), weights.to(device), 1)
    role_atoms: list[torch.Tensor] = []
    for target in ("mlp.c_fc", "mlp.c_proj"):
        role_rows, role_weights = concatenate_learning_rows(nodes, target=target)
        role_rows = role_rows.to(device)
        dots = role_rows @ common.T
        residual = role_rows - dots @ common
        role_atoms.append(
            learned_sign_atoms(residual, role_weights.to(device), 1)
        )
    return torch.cat((common, *role_atoms), dim=0).cpu()


def pack_atoms(path: Path, atoms: dict[str, torch.Tensor]) -> dict[str, Any]:
    payload: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {}
    for name, tensor in atoms.items():
        bits = (tensor.flatten().numpy() > 0).astype(np.uint8)
        packed = np.packbits(bits, bitorder="little")
        payload[name] = packed
        manifest[name] = {
            "shape": list(tensor.shape),
            "logical_bits": int(bits.size),
            "packed_bytes": int(packed.size),
            "unpacked_sign_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
        }
    np.savez_compressed(path, **payload)
    return manifest


def synthetic_self_check(device: str) -> float:
    generator = torch.Generator(device="cpu").manual_seed(71)
    atoms = normalized_sign(torch.randn(3, 48, generator=generator)).to(device)
    support = procedural_support(48, 7, seed=17, device=device)
    coefficients = torch.randn(5, 3, generator=generator).to(device)
    coordinate = torch.zeros(5, 48, device=device)
    coordinate[:, support] = torch.randn(5, 7, generator=generator).to(device)
    rows = coefficients @ atoms + coordinate
    return aggregate_rows_capture(rows, atoms, support)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--run-a-probe-dir", required=True, type=Path)
    parser.add_argument("--run-b-probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,6,11")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--bank-sizes", default="1,2,3")
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--late-start", type=int, default=180)
    parser.add_argument("--deployment-matrix-count", type=int, default=24)
    parser.add_argument("--maximum-byte-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {value for value in args.targets.split(",") if value}
    bank_sizes = parse_int_list(args.bank_sizes)
    if bank_sizes != [1, 2, 3]:
        raise ValueError("the frozen H1 discriminator requires bank sizes 1,2,3")
    if args.maximum_byte_fraction != 0.01:
        raise ValueError("the frozen H1 discriminator requires a one-percent byte gate")
    torch.manual_seed(args.seed)

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
    ambient = nodes[0].full_rows.shape[1]
    if any(node.full_rows.shape[1] != ambient for node in nodes):
        raise ValueError("all oriented matrices must share one ambient dimension")
    self_check = synthetic_self_check(args.device)
    if self_check < 0.999:
        raise ValueError(f"synthetic union projection failed: {self_check}")

    learning_rows, learning_weights = concatenate_learning_rows(nodes)
    learned_common_full = learned_sign_atoms(
        learning_rows.to(args.device),
        learning_weights.to(args.device),
        max(bank_sizes),
    ).cpu()
    learned_common = {
        count: learned_common_full[:count].clone()
        for count in bank_sizes
    }
    learned_role = role_private_atoms(nodes, device=args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_common = {
        count: normalized_sign(torch.randn(count, ambient, generator=generator))
        for count in bank_sizes
    }
    per_node_atoms = {
        node.parameter: learned_sign_atoms(
            node.discovery_rows.to(args.device),
            node.discovery_weights.to(args.device),
            3,
        ).cpu()
        for node in nodes
    }

    families: list[Family] = []
    empty_active = {target: tuple() for target in targets}
    families.append(
        Family(
            name="procedural_residual_only",
            atoms=torch.empty(0, ambient),
            active=empty_active,
            residual=True,
            fixed_bits=0,
            local_coefficients_per_matrix=0,
        )
    )
    for count in bank_sizes:
        active = {target: tuple(range(count)) for target in targets}
        families.extend(
            (
                Family(
                    name=f"random_common_k{count}_with_residual",
                    atoms=random_common[count],
                    active=active,
                    residual=True,
                    fixed_bits=count * ambient,
                    local_coefficients_per_matrix=count,
                ),
                Family(
                    name=f"learned_common_k{count}_no_residual",
                    atoms=learned_common[count],
                    active=active,
                    residual=False,
                    fixed_bits=count * ambient,
                    local_coefficients_per_matrix=count,
                ),
                Family(
                    name=f"learned_common_k{count}_with_residual",
                    atoms=learned_common[count],
                    active=active,
                    residual=True,
                    fixed_bits=count * ambient,
                    local_coefficients_per_matrix=count,
                ),
            )
        )
    role_active = {"mlp.c_fc": (0, 1), "mlp.c_proj": (0, 2)}
    families.extend(
        (
            Family(
                name="learned_role_private_k3_no_residual",
                atoms=learned_role,
                active=role_active,
                residual=False,
                fixed_bits=3 * ambient,
                local_coefficients_per_matrix=2,
            ),
            Family(
                name="learned_role_private_k3_with_residual",
                atoms=learned_role,
                active=role_active,
                residual=True,
                fixed_bits=3 * ambient,
                local_coefficients_per_matrix=2,
            ),
        )
    )

    accounting: dict[str, dict[str, float | int | bool]] = {}
    for family in families:
        residual_count, record = residual_budget_per_matrix(
            dense_scalars_per_matrix=ambient,
            deployment_matrix_count=args.deployment_matrix_count,
            maximum_fraction=args.maximum_byte_fraction,
            fixed_bits=family.fixed_bits,
            coefficients_per_matrix=family.local_coefficients_per_matrix,
            residual=family.residual,
        )
        accounting[family.name] = {
            **record,
            "premise_control": family.premise_control,
        }

    summary_rows: list[dict[str, Any]] = []
    late_rows: list[dict[str, Any]] = []
    for family in families:
        record = accounting[family.name]
        residual_count = int(record["residual_coordinates_per_matrix"])
        for node in nodes:
            active = family.active[node.target]
            atoms = (
                family.atoms[list(active)].to(args.device)
                if active
                else torch.empty(0, ambient, device=args.device)
            )
            support = procedural_support(
                ambient,
                residual_count,
                seed=stable_seed(f"{family.name}:{node.parameter}", args.seed),
                device=args.device,
            )
            rows = node.full_rows.to(args.device)
            captures = union_capture(rows, atoms, support)
            weighted, minimum, maximum = weighted_summary(
                captures, node.full_weights.to(args.device)
            )
            summary_rows.append(
                {
                    "family": family.name,
                    "parameter": node.parameter,
                    "target": node.target,
                    "active_global_atoms": len(active),
                    "weighted_top16_pc_capture": weighted,
                    "minimum_pc_capture": minimum,
                    "maximum_pc_capture": maximum,
                    "top16_retained_path_energy": node.retained_full_energy,
                    "full_path_weighted_capture_lower_bound": weighted * node.retained_full_energy,
                    "total_checkpoint_byte_fraction": record["total_checkpoint_byte_fraction"],
                }
            )

    # Reload one node at a time for the untouched late-update field.
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    for layer in sorted(layers):
        node_steps, values, _metadata = load_snapshots(paths, layers={layer}, targets=targets)
        late_indices = [index for index, step in enumerate(node_steps[1:]) if step >= args.late_start]
        for parameter, tensors in sorted(values.items()):
            match = PARAMETER_PATTERN.match(parameter)
            assert match is not None
            target = match.group("target")
            positions = orient(target, torch.stack(tensors).to(args.device, torch.float32))
            updates = (positions[1:] - positions[:-1]).flatten(1)[late_indices]
            centered = updates - updates.mean(dim=0, keepdim=True)
            for family in families:
                record = accounting[family.name]
                active = family.active[target]
                atoms = (
                    family.atoms[list(active)].to(args.device)
                    if active
                    else torch.empty(0, ambient, device=args.device)
                )
                support = procedural_support(
                    ambient,
                    int(record["residual_coordinates_per_matrix"]),
                    seed=stable_seed(f"{family.name}:{parameter}", args.seed),
                    device=args.device,
                )
                late_rows.append(
                    {
                        "family": family.name,
                        "parameter": parameter,
                        "target": target,
                        "late_start": args.late_start,
                        "late_update_count": updates.shape[0],
                        "uncentered_late_update_capture": aggregate_rows_capture(updates, atoms, support),
                        "centered_late_update_capture": aggregate_rows_capture(centered, atoms, support),
                        "total_checkpoint_byte_fraction": record["total_checkpoint_byte_fraction"],
                    }
                )
            del positions, updates, centered
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    # Multi-manifold heldout control at layer 6.
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
        common = common - common.mean(dim=0, keepdim=True)
        innovation = innovation - innovation.mean(dim=0, keepdim=True)
        for family in families:
            record = accounting[family.name]
            active = family.active[target]
            atoms = (
                family.atoms[list(active)].to(args.device)
                if active
                else torch.empty(0, ambient, device=args.device)
            )
            support = procedural_support(
                ambient,
                int(record["residual_coordinates_per_matrix"]),
                seed=stable_seed(f"{family.name}:{parameter}", args.seed),
                device=args.device,
            )
            multimanifold_rows.append(
                {
                    "family": family.name,
                    "parameter": parameter,
                    "target": target,
                    "common_centered_capture": aggregate_rows_capture(common, atoms, support),
                    "heldout_b_innovation_centered_capture": aggregate_rows_capture(innovation, atoms, support),
                    "total_checkpoint_byte_fraction": record["total_checkpoint_byte_fraction"],
                }
            )
        del a, b, da, db, common, innovation
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Per-node sign upper bound is deliberately outside the shared-bank budget.
    upper_rows: list[dict[str, Any]] = []
    for node in nodes:
        atoms = per_node_atoms[node.parameter].to(args.device)
        captures = project_rows(node.full_rows.to(args.device), atoms)
        weighted, minimum, maximum = weighted_summary(
            captures, node.full_weights.to(args.device)
        )
        upper_rows.append(
            {
                "family": "per_node_binary_k3_upper_bound",
                "parameter": node.parameter,
                "target": node.target,
                "weighted_top16_pc_capture": weighted,
                "minimum_pc_capture": minimum,
                "maximum_pc_capture": maximum,
                "premise_control": True,
                "reason": "stores three independent ambient sign atoms per matrix",
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    paths_out = {
        "summary": args.output / "pc_capture.csv",
        "late": args.output / "late_update_capture.csv",
        "multimanifold": args.output / "multimanifold_capture.csv",
        "upper": args.output / "per_node_upper_bound.csv",
        "accounting": args.output / "accounting.json",
        "banks": args.output / "global_sign_banks.npz",
    }
    write_csv(paths_out["summary"], summary_rows)
    write_csv(paths_out["late"], late_rows)
    write_csv(paths_out["multimanifold"], multimanifold_rows)
    write_csv(paths_out["upper"], upper_rows)
    paths_out["accounting"].write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n"
    )
    bank_manifest = pack_atoms(
        paths_out["banks"],
        {
            **{f"learned_common_k{count}": atoms for count, atoms in learned_common.items()},
            "learned_role_private_k3": learned_role,
        },
    )

    family_gates: dict[str, Any] = {}
    for family in families:
        pc = [row for row in summary_rows if row["family"] == family.name]
        late = [row for row in late_rows if row["family"] == family.name]
        multi = [row for row in multimanifold_rows if row["family"] == family.name]
        minimum_role_weighted = {
            target: min(
                float(row["weighted_top16_pc_capture"])
                for row in pc
                if row["target"] == target
            )
            for target in sorted(targets)
        }
        gate = {
            "minimum_role_weighted_pc_capture": minimum_role_weighted,
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
            "retained": (
                all(value >= 0.20 for value in minimum_role_weighted.values())
                and min(float(row["minimum_pc_capture"]) for row in pc) >= 0.05
                and min(float(row["common_centered_capture"]) for row in multi) >= 0.50
                and min(float(row["heldout_b_innovation_centered_capture"]) for row in multi) >= 0.20
                and min(float(row["centered_late_update_capture"]) for row in late) >= 0.05
            ),
        }
        family_gates[family.name] = gate

    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_global_sign_bank_v1",
        "method": "discovery-fit bitpacked global sign atoms plus procedural coordinate residual",
        "steps": steps,
        "layers": sorted(layers),
        "targets": sorted(targets),
        "basis_rank": args.basis_rank,
        "discovery_stop": args.discovery_stop,
        "late_start": args.late_start,
        "deployment_matrix_count": args.deployment_matrix_count,
        "maximum_checkpoint_byte_fraction": args.maximum_byte_fraction,
        "synthetic_union_capture": self_check,
        "snapshot_metadata": snapshot_metadata,
        "run_a_metadata": metadata_a,
        "run_b_metadata": metadata_b,
        "bank_manifest": bank_manifest,
        "family_gates": family_gates,
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
            for name, path in paths_out.items()
        },
        "limitations": [
            "The sign bank is frozen after discovery-window PCA fitting.",
            "The local residual is an exact coordinate-span control, not yet a fast deployed sparse-expander kernel.",
            "FP16-equivalent byte accounting is memory accounting, not latent-scalar compression.",
            "Euclidean residual capture is necessary but not sufficient for task-metric/JVP or CE closure.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "family_gates": family_gates}, sort_keys=True))


if __name__ == "__main__":
    main()
