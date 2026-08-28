#!/usr/bin/env python3
"""Analyze reusable structure in the residual missed by the H29d tangent."""
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

from examples.nanogpt.analyze_mlp_global_sign_bank import project_rows
from examples.nanogpt.analyze_mlp_synthetic_muon_program import (
    build_dense_model,
    initialization_match,
    latent_accounting,
    make_prompt,
    project_direction,
    projection_metrics,
    sha256,
    write_csv,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_full_audit import (
    LATE_START_STEP,
    LAYER6_PARAMETERS,
    assemble_component,
    joint_principal_components,
    load_probe_inventory,
    load_trajectory_inventory,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_joint import (
    FROZEN_PARAMETERS,
    make_joint_program_function,
)


CANONICAL_SHAPE = (3072, 768)
FRAME_RANK = 32
FACTOR_RANKS = (16, 32, 64)
SIGN_ATOMS = 3
PHASE_RANK = 16
DISCOVERY_COMPONENTS = 8


@dataclass
class ResidualSample:
    path: str
    component: int
    parameter: str
    eigenvalue: float
    path_energy_fraction: float
    matrix: torch.Tensor


def canonicalize(parameter: str, matrix: torch.Tensor) -> torch.Tensor:
    if parameter.endswith("mlp.c_fc.weight"):
        result = matrix
    elif parameter.endswith("mlp.c_proj.weight"):
        result = matrix.T
    else:
        raise ValueError(f"unsupported parameter: {parameter}")
    if tuple(result.shape) != CANONICAL_SHAPE:
        raise ValueError(f"canonical shape mismatch for {parameter}: {tuple(result.shape)}")
    return result.contiguous()


def split_residual(
    residual: torch.Tensor,
    *,
    parameter_shapes: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for parameter in FROZEN_PARAMETERS:
        shape = parameter_shapes[parameter]
        count = math.prod(shape)
        matrix = residual[offset : offset + count].reshape(shape)
        result[parameter] = canonicalize(parameter, matrix)
        offset += count
    if offset != residual.numel():
        raise ValueError("residual layout mismatch")
    return result


def collect_path_residuals(
    *,
    path_name: str,
    states: dict[str, torch.Tensor],
    function: Any,
    primals: tuple[torch.Tensor, ...],
    parameter_shapes: dict[str, tuple[int, ...]],
    component_count: int,
    cg_iterations: int,
    cg_tolerance: float,
    relative_ridge: float,
    device: str,
    projection_rows: list[dict[str, Any]],
    progress_path: Path,
    completed_before: int,
    total_solves: int,
) -> tuple[list[ResidualSample], dict[str, Any], int]:
    build_started = time.time()
    bundle = joint_principal_components(
        states,
        parameter_order=FROZEN_PARAMETERS,
        component_count=component_count,
        device=device,
    )
    build_seconds = time.time() - build_started
    eigenvalues: torch.Tensor = bundle["eigenvalues"]
    total_energy = float(bundle["total_energy"])
    samples: list[ResidualSample] = []
    completed = completed_before
    weighted_capture = 0.0
    component_captures: list[float] = []
    for component in range(component_count):
        target = assemble_component(
            bundle,
            component=component,
            parameter_order=FROZEN_PARAMETERS,
            parameter_shapes=parameter_shapes,
            device=device,
        )
        projected, diagnostics = project_direction(
            function,
            primals,
            target,
            cg_iterations=cg_iterations,
            cg_tolerance=cg_tolerance,
            relative_ridge=relative_ridge,
        )
        metrics = projection_metrics(target, projected)
        eigenvalue = float(eigenvalues[component])
        energy_fraction = eigenvalue / total_energy
        weighted_capture += energy_fraction * metrics["path_energy_capture"]
        component_captures.append(metrics["path_energy_capture"])
        residual = target - projected
        matrices = split_residual(residual, parameter_shapes=parameter_shapes)
        for parameter, matrix in matrices.items():
            samples.append(
                ResidualSample(
                    path=path_name,
                    component=component + 1,
                    parameter=parameter,
                    eigenvalue=eigenvalue,
                    path_energy_fraction=energy_fraction,
                    matrix=matrix.detach().cpu().to(torch.float16).contiguous(),
                )
            )
        projection_rows.append(
            {
                "path": path_name,
                "component": component + 1,
                "eigenvalue": eigenvalue,
                "path_energy_fraction": energy_fraction,
                **metrics,
                **diagnostics,
            }
        )
        completed += 1
        progress_path.write_text(
            json.dumps(
                {
                    "state": "projecting",
                    "completed": completed,
                    "total": total_solves,
                    "path": path_name,
                    "component": component + 1,
                },
                sort_keys=True,
            )
            + "\n"
        )
        del target, projected, residual, matrices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "state_count": int(bundle["state_count"]),
        "component_count": component_count,
        "pca_build_seconds": build_seconds,
        "total_path_energy": total_energy,
        "top_k_energy_fraction": float(eigenvalues[:component_count].sum()) / total_energy,
        "weighted_projection_capture": weighted_capture,
        "minimum_projection_capture": min(component_captures),
        "maximum_projection_capture": max(component_captures),
    }
    del bundle
    return samples, summary, completed


def entropy(probabilities: torch.Tensor) -> float:
    values = probabilities.double().clamp_min(1e-30)
    values = values / values.sum().clamp_min(1e-30)
    return float(-(values * values.log()).sum())


def energy_rank(energy: torch.Tensor, threshold: float) -> int:
    probabilities = energy.double() / energy.double().sum().clamp_min(1e-30)
    return int(
        torch.searchsorted(
            probabilities.cumsum(0),
            torch.tensor(threshold, dtype=torch.float64, device=probabilities.device),
        ).item()
        + 1
    )


def canonical_overlap(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, float]:
    singular = torch.linalg.svdvals(left.T.float() @ right.float()).double().square()
    return float(singular.mean()), float(singular.min()), float(singular.max())


def sample_structure_and_factors(
    samples: list[ResidualSample],
    *,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    factors: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sample in samples:
        matrix = sample.matrix.to(device, dtype=torch.float32)
        left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
        energy = singular.double().square()
        total = energy.sum().clamp_min(1e-30)
        probabilities = energy / total
        entry = matrix.double().flatten()
        centered = entry - entry.mean()
        variance = centered.square().mean().clamp_min(1e-30)
        row_energy = matrix.double().square().sum(1)
        column_energy = matrix.double().square().sum(0)
        rows.append(
            {
                "path": sample.path,
                "component": sample.component,
                "parameter": sample.parameter,
                "eigenvalue": sample.eigenvalue,
                "residual_energy": float(total),
                "rank_90": energy_rank(energy, 0.90),
                "rank_95": energy_rank(energy, 0.95),
                "rank_99": energy_rank(energy, 0.99),
                "stable_rank": float(total / energy[0].clamp_min(1e-30)),
                "entropy_rank": math.exp(entropy(probabilities)),
                "row_leverage_entropy": entropy(row_energy),
                "column_leverage_entropy": entropy(column_energy),
                "positive_fraction": float((entry >= 0).double().mean()),
                "magnitude_kurtosis": float(centered.pow(4).mean() / variance.square()),
            }
        )
        factors.setdefault((sample.path, sample.component), []).append(
            {
                "parameter": sample.parameter,
                "left": left[:, : max(FACTOR_RANKS)].detach().cpu(),
                "right": right_h[: max(FACTOR_RANKS)].T.detach().cpu(),
            }
        )
        del matrix, left, singular, right_h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overlap_rows: list[dict[str, Any]] = []
    for (path, component), group in sorted(factors.items()):
        for first_index, first in enumerate(group):
            for second in group[first_index + 1 :]:
                for rank in FACTOR_RANKS:
                    left_stats = canonical_overlap(first["left"][:, :rank], second["left"][:, :rank])
                    right_stats = canonical_overlap(first["right"][:, :rank], second["right"][:, :rank])
                    overlap_rows.append(
                        {
                            "path": path,
                            "component": component,
                            "parameter_a": first["parameter"],
                            "parameter_b": second["parameter"],
                            "rank": rank,
                            "left_mean_squared_overlap": left_stats[0],
                            "left_minimum_squared_overlap": left_stats[1],
                            "left_maximum_squared_overlap": left_stats[2],
                            "right_mean_squared_overlap": right_stats[0],
                            "right_minimum_squared_overlap": right_stats[1],
                            "right_maximum_squared_overlap": right_stats[2],
                        }
                    )
    return rows, overlap_rows


def stack_rows(samples: list[ResidualSample], device: str) -> torch.Tensor:
    return torch.stack([sample.matrix.flatten() for sample in samples]).to(device, dtype=torch.float32)


def sample_weights(samples: list[ResidualSample], device: str) -> torch.Tensor:
    return torch.tensor([sample.eigenvalue for sample in samples], device=device, dtype=torch.float64)


def fit_dense_atoms(
    samples: list[ResidualSample],
    *,
    count: int,
    device: str,
) -> torch.Tensor:
    rows = stack_rows(samples, device)
    weights = sample_weights(samples, device)
    weights = weights / weights.sum().clamp_min(1e-30)
    weighted = rows.double() * weights.sqrt().unsqueeze(1)
    gram = weighted @ weighted.T
    gram = (gram + gram.T) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:count]
    atoms = (vectors[:, order].T @ weighted) / eigenvalues[order].sqrt().clamp_min(1e-30).unsqueeze(1)
    del rows, weights, weighted, gram, eigenvalues, vectors
    return atoms.float().contiguous()


def normalized_sign_atoms(dense_atoms: torch.Tensor) -> torch.Tensor:
    signs = torch.where(dense_atoms >= 0, torch.ones_like(dense_atoms), -torch.ones_like(dense_atoms))
    return signs / math.sqrt(dense_atoms.shape[1])


def atom_hash(atoms: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for atom in atoms.detach().cpu():
        packed = np.packbits((atom.numpy() >= 0).astype(np.uint8), bitorder="little")
        digest.update(packed.tobytes())
    return digest.hexdigest()


def aggregate_atom_capture(
    samples: list[ResidualSample],
    atoms: torch.Tensor,
    *,
    device: str,
) -> dict[str, float]:
    rows = stack_rows(samples, device)
    captures = project_rows(rows, atoms)
    energies = rows.double().square().sum(1)
    weights = sample_weights(samples, device) * energies
    normalized = weights / weights.sum().clamp_min(1e-30)
    result = {
        "weighted_capture": float((normalized * captures.double()).sum()),
        "minimum_sample_capture": float(captures.min()),
        "maximum_sample_capture": float(captures.max()),
    }
    del rows, captures, energies, weights, normalized
    return result


def fit_shared_frames(
    samples: list[ResidualSample],
    *,
    rank: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("shared-frame fit needs samples")
    rows, columns = samples[0].matrix.shape
    if any(tuple(sample.matrix.shape) != (rows, columns) for sample in samples):
        raise ValueError("shared-frame samples have different shapes")
    if rank > min(rows, columns):
        raise ValueError("shared-frame rank exceeds matrix shape")
    left_covariance = torch.zeros(rows, rows, device=device)
    right_covariance = torch.zeros(columns, columns, device=device)
    weight_total = sum(sample.eigenvalue for sample in samples)
    for sample in samples:
        matrix = sample.matrix.to(device, dtype=torch.float32)
        weight = sample.eigenvalue / max(weight_total, 1e-30)
        left_covariance.addmm_(matrix, matrix.T, beta=1.0, alpha=weight)
        right_covariance.addmm_(matrix.T, matrix, beta=1.0, alpha=weight)
        del matrix
    left_values, left_vectors = torch.linalg.eigh((left_covariance + left_covariance.T) * 0.5)
    right_values, right_vectors = torch.linalg.eigh((right_covariance + right_covariance.T) * 0.5)
    del left_covariance, right_covariance, left_values, right_values
    return left_vectors[:, -rank:].contiguous(), right_vectors[:, -rank:].contiguous()


def aggregate_frame_capture(
    samples: list[ResidualSample],
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    device: str,
) -> dict[str, float]:
    numerator_left = 0.0
    numerator_right = 0.0
    numerator_bilateral = 0.0
    denominator = 0.0
    minimum_bilateral = 1.0
    for sample in samples:
        matrix = sample.matrix.to(device, dtype=torch.float32)
        weight = sample.eigenvalue
        total = float(matrix.double().square().sum())
        left_energy = float((left.T @ matrix).double().square().sum())
        right_energy = float((matrix @ right).double().square().sum())
        bilateral_energy = float((left.T @ matrix @ right).double().square().sum())
        denominator += weight * total
        numerator_left += weight * left_energy
        numerator_right += weight * right_energy
        numerator_bilateral += weight * bilateral_energy
        minimum_bilateral = min(minimum_bilateral, bilateral_energy / max(total, 1e-30))
        del matrix
    return {
        "left_capture": numerator_left / max(denominator, 1e-30),
        "right_capture": numerator_right / max(denominator, 1e-30),
        "bilateral_capture": numerator_bilateral / max(denominator, 1e-30),
        "minimum_sample_bilateral_capture": minimum_bilateral,
    }


def path_groups(samples: list[ResidualSample]) -> dict[str, list[ResidualSample]]:
    result: dict[str, list[ResidualSample]] = {}
    for sample in samples:
        result.setdefault(sample.path, []).append(sample)
    return result


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(20260904)
    shared = torch.randn(8, 6)
    samples = [
        ResidualSample("joint_centered", index + 1, f"node{index}", 1.0, 0.25, shared + 0.01 * torch.randn_like(shared))
        for index in range(4)
    ]
    atoms = fit_dense_atoms(samples, count=1, device=device)
    capture = aggregate_atom_capture(samples, atoms, device=device)["weighted_capture"]
    signed = normalized_sign_atoms(atoms)
    if capture < 0.999 or signed.shape != atoms.shape:
        raise AssertionError((capture, signed.shape))
    left, right = fit_shared_frames(samples, rank=2, device=device)
    frame = aggregate_frame_capture(samples, left, right, device=device)
    if not 0.0 <= frame["bilateral_capture"] <= 1.00001:
        raise AssertionError(frame)
    return {"status": "passed", "dense_atom_capture": capture, **frame}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--run-a-probe-dir", type=Path)
    parser.add_argument("--run-b-probe-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--cg-iterations", type=int, default=20)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    required = (args.config, args.plan, args.trajectory_dir, args.output)
    if any(value is None for value in required):
        parser.error("config, plan, trajectory, and output are required")
    if not args.preflight and any(value is None for value in (args.run_a_probe_dir, args.run_b_probe_dir)):
        parser.error("binding residual audit requires A/B probe directories")
    expected = (2, 4) if args.preflight else (16, 20)
    if (args.components, args.cg_iterations) != expected:
        raise ValueError(f"mode requires components/CG={expected}")
    assert args.config is not None and args.plan is not None and args.output is not None
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "progress.json"
    total_solves = 2 if args.preflight else 64
    progress_path.write_text(json.dumps({"state": "starting", "completed": 0, "total": total_solves}) + "\n")
    started = time.time()
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    model = build_dense_model(config, args.device)
    prompt, targets, prompt_manifest = make_prompt(model, config, prompt_length=737, device=args.device)
    function, function_manifest = make_joint_program_function(
        model, parameters=FROZEN_PARAMETERS, targets=targets, ns_steps=5
    )
    parameter_shapes = {name: tuple(dict(model.named_parameters())[name].shape) for name in FROZEN_PARAMETERS}
    primals = (prompt, torch.ones(len(FROZEN_PARAMETERS), device=args.device))
    accounting = latent_accounting(737, 768)
    torch.cuda.reset_peak_memory_stats()

    trajectory, trajectory_identity = load_trajectory_inventory(args.trajectory_dir, FROZEN_PARAMETERS)
    w0_matches = {
        name: initialization_match(dict(model.named_parameters())[name], values[0])
        for name, values in trajectory.items()
    }
    if not all(bool(record["accepted"]) for record in w0_matches.values()):
        raise ValueError("model/trajectory W0 mismatch")
    w0_references = {name: values[0].clone() for name, values in trajectory.items()}

    projection_rows: list[dict[str, Any]] = []
    all_samples: list[ResidualSample] = []
    path_summaries: dict[str, Any] = {}
    completed = 0
    joint_samples, path_summaries["joint_centered"], completed = collect_path_residuals(
        path_name="joint_centered",
        states=trajectory,
        function=function,
        primals=primals,
        parameter_shapes=parameter_shapes,
        component_count=args.components,
        cg_iterations=args.cg_iterations,
        cg_tolerance=1e-5,
        relative_ridge=1e-6,
        device=args.device,
        projection_rows=projection_rows,
        progress_path=progress_path,
        completed_before=completed,
        total_solves=total_solves,
    )
    all_samples.extend(joint_samples)
    identities: dict[str, Any] = {"trajectory": trajectory_identity}
    if not args.preflight:
        late = {name: values[LATE_START_STEP:] for name, values in trajectory.items()}
        late_samples, path_summaries["joint_late"], completed = collect_path_residuals(
            path_name="joint_late",
            states=late,
            function=function,
            primals=primals,
            parameter_shapes=parameter_shapes,
            component_count=args.components,
            cg_iterations=args.cg_iterations,
            cg_tolerance=1e-5,
            relative_ridge=1e-6,
            device=args.device,
            projection_rows=projection_rows,
            progress_path=progress_path,
            completed_before=completed,
            total_solves=total_solves,
        )
        all_samples.extend(late_samples)
        del late, trajectory

        assert args.run_a_probe_dir is not None and args.run_b_probe_dir is not None
        run_a, steps_a, identity_a = load_probe_inventory(args.run_a_probe_dir, LAYER6_PARAMETERS)
        run_b, steps_b, identity_b = load_probe_inventory(args.run_b_probe_dir, LAYER6_PARAMETERS)
        if steps_a != steps_b:
            raise ValueError("A/B schedules differ")
        for parameter in LAYER6_PARAMETERS:
            if not torch.equal(run_a[parameter][0], run_b[parameter][0]):
                raise ValueError("A/B step-zero mismatch")
            if not torch.equal(run_a[parameter][0], w0_references[parameter]):
                raise ValueError("trajectory/probe step-zero mismatch")
        identities.update({"run_a": identity_a, "run_b": identity_b, "probe_steps": steps_a})
        common = {parameter: 0.5 * (run_a[parameter].float() + run_b[parameter].float()) for parameter in LAYER6_PARAMETERS}
        common_samples, path_summaries["layer6_common"], completed = collect_path_residuals(
            path_name="layer6_common",
            states=common,
            function=function,
            primals=primals,
            parameter_shapes=parameter_shapes,
            component_count=args.components,
            cg_iterations=args.cg_iterations,
            cg_tolerance=1e-5,
            relative_ridge=1e-6,
            device=args.device,
            projection_rows=projection_rows,
            progress_path=progress_path,
            completed_before=completed,
            total_solves=total_solves,
        )
        all_samples.extend(common_samples)
        del common
        innovation = {parameter: 0.5 * (run_b[parameter].float() - run_a[parameter].float()) for parameter in LAYER6_PARAMETERS}
        innovation_samples, path_summaries["layer6_stream_b_innovation"], completed = collect_path_residuals(
            path_name="layer6_stream_b_innovation",
            states=innovation,
            function=function,
            primals=primals,
            parameter_shapes=parameter_shapes,
            component_count=args.components,
            cg_iterations=args.cg_iterations,
            cg_tolerance=1e-5,
            relative_ridge=1e-6,
            device=args.device,
            projection_rows=projection_rows,
            progress_path=progress_path,
            completed_before=completed,
            total_solves=total_solves,
        )
        all_samples.extend(innovation_samples)
        del innovation, run_a, run_b
    else:
        del trajectory

    progress_path.write_text(json.dumps({"state": "structure", "completed": completed, "total": total_solves}) + "\n")
    structure_rows, overlap_rows = sample_structure_and_factors(all_samples, device=args.device)
    groups = path_groups(all_samples)

    carrier_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    phase_summary: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    if not args.preflight:
        discovery = [sample for sample in groups["joint_centered"] if sample.component <= DISCOVERY_COMPONENTS]
        evaluation_groups = {
            "joint_holdout": [sample for sample in groups["joint_centered"] if sample.component > DISCOVERY_COMPONENTS],
            "joint_late": groups["joint_late"],
            "layer6_common": groups["layer6_common"],
            "layer6_stream_b_innovation": groups["layer6_stream_b_innovation"],
        }
        dense_atoms = fit_dense_atoms(discovery, count=SIGN_ATOMS, device=args.device)
        sign_atoms = normalized_sign_atoms(dense_atoms)
        sign_hash = atom_hash(sign_atoms)
        for family_name, family_atoms in (("dense_control", dense_atoms), ("sign3", sign_atoms)):
            for path_name, samples in {"discovery": discovery, **evaluation_groups}.items():
                carrier_rows.append(
                    {"family": family_name, "path": path_name, **aggregate_atom_capture(samples, family_atoms, device=args.device)}
                )
        left_frame, right_frame = fit_shared_frames(discovery, rank=FRAME_RANK, device=args.device)
        for path_name, samples in {"discovery": discovery, **evaluation_groups}.items():
            frame_rows.append({"path": path_name, **aggregate_frame_capture(samples, left_frame, right_frame, device=args.device)})

        joint_phase_atoms = fit_dense_atoms(groups["joint_centered"], count=PHASE_RANK, device=args.device)
        late_phase_atoms = fit_dense_atoms(groups["joint_late"], count=PHASE_RANK, device=args.device)
        phase_summary = {
            "joint_on_joint": aggregate_atom_capture(groups["joint_centered"], joint_phase_atoms, device=args.device)["weighted_capture"],
            "joint_on_late": aggregate_atom_capture(groups["joint_late"], joint_phase_atoms, device=args.device)["weighted_capture"],
            "late_on_late": aggregate_atom_capture(groups["joint_late"], late_phase_atoms, device=args.device)["weighted_capture"],
            "late_on_joint": aggregate_atom_capture(groups["joint_centered"], late_phase_atoms, device=args.device)["weighted_capture"],
        }
        sign_fraction = SIGN_ATOMS * math.prod(CANONICAL_SHAPE) / (16 * 56_623_104)
        sign_required = [row for row in carrier_rows if row["family"] == "sign3" and row["path"] in evaluation_groups]
        rank32_overlap = [row for row in overlap_rows if row["rank"] == FRAME_RANK and row["path"] == "joint_centered" and row["component"] <= DISCOVERY_COMPONENTS]
        left_overlap = sum(row["left_mean_squared_overlap"] for row in rank32_overlap) / len(rank32_overlap)
        right_overlap = sum(row["right_mean_squared_overlap"] for row in rank32_overlap) / len(rank32_overlap)
        heldout_frames = [row for row in frame_rows if row["path"] in evaluation_groups]
        gates = {
            "sign_checkpoint_fraction": sign_fraction,
            "sign_minimum_family_capture": min(row["weighted_capture"] for row in sign_required),
            "sign_minimum_enrichment": min(row["weighted_capture"] / sign_fraction for row in sign_required),
            "H29f_authorized": all(row["weighted_capture"] >= 0.20 and row["weighted_capture"] / sign_fraction >= 5.0 for row in sign_required),
            "rank32_left_mean_overlap": left_overlap,
            "rank32_right_mean_overlap": right_overlap,
            "frame_minimum_bilateral_capture": min(row["bilateral_capture"] for row in heldout_frames),
            "H29g_authorized": left_overlap >= 0.50 and right_overlap >= 0.50 and all(row["bilateral_capture"] >= 0.20 for row in heldout_frames),
            "H29h_authorized": phase_summary["joint_on_joint"] >= 0.50 and phase_summary["late_on_late"] >= 0.50 and phase_summary["joint_on_late"] < 0.20 and phase_summary["late_on_joint"] < 0.20,
            "sign_atom_bit_sha256": sign_hash,
        }
        classification = "HYBRID_AUTHORIZED" if any(bool(gates[key]) for key in ("H29f_authorized", "H29g_authorized", "H29h_authorized")) else "NO_REUSABLE_CARRIER"
        del dense_atoms, sign_atoms, left_frame, right_frame, joint_phase_atoms, late_phase_atoms
    else:
        classification = "PREFLIGHT"

    projection_path = output / "projection.csv"
    structure_path = output / "matrix_structure.csv"
    overlap_path = output / "cross_node_overlap.csv"
    carrier_path = output / "carrier_summary.csv"
    frame_path = output / "frame_summary.csv"
    summary_path = output / "summary.json"
    accounting_path = output / "accounting.json"
    write_csv(projection_path, projection_rows)
    write_csv(structure_path, structure_rows)
    write_csv(overlap_path, overlap_rows)
    if carrier_rows:
        write_csv(carrier_path, carrier_rows)
        write_csv(frame_path, frame_rows)
    summary_path.write_text(json.dumps({"classification": classification, "paths": path_summaries, "phase": phase_summary, "gates": gates}, indent=2, sort_keys=True) + "\n")
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    progress_path.write_text(json.dumps({"state": "finished", "classification": classification, "completed": completed, "total": total_solves}, sort_keys=True) + "\n")
    torch.cuda.synchronize()
    script = Path(__file__).resolve()
    output_files = [projection_path, structure_path, overlap_path, summary_path, accounting_path, progress_path]
    if carrier_rows:
        output_files.extend([carrier_path, frame_path])
    metadata = {
        "schema_version": "nanogpt_mlp_synthetic_program_residual_structure_v1",
        "classification": classification,
        "preflight": args.preflight,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "function_manifest": function_manifest,
        "identities": identities,
        "w0_storage_matches": w0_matches,
        "path_summaries": path_summaries,
        "phase_summary": phase_summary,
        "gates": gates,
        "execution": {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
            "entrypoint": str(script),
            "entrypoint_sha256": sha256(script),
            "config_sha256": sha256(args.config),
            "plan_sha256": sha256(args.plan),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": time.time() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": args.device,
        },
        "outputs": {path.name: sha256(path) for path in output_files},
        "nonpersistent_analysis_objects": [
            "all tangent residual matrices",
            "three dense control atoms",
            "three sign atoms",
            "rank-32 shared frames",
            "rank-16 phase bases",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "finished", "classification": classification, "metadata": str(metadata_path), "gates": gates}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
