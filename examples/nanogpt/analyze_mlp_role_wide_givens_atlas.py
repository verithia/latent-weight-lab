#!/usr/bin/env python3
"""H42 top-PC/LOO gate for a wide role-conditioned shallow Givens atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from examples.nanogpt.analyze_mlp_role_givens_transport_loo import (
    fullcoverage_transport,
    make_stage_permutations,
)
from examples.nanogpt.analyze_mlp_shared_separable_conditioner_loo import (
    CANONICAL_SHAPE,
    canonicalize,
    normalized,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program import (
    DENSE_MLP_SCALARS,
    build_dense_model,
    initialization_match,
    make_prompt,
    sha256,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_full_audit import (
    joint_principal_components,
    load_trajectory_inventory,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_joint import (
    FROZEN_PARAMETERS,
)
from examples.nanogpt.analyze_mlp_virtual_lookahead_joint import (
    make_model_lookahead_program,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


PROMPT_LENGTH = 415
PROMPT_WIDTH = 768
DEPLOYED_MLP_MATRICES = 24
ROLE_GROUPS = {"c_fc": (0, 2, 4), "c_proj": (1, 3, 5)}
ROLE_SEEDS = {"c_fc": 431, "c_proj": 503}
BRANCHES = 64
STAGES = 1
BINDING_COMPONENTS = 16
PREFLIGHT_COMPONENTS = 4
BINDING_ITERATIONS = 64
PREFLIGHT_ITERATIONS = 4
LEARNING_RATE = 0.01
GRADIENT_CLIP_NORM = 10.0
RELATIVE_PROJECTION_RIDGE = 1e-4


def checkpoint_accounting() -> dict[str, int | float]:
    rows, columns = CANONICAL_SHAPE
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    angles_per_branch = rows // 2 + columns // 2
    role_angle_scalars = len(ROLE_GROUPS) * BRANCHES * angles_per_branch
    local_coefficient_scalars = DEPLOYED_MLP_MATRICES * BRANCHES
    total_scalars = (
        prompt_scalars + role_angle_scalars + local_coefficient_scalars
    )
    return {
        "prompt_scalars": prompt_scalars,
        "roles": len(ROLE_GROUPS),
        "branches_per_role": BRANCHES,
        "stages_per_branch": STAGES,
        "angles_per_branch": angles_per_branch,
        "role_angle_scalars": role_angle_scalars,
        "local_coefficient_scalars": local_coefficient_scalars,
        "total_state_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "state_fraction": total_scalars / DENSE_MLP_SCALARS,
        "fp16_checkpoint_bytes": 2 * total_scalars,
        "persistent_dense_basis_scalars": 0,
        "persistent_branch_matrix_scalars": 0,
        "persistent_permutation_scalars": 0,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def make_branch_geometry(
    shape: tuple[int, int],
    *,
    branches: int,
    seed_base: int,
    device: torch.device,
) -> dict[str, Any]:
    rows, columns = shape
    hidden_permutations = []
    residual_permutations = []
    base_hidden = []
    base_residual = []
    for branch in range(branches):
        seed = seed_base + 2 * branch
        hidden_permutations.append(
            make_stage_permutations(rows, 1, seed, device)[0]
        )
        residual_permutations.append(
            make_stage_permutations(columns, 1, seed + 1, device)[0]
        )
        if branch == 0:
            base_hidden.append(torch.zeros(rows // 2, device=device))
            base_residual.append(torch.zeros(columns // 2, device=device))
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 10_000_019)
        base_hidden.append(
            ((2.0 * math.pi) * torch.rand(rows // 2, generator=generator) - math.pi).to(device)
        )
        base_residual.append(
            ((2.0 * math.pi) * torch.rand(columns // 2, generator=generator) - math.pi).to(device)
        )
    return {
        "hidden_permutations": tuple(hidden_permutations),
        "residual_permutations": tuple(residual_permutations),
        "base_hidden": torch.stack(base_hidden),
        "base_residual": torch.stack(base_residual),
    }


def atlas_basis(
    atom: torch.Tensor,
    hidden_residual: torch.Tensor,
    residual_residual: torch.Tensor,
    geometry: dict[str, Any],
) -> torch.Tensor:
    rows = []
    for branch in range(hidden_residual.shape[0]):
        transported = fullcoverage_transport(
            atom,
            (geometry["base_hidden"][branch] + hidden_residual[branch]).unsqueeze(0),
            (geometry["base_residual"][branch] + residual_residual[branch]).unsqueeze(0),
            (geometry["hidden_permutations"][branch],),
            (geometry["residual_permutations"][branch],),
        )
        rows.append(transported.reshape(-1))
    return torch.stack(rows)


def projection_metrics(
    basis: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    relative_ridge: float = RELATIVE_PROJECTION_RIDGE,
    differentiable: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gram = basis @ basis.T
    scale = gram.diagonal().mean().clamp_min(1e-12)
    ridge = relative_ridge * scale
    regularized = gram + ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    cross = targets @ basis.T
    coefficients = torch.linalg.solve(regularized, cross.T).T
    predictions = coefficients @ basis
    numerators = (predictions * targets).sum(dim=1).square()
    denominators = predictions.square().sum(dim=1).clamp_min(1e-20)
    captures = (numerators / denominators).clamp(0.0, 1.0)
    weighted = (captures * weights).sum()
    if differentiable:
        return weighted, {}
    residual = regularized @ coefficients.T - cross.T
    return weighted, {
        "component_captures": [float(value) for value in captures],
        "weighted_topk_capture": float(weighted),
        "minimum_component_capture": float(captures.min()),
        "median_component_capture": float(captures.median()),
        "maximum_component_capture": float(captures.max()),
        "relative_ridge": relative_ridge,
        "absolute_ridge": float(ridge),
        "gram_condition_number_regularized": float(torch.linalg.cond(regularized)),
        "relative_solver_residual": float(
            residual.norm() / cross.T.norm().clamp_min(1e-20)
        ),
        "coefficient_norms": [float(value) for value in coefficients.norm(dim=1)],
    }


def load_node_pc_inventory(
    trajectory_dir: Path,
    *,
    components: int,
    device: str,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, Any], dict[str, torch.Tensor]]:
    states, identity = load_trajectory_inventory(trajectory_dir, FROZEN_PARAMETERS)
    pc_rows = []
    weight_rows = []
    manifest: dict[str, Any] = {
        "trajectory_identity_sha256": identity,
        "state_count": 239,
        "components": components,
        "parameters": {},
    }
    w0_references = {name: values[0].clone() for name, values in states.items()}
    for parameter in FROZEN_PARAMETERS:
        bundle = joint_principal_components(
            {parameter: states[parameter]},
            parameter_order=(parameter,),
            component_count=components,
            device=device,
        )
        parts = bundle["parts"][parameter]
        canonical_parts = torch.stack(
            [
                canonicalize(parameter, part.reshape(states[parameter].shape[1:]))
                for part in parts
            ]
        ).to(device=device, dtype=torch.float32)
        canonical_parts = canonical_parts / canonical_parts.flatten(1).norm(
            dim=1, keepdim=True
        ).clamp_min(1e-20).view(-1, 1, 1)
        eigenvalues = bundle["eigenvalues"][:components].float()
        observed_energy = float(eigenvalues.sum())
        total_energy = float(bundle["total_energy"])
        weights = (eigenvalues / max(observed_energy, 1e-30)).to(
            device=device, dtype=torch.float32
        )
        pc_rows.append(canonical_parts)
        weight_rows.append(weights)
        manifest["parameters"][parameter] = {
            "shape": list(states[parameter].shape[1:]),
            "canonical_shape": list(canonical_parts.shape[1:]),
            "top_component_energy_fraction": float(eigenvalues[0]) / total_energy,
            "top_k_energy_fraction": observed_energy / total_energy,
            "eigenvalues": [float(value) for value in eigenvalues],
        }
    return tuple(pc_rows), tuple(weight_rows), manifest, w0_references


def fit_role_atlas(
    atoms: tuple[torch.Tensor, ...],
    pcs: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    train_indices: tuple[int, ...],
    evaluation_indices: tuple[int, ...],
    seed_base: int,
    branches: int = BRANCHES,
    iterations: int = BINDING_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    basis_function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor
    ] = atlas_basis,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if not train_indices or not evaluation_indices:
        raise ValueError("H42 requires nonempty train and evaluation folds")
    shape = tuple(atoms[train_indices[0]].shape)
    device = atoms[train_indices[0]].device
    geometry = make_branch_geometry(
        shape, branches=branches, seed_base=seed_base, device=device
    )
    hidden_residual = torch.nn.Parameter(
        torch.zeros(branches, shape[0] // 2, device=device)
    )
    residual_residual = torch.nn.Parameter(
        torch.zeros(branches, shape[1] // 2, device=device)
    )
    parameters = [hidden_residual, residual_residual]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    history = []
    recorded_steps = {0, 1, 2, 3, 7, 15, 31, 63}
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), device=device)
        for index in train_indices:
            basis = basis_function(
                atoms[index], hidden_residual, residual_residual, geometry
            )
            weighted, _ = projection_metrics(
                basis,
                pcs[index].flatten(1),
                weights[index],
                differentiable=True,
            )
            objective = objective + weighted / len(train_indices)
        (-objective).backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        )
        optimizer.step()
        if step in recorded_steps or step == iterations - 1:
            history.append(
                {
                    "iteration": step + 1,
                    "weighted_capture_objective": float(objective.detach()),
                    "gradient_norm": gradient_norm,
                }
            )

    rows = []
    zero_hidden = torch.zeros_like(hidden_residual)
    zero_residual = torch.zeros_like(residual_residual)
    with torch.no_grad():
        for index in evaluation_indices:
            learned_basis = basis_function(
                atoms[index], hidden_residual, residual_residual, geometry
            )
            _, learned = projection_metrics(
                learned_basis,
                pcs[index].flatten(1),
                weights[index],
                differentiable=False,
            )
            base_basis = basis_function(
                atoms[index], zero_hidden, zero_residual, geometry
            )
            _, base = projection_metrics(
                base_basis,
                pcs[index].flatten(1),
                weights[index],
                differentiable=False,
            )
            rows.append(
                {
                    "index": index,
                    "learned": learned,
                    "procedural_base": base,
                    "weighted_capture_margin": learned["weighted_topk_capture"]
                    - base["weighted_topk_capture"],
                }
            )
    return (
        {
            "train_indices": list(train_indices),
            "evaluation_indices": list(evaluation_indices),
            "iterations": iterations,
            "branches": branches,
            "learning_rate": learning_rate,
            "seed_base": seed_base,
            "history": history,
            "rows": rows,
            "learned_residual_norms": {
                "hidden": float(hidden_residual.detach().norm()),
                "residual": float(residual_residual.detach().norm()),
            },
            "learned_residual_sha256": {
                "hidden": tensor_sha256(hidden_residual),
                "residual": tensor_sha256(residual_residual),
            },
        },
        hidden_residual.detach(),
        residual_residual.detach(),
    )


def summarize_role(rows: list[dict[str, Any]]) -> dict[str, float]:
    weighted = [row["learned"]["weighted_topk_capture"] for row in rows]
    minimum_pc = [row["learned"]["minimum_component_capture"] for row in rows]
    margins = [row["weighted_capture_margin"] for row in rows]
    return {
        "minimum_weighted_topk_capture": min(weighted),
        "median_weighted_topk_capture": statistics.median(weighted),
        "maximum_weighted_topk_capture": max(weighted),
        "minimum_component_capture": min(minimum_pc),
        "minimum_base_margin": min(margins),
        "median_base_margin": statistics.median(margins),
    }


def atlas_audit(
    atoms: tuple[torch.Tensor, ...],
    pcs: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    iterations: int,
    branches: int = BRANCHES,
    preflight: bool,
    basis_function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor
    ] = atlas_basis,
) -> tuple[dict[str, Any], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    roles = {}
    stored_angles = {}
    for role, group in ROLE_GROUPS.items():
        fit_all, hidden, residual = fit_role_atlas(
            atoms,
            pcs,
            weights,
            train_indices=group,
            evaluation_indices=group,
            seed_base=ROLE_SEEDS[role],
            branches=branches,
            iterations=iterations,
            basis_function=basis_function,
        )
        stored_angles[role] = (hidden, residual)
        heldouts = group[:1] if preflight else group
        loo_rows = []
        for heldout in heldouts:
            train = tuple(index for index in group if index != heldout)
            loo, _, _ = fit_role_atlas(
                atoms,
                pcs,
                weights,
                train_indices=train,
                evaluation_indices=(heldout,),
                seed_base=ROLE_SEEDS[role],
                branches=branches,
                iterations=iterations,
                basis_function=basis_function,
            )
            loo_rows.append(
                {
                    "heldout_index": heldout,
                    "train_indices": list(train),
                    **loo["rows"][0],
                    "fit_history": loo["history"],
                    "learned_residual_norms": loo["learned_residual_norms"],
                    "learned_residual_sha256": loo["learned_residual_sha256"],
                }
            )
        roles[role] = {
            "group": list(group),
            "fit_all": fit_all,
            "fit_all_summary": summarize_role(fit_all["rows"]),
            "leave_one_out_rows": loo_rows,
            "leave_one_out_complete": len(heldouts) == len(group),
            "leave_one_out_summary": summarize_role(loo_rows),
        }
    fit_pass = all(
        row["fit_all_summary"]["minimum_weighted_topk_capture"] >= 0.30
        and row["fit_all_summary"]["minimum_component_capture"] >= 0.05
        for row in roles.values()
    )
    transfer_pass = (not preflight) and all(
        row["leave_one_out_summary"]["minimum_weighted_topk_capture"] >= 0.30
        and row["leave_one_out_summary"]["minimum_component_capture"] >= 0.05
        and row["leave_one_out_summary"]["minimum_base_margin"] >= 0.05
        for row in roles.values()
    )
    return {
        "roles": roles,
        "fit_all_pass": fit_pass,
        "transfer_pass": transfer_pass,
        "retained": fit_pass and transfer_pass,
    }, stored_angles


def compact_checkpoint_bytes(
    prompt: torch.Tensor,
    stored_angles: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> bytes:
    chunks = [prompt.detach().half().cpu().contiguous().numpy().tobytes()]
    for role in ROLE_GROUPS:
        hidden, residual = stored_angles[role]
        chunks.append(hidden.half().cpu().contiguous().numpy().tobytes())
        chunks.append(residual.half().cpu().contiguous().numpy().tobytes())
    coefficients = torch.zeros(DEPLOYED_MLP_MATRICES, BRANCHES, dtype=torch.float16)
    coefficients[:, 0] = 1.0
    chunks.append(coefficients.numpy().tobytes())
    payload = b"".join(chunks)
    expected = checkpoint_accounting()["fp16_checkpoint_bytes"]
    if len(payload) != expected:
        raise ValueError(f"checkpoint has {len(payload)} bytes, expected {expected}")
    return payload


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(541)
    shape = (16, 8)
    branches = 4
    components = 3
    atoms = tuple(normalized(torch.randn(shape, device=device)) for _ in range(3))
    geometry = make_branch_geometry(shape, branches=branches, seed_base=37, device=device)
    true_hidden = 0.08 * torch.randn(branches, shape[0] // 2, device=device)
    true_residual = 0.08 * torch.randn(branches, shape[1] // 2, device=device)
    pcs = []
    weights = []
    for atom in atoms:
        basis = atlas_basis(atom, true_hidden, true_residual, geometry)
        coefficients = torch.randn(components, branches, device=device)
        targets = coefficients @ basis
        targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-20)
        pcs.append(targets.reshape(components, *shape))
        weights.append(torch.full((components,), 1.0 / components, device=device))
    fit, _, _ = fit_role_atlas(
        atoms,
        tuple(pcs),
        tuple(weights),
        train_indices=(0, 1, 2),
        evaluation_indices=(0, 1, 2),
        seed_base=37,
        branches=branches,
        iterations=128,
    )
    minimum = min(
        row["learned"]["weighted_topk_capture"] for row in fit["rows"]
    )
    identity_geometry = make_branch_geometry(
        shape, branches=branches, seed_base=37, device=device
    )
    zeros_hidden = torch.zeros(branches, shape[0] // 2, device=device)
    zeros_residual = torch.zeros(branches, shape[1] // 2, device=device)
    identity_branch = atlas_basis(
        atoms[0], zeros_hidden, zeros_residual, identity_geometry
    )[0].reshape(shape)
    if not torch.equal(identity_branch, atoms[0]):
        raise AssertionError("branch zero is not exact identity")
    if minimum < 0.90:
        raise AssertionError({"minimum_weighted_capture": minimum, "fit": fit})
    accounting = checkpoint_accounting()
    if accounting["total_state_scalars"] != 566_016:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "minimum_known_atlas_weighted_capture": minimum,
        "accounting": accounting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    if any(
        value is None
        for value in (args.config, args.plan, args.trajectory_dir, args.output)
    ):
        parser.error("config, plan, trajectory, and output are required")
    assert args.config is not None and args.plan is not None
    assert args.trajectory_dir is not None and args.output is not None

    accounting = checkpoint_accounting()
    if accounting["total_state_scalars"] != 566_016 or accounting["state_fraction"] > 0.01:
        raise ValueError("H42 accounting mismatch")
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 566_016:
        raise ValueError("plan/accounting mismatch")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    config = json.loads(args.config.read_text())
    model = build_dense_model(config, args.device)
    prompt, task_targets, prompt_manifest = make_prompt(
        model, config, prompt_length=PROMPT_LENGTH, device=args.device
    )
    component_count = PREFLIGHT_COMPONENTS if args.preflight else BINDING_COMPONENTS
    pcs, weights, pc_manifest, w0_references = load_node_pc_inventory(
        args.trajectory_dir,
        components=component_count,
        device=args.device,
    )
    model_parameters = dict(model.named_parameters())
    w0_matches = {
        parameter: initialization_match(model_parameters[parameter], w0_references[parameter])
        for parameter in FROZEN_PARAMETERS
    }
    if not all(bool(record["accepted"]) for record in w0_matches.values()):
        raise ValueError(f"model/trajectory W0 mismatch: {w0_matches}")
    _, loss_function, initial_weights, function_manifest = make_model_lookahead_program(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=task_targets,
        ns_steps=5,
        momentum=0.0,
    )
    gradients = torch.func.grad(loss_function, argnums=0)(initial_weights, prompt)
    atoms = tuple(
        normalized(canonicalize(parameter, zeropower_via_newtonschulz5(gradient, steps=5).detach()))
        for parameter, gradient in zip(FROZEN_PARAMETERS, gradients, strict=True)
    )
    iterations = PREFLIGHT_ITERATIONS if args.preflight else BINDING_ITERATIONS
    audit, stored_angles = atlas_audit(
        atoms,
        pcs,
        weights,
        iterations=iterations,
        preflight=args.preflight,
    )
    classification = (
        "PREFLIGHT"
        if args.preflight
        else ("RETAINED" if audit["retained"] else "REJECTED")
    )
    checkpoint_path = output / "compact_checkpoint.bin"
    checkpoint_path.write_bytes(compact_checkpoint_bytes(prompt, stored_angles))
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_seconds = time.time() - started
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_role_wide_givens_atlas_v1",
        "classification": classification,
        "retained": audit["retained"],
        "preflight": args.preflight,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "pc_manifest": pc_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": {
            **function_manifest,
            "task_atom": "NS5 first task gradient at W0",
            "canonical_shape": list(CANONICAL_SHAPE),
            "role_atlas": "64 one-stage procedural-anchor plus learned-residual Givens branches",
            "local_image_dimension_ceiling": BRANCHES,
            "persistent_dense_basis": False,
            "persistent_branch_matrices": False,
        },
        "self_test": self_test(args.device),
        "audit": audit,
        "execution": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ).splitlines(),
            "entrypoint": str(script),
            "entrypoint_sha256": sha256(script),
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "plan": str(args.plan),
            "plan_sha256": sha256(args.plan),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": runtime_seconds,
            "projected_binding_runtime_seconds": (
                runtime_seconds
                * BINDING_ITERATIONS
                / PREFLIGHT_ITERATIONS
                * BINDING_COMPONENTS
                / PREFLIGHT_COMPONENTS
                * 2
                if args.preflight
                else runtime_seconds
            ),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
            ),
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)},
            "compact_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256(checkpoint_path),
            },
        },
        "limitations": [
            "H42 tests one exact 64-branch, one-stage role atlas around one task atom.",
            "The oracle measures top-PC representation, not LM CE.",
            "A pass authorizes only late/disjoint-stream representation audits.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": classification,
                "fit_all_pass": audit["fit_all_pass"],
                "transfer_pass": audit["transfer_pass"],
                "roles": {
                    role: {
                        "fit_all": row["fit_all_summary"],
                        "leave_one_out": row["leave_one_out_summary"],
                    }
                    for role, row in audit["roles"].items()
                },
                "runtime_seconds": runtime_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
