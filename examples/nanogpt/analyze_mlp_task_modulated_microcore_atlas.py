#!/usr/bin/env python3
"""H44 top-PC/LOO gate for a task-modulated shared microcore atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_role_wide_givens_atlas import (
    BINDING_COMPONENTS,
    BINDING_ITERATIONS,
    BRANCHES,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    PREFLIGHT_COMPONENTS,
    PREFLIGHT_ITERATIONS,
    PROMPT_LENGTH,
    PROMPT_WIDTH,
    RELATIVE_PROJECTION_RIDGE,
    ROLE_GROUPS,
    load_node_pc_inventory,
    projection_metrics,
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
from examples.nanogpt.analyze_mlp_synthetic_muon_program_joint import (
    FROZEN_PARAMETERS,
)
from examples.nanogpt.analyze_mlp_virtual_lookahead_joint import (
    make_model_lookahead_program,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


CORE_SIZE = 320
DEPLOYED_MLP_MATRICES = 24
ROLE_SEEDS = {"c_fc": 607, "c_proj": 683}
CORE_INITIALIZATION_SEEDS = {"c_fc": 7907, "c_proj": 7919}
NODE_SEED_STRIDE = 1009
BRANCH_SEED_STRIDE = 2


def checkpoint_accounting() -> dict[str, int | float]:
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    role_core_scalars = len(ROLE_GROUPS) * CORE_SIZE * CORE_SIZE
    local_coefficient_scalars = DEPLOYED_MLP_MATRICES * BRANCHES
    total_scalars = prompt_scalars + role_core_scalars + local_coefficient_scalars
    return {
        "prompt_scalars": prompt_scalars,
        "roles": len(ROLE_GROUPS),
        "core_size": CORE_SIZE,
        "role_core_scalars": role_core_scalars,
        "branches": BRANCHES,
        "local_coefficient_scalars": local_coefficient_scalars,
        "total_state_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "state_fraction": total_scalars / DENSE_MLP_SCALARS,
        "fp16_checkpoint_bytes": 2 * total_scalars,
        "persistent_dense_basis_scalars": 0,
        "persistent_hash_index_scalars": 0,
        "persistent_generated_atom_scalars": 0,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def initial_core(
    core_size: int, *, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    core = torch.randn(core_size, core_size, generator=generator)
    core = core / core.norm().clamp_min(1e-20)
    return core.to(device=device, dtype=torch.float32)


def make_hash_geometry(
    shape: tuple[int, int],
    *,
    core_size: int,
    branches: int,
    seed_base: int,
    node_index: int,
    device: torch.device,
) -> dict[str, Any]:
    rows, columns = shape
    row_indices = []
    column_indices = []
    row_signs = []
    column_signs = []
    for branch in range(branches):
        seed = seed_base + NODE_SEED_STRIDE * node_index + BRANCH_SEED_STRIDE * branch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        row_indices.append(
            torch.randint(core_size, (rows,), generator=generator).to(device)
        )
        column_indices.append(
            torch.randint(core_size, (columns,), generator=generator).to(device)
        )
        row_signs.append(
            (2 * torch.randint(2, (rows,), generator=generator) - 1)
            .to(device=device, dtype=torch.float32)
        )
        column_signs.append(
            (2 * torch.randint(2, (columns,), generator=generator) - 1)
            .to(device=device, dtype=torch.float32)
        )
    return {
        "row_indices": tuple(row_indices),
        "column_indices": tuple(column_indices),
        "row_signs": tuple(row_signs),
        "column_signs": tuple(column_signs),
    }


def microcore_basis(
    atom: torch.Tensor,
    core: torch.Tensor,
    geometry: dict[str, Any],
) -> torch.Tensor:
    rows = [atom.reshape(-1)]
    task_sign = torch.where(atom >= 0, torch.ones_like(atom), -torch.ones_like(atom))
    for branch in range(1, len(geometry["row_indices"])):
        expanded = core.index_select(0, geometry["row_indices"][branch])
        expanded = expanded.index_select(1, geometry["column_indices"][branch])
        expanded = (
            expanded
            * geometry["row_signs"][branch].unsqueeze(1)
            * geometry["column_signs"][branch].unsqueeze(0)
        )
        feature = task_sign * expanded
        feature = feature / feature.norm().clamp_min(1e-20)
        rows.append(feature.reshape(-1))
    return torch.stack(rows)


def summarize_role(rows: list[dict[str, Any]]) -> dict[str, float]:
    weighted = [row["learned"]["weighted_topk_capture"] for row in rows]
    minimum_pc = [row["learned"]["minimum_component_capture"] for row in rows]
    margins = [row["weighted_capture_margin"] for row in rows]
    task_only = [row["task_atom_only"]["weighted_topk_capture"] for row in rows]
    return {
        "minimum_weighted_topk_capture": min(weighted),
        "median_weighted_topk_capture": statistics.median(weighted),
        "maximum_weighted_topk_capture": max(weighted),
        "minimum_component_capture": min(minimum_pc),
        "minimum_random_core_margin": min(margins),
        "median_random_core_margin": statistics.median(margins),
        "maximum_task_atom_only_capture": max(task_only),
    }


def fit_role_microcore(
    atoms: tuple[torch.Tensor, ...],
    pcs: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    train_indices: tuple[int, ...],
    evaluation_indices: tuple[int, ...],
    seed_base: int,
    core_initialization_seed: int,
    core_size: int = CORE_SIZE,
    branches: int = BRANCHES,
    iterations: int = BINDING_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
) -> tuple[dict[str, Any], torch.Tensor]:
    if not train_indices or not evaluation_indices:
        raise ValueError("H44 requires nonempty train and evaluation folds")
    shape = tuple(atoms[train_indices[0]].shape)
    device = atoms[train_indices[0]].device
    relevant_indices = tuple(sorted(set(train_indices + evaluation_indices)))
    geometries = {
        index: make_hash_geometry(
            shape,
            core_size=core_size,
            branches=branches,
            seed_base=seed_base,
            node_index=index,
            device=device,
        )
        for index in relevant_indices
    }
    base_core = initial_core(core_size, seed=core_initialization_seed, device=device)
    core = torch.nn.Parameter(base_core.clone())
    optimizer = torch.optim.Adam([core], lr=learning_rate, weight_decay=0.0)
    history = []
    recorded_steps = {0, 1, 2, 3, 7, 15, 31, 63}
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), device=device)
        for index in train_indices:
            basis = microcore_basis(atoms[index], core, geometries[index])
            weighted, _ = projection_metrics(
                basis,
                pcs[index].flatten(1),
                weights[index],
                differentiable=True,
            )
            objective = objective + weighted / len(train_indices)
        (-objective).backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_([core], GRADIENT_CLIP_NORM)
        )
        optimizer.step()
        with torch.no_grad():
            core.div_(core.norm().clamp_min(1e-20))
        if step in recorded_steps or step == iterations - 1:
            history.append(
                {
                    "iteration": step + 1,
                    "weighted_capture_objective": float(objective.detach()),
                    "gradient_norm": gradient_norm,
                }
            )

    rows = []
    with torch.no_grad():
        for index in evaluation_indices:
            learned_basis = microcore_basis(atoms[index], core, geometries[index])
            _, learned = projection_metrics(
                learned_basis,
                pcs[index].flatten(1),
                weights[index],
                differentiable=False,
            )
            base_basis = microcore_basis(atoms[index], base_core, geometries[index])
            _, base = projection_metrics(
                base_basis,
                pcs[index].flatten(1),
                weights[index],
                differentiable=False,
            )
            _, task_only = projection_metrics(
                learned_basis[:1],
                pcs[index].flatten(1),
                weights[index],
                differentiable=False,
            )
            rows.append(
                {
                    "index": index,
                    "learned": learned,
                    "random_core_base": base,
                    "task_atom_only": task_only,
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
            "core_size": core_size,
            "learning_rate": learning_rate,
            "seed_base": seed_base,
            "core_initialization_seed": core_initialization_seed,
            "history": history,
            "rows": rows,
            "learned_core_norm": float(core.detach().norm()),
            "learned_core_sha256": tensor_sha256(core),
            "initial_core_sha256": tensor_sha256(base_core),
        },
        core.detach(),
    )


def microcore_audit(
    atoms: tuple[torch.Tensor, ...],
    pcs: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    iterations: int,
    preflight: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    roles = {}
    stored_cores = {}
    for role, group in ROLE_GROUPS.items():
        fit_all, core = fit_role_microcore(
            atoms,
            pcs,
            weights,
            train_indices=group,
            evaluation_indices=group,
            seed_base=ROLE_SEEDS[role],
            core_initialization_seed=CORE_INITIALIZATION_SEEDS[role],
            iterations=iterations,
        )
        stored_cores[role] = core
        heldouts = group[:1] if preflight else group
        loo_rows = []
        for heldout in heldouts:
            train = tuple(index for index in group if index != heldout)
            loo, _ = fit_role_microcore(
                atoms,
                pcs,
                weights,
                train_indices=train,
                evaluation_indices=(heldout,),
                seed_base=ROLE_SEEDS[role],
                core_initialization_seed=CORE_INITIALIZATION_SEEDS[role],
                iterations=iterations,
            )
            loo_rows.append(
                {
                    "heldout_index": heldout,
                    "train_indices": list(train),
                    **loo["rows"][0],
                    "fit_history": loo["history"],
                    "learned_core_norm": loo["learned_core_norm"],
                    "learned_core_sha256": loo["learned_core_sha256"],
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
        row["fit_all_summary"]["minimum_weighted_topk_capture"] >= 0.20
        and row["fit_all_summary"]["minimum_component_capture"] >= 0.05
        for row in roles.values()
    )
    transfer_pass = (not preflight) and all(
        row["leave_one_out_summary"]["minimum_weighted_topk_capture"] >= 0.20
        and row["leave_one_out_summary"]["minimum_component_capture"] >= 0.05
        and row["leave_one_out_summary"]["minimum_random_core_margin"] >= 0.05
        for row in roles.values()
    )
    return {
        "roles": roles,
        "fit_all_pass": fit_pass,
        "transfer_pass": transfer_pass,
        "retained": fit_pass and transfer_pass,
    }, stored_cores


def compact_checkpoint_bytes(
    prompt: torch.Tensor, stored_cores: dict[str, torch.Tensor]
) -> bytes:
    chunks = [prompt.detach().half().cpu().contiguous().numpy().tobytes()]
    for role in ROLE_GROUPS:
        chunks.append(stored_cores[role].half().cpu().contiguous().numpy().tobytes())
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
    torch.manual_seed(557)
    shape = (16, 8)
    core_size = 4
    branches = 8
    components = 4
    atom = normalized(torch.randn(shape, device=device))
    core = initial_core(core_size, seed=101, device=device)
    geometry = make_hash_geometry(
        shape,
        core_size=core_size,
        branches=branches,
        seed_base=67,
        node_index=2,
        device=device,
    )
    basis = microcore_basis(atom, core, geometry)
    regenerated = microcore_basis(
        atom,
        initial_core(core_size, seed=101, device=device),
        make_hash_geometry(
            shape,
            core_size=core_size,
            branches=branches,
            seed_base=67,
            node_index=2,
            device=device,
        ),
    )
    if not torch.equal(basis, regenerated):
        raise AssertionError("procedural microcore regeneration is not exact")
    if not torch.equal(basis[0].reshape(shape), atom):
        raise AssertionError("branch zero is not exact identity")
    differentiable_core = core.clone().requires_grad_(True)
    differentiable_basis = microcore_basis(atom, differentiable_core, geometry)
    differentiable_basis[1:, :17].square().sum().backward()
    if (
        differentiable_core.grad is None
        or not torch.isfinite(differentiable_core.grad).all()
        or float(differentiable_core.grad.norm()) == 0.0
    ):
        raise AssertionError("microcore basis has no finite nonzero gradient")
    coefficients = torch.randn(components, branches, device=device)
    targets = coefficients @ basis
    targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-20)
    weights = torch.full((components,), 1.0 / components, device=device)
    weighted, metrics = projection_metrics(
        basis, targets, weights, relative_ridge=1e-8, differentiable=False
    )
    if not torch.isfinite(basis).all():
        raise AssertionError("microcore atlas is not finite")
    if float(weighted) < 0.90:
        raise AssertionError({"weighted_capture": float(weighted), "metrics": metrics})
    accounting = checkpoint_accounting()
    if accounting["total_state_scalars"] != 525_056:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "minimum_known_microcore_weighted_capture": float(weighted),
        "branch_zero_identity": True,
        "deterministic_regeneration": True,
        "finite_nonzero_core_gradient": True,
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
    if accounting["total_state_scalars"] != 525_056 or accounting["state_fraction"] > 0.01:
        raise ValueError("H44 accounting mismatch")
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 525_056:
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
    audit, stored_cores = microcore_audit(
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
    checkpoint_path.write_bytes(compact_checkpoint_bytes(prompt, stored_cores))
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_seconds = time.time() - started
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_task_modulated_microcore_atlas_v1",
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
            "role_atlas": "one 320x320 continuous core per role, 63 procedural task-modulated hash views, and one task anchor",
            "local_image_dimension_ceiling": BRANCHES,
            "persistent_dense_basis": False,
            "persistent_hash_geometry": False,
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
            "H44 tests one exact 320x320 role-core and procedural hash schedule.",
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
