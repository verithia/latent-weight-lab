#!/usr/bin/env python3
"""H41 same-role LOO gate for exact-budget full-coverage Givens transports."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_shared_fullcoverage_givens_capacity import (
    fullcoverage_transport,
    make_stage_permutations,
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_shared_separable_conditioner_loo import (
    CANONICAL_SHAPE,
    canonicalize,
    normalized,
    optimal_scalar,
    squared_cosine,
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
    joint_leading_pc,
)
from examples.nanogpt.analyze_mlp_virtual_lookahead_joint import (
    make_model_lookahead_program,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


PROMPT_LENGTH = 372
PROMPT_WIDTH = 768
DEPLOYED_MLP_MATRICES = 24
ROLE_GROUPS = {"c_fc": (0, 2, 4), "c_proj": (1, 3, 5)}
ROLE_SEEDS = {"c_fc": 409, "c_proj": 419}
STAGES = 73
BINDING_ITERATIONS = 64
PREFLIGHT_ITERATIONS = 4
LEARNING_RATE = 0.01
GRADIENT_CLIP_NORM = 10.0
SMOOTH_ABSOLUTE_EPSILON = 1e-12


def checkpoint_accounting() -> dict[str, int | float]:
    rows, columns = CANONICAL_SHAPE
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    angles_per_stage = rows // 2 + columns // 2
    role_angle_scalars = len(ROLE_GROUPS) * STAGES * angles_per_stage
    total_scalars = prompt_scalars + role_angle_scalars + DEPLOYED_MLP_MATRICES
    return {
        "prompt_scalars": prompt_scalars,
        "roles": len(ROLE_GROUPS),
        "stages_per_role": STAGES,
        "angles_per_stage_per_role": angles_per_stage,
        "role_angle_scalars": role_angle_scalars,
        "node_coefficient_scalars": DEPLOYED_MLP_MATRICES,
        "total_state_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "state_fraction": total_scalars / DENSE_MLP_SCALARS,
        "fp16_checkpoint_bytes": 2 * total_scalars,
        "persistent_dense_basis_scalars": 0,
        "persistent_permutation_scalars": 0,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum_capture": min(values),
        "median_capture": statistics.median(values),
        "maximum_capture": max(values),
    }


def fit_role_transport(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    train_indices: tuple[int, ...],
    evaluation_indices: tuple[int, ...],
    permutation_seed: int,
    stages: int = STAGES,
    iterations: int = BINDING_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if not train_indices or not evaluation_indices:
        raise ValueError("H41 requires nonempty train and evaluation folds")
    atom_rows = tuple(normalized(value.detach().float()) for value in atoms)
    target_rows = tuple(normalized(value.detach().float()) for value in targets)
    shape = tuple(atom_rows[0].shape)
    if len(shape) != 2 or any(
        tuple(value.shape) != shape for value in (*atom_rows, *target_rows)
    ):
        raise ValueError("H41 requires equal matrix shapes")
    rows, columns = shape
    device = atom_rows[0].device
    hidden_permutations = make_stage_permutations(
        rows, stages, permutation_seed, device
    )
    residual_permutations = make_stage_permutations(
        columns, stages, permutation_seed + 1, device
    )
    hidden_angles = torch.nn.Parameter(
        torch.zeros(stages, rows // 2, device=device)
    )
    residual_angles = torch.nn.Parameter(
        torch.zeros(stages, columns // 2, device=device)
    )
    parameters = [hidden_angles, residual_angles]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, float | int]] = []
    recorded_steps = {0, 1, 2, 3, 7, 15, 31, 63}
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for index in train_indices:
            prediction = fullcoverage_transport(
                atom_rows[index],
                hidden_angles,
                residual_angles,
                hidden_permutations,
                residual_permutations,
            )
            inner = (prediction * target_rows[index]).sum()
            loss = loss - torch.sqrt(inner.square() + SMOOTH_ABSOLUTE_EPSILON)
        loss = loss / len(train_indices)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        )
        optimizer.step()
        if step in recorded_steps or step == iterations - 1:
            history.append(
                {
                    "iteration": step + 1,
                    "loss": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                }
            )

    rows_out = []
    with torch.no_grad():
        for index in evaluation_indices:
            prediction = fullcoverage_transport(
                atom_rows[index],
                hidden_angles,
                residual_angles,
                hidden_permutations,
                residual_permutations,
            )
            rows_out.append(
                {
                    "index": index,
                    "raw_capture": squared_cosine(
                        atom_rows[index], target_rows[index]
                    ),
                    "transport_capture": squared_cosine(
                        prediction, target_rows[index]
                    ),
                    "optimal_scalar": float(
                        optimal_scalar(prediction, target_rows[index])
                    ),
                    "prediction_norm": float(prediction.norm()),
                }
            )
    captures = [row["transport_capture"] for row in rows_out]
    fit = {
        "train_indices": list(train_indices),
        "evaluation_indices": list(evaluation_indices),
        "iterations": iterations,
        "stages": stages,
        "learning_rate": learning_rate,
        "permutation_seed": permutation_seed,
        "history": history,
        "rows": rows_out,
        **_summary(captures),
        "maximum_norm_error": max(
            abs(row["prediction_norm"] - 1.0) for row in rows_out
        ),
        "angle_norms": {
            "hidden": float(hidden_angles.norm()),
            "residual": float(residual_angles.norm()),
        },
        "angle_sha256": {
            "hidden": tensor_sha256(hidden_angles),
            "residual": tensor_sha256(residual_angles),
        },
    }
    return fit, hidden_angles.detach(), residual_angles.detach()


def role_transport_audit(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    iterations: int = BINDING_ITERATIONS,
    preflight: bool = False,
    stages: int = STAGES,
) -> tuple[dict[str, Any], dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    if len(atoms) != 6 or len(targets) != 6:
        raise ValueError("H41 requires the frozen six-node inventory")
    role_angles = {}
    deployed_coefficients = torch.ones(
        DEPLOYED_MLP_MATRICES, device=atoms[0].device, dtype=torch.float32
    )
    deployed_indices = (0, 1, 12, 13, 22, 23)
    roles: dict[str, Any] = {}
    for role, group in ROLE_GROUPS.items():
        fit_all, hidden_angles, residual_angles = fit_role_transport(
            atoms,
            targets,
            train_indices=group,
            evaluation_indices=group,
            permutation_seed=ROLE_SEEDS[role],
            stages=stages,
            iterations=iterations,
        )
        role_angles[role] = (hidden_angles, residual_angles)
        for row in fit_all["rows"]:
            deployed_coefficients[deployed_indices[row["index"]]] = row[
                "optimal_scalar"
            ]
        heldouts = group[:1] if preflight else group
        loo_rows = []
        for heldout in heldouts:
            train = tuple(index for index in group if index != heldout)
            loo_fit, _, _ = fit_role_transport(
                atoms,
                targets,
                train_indices=train,
                evaluation_indices=(heldout,),
                permutation_seed=ROLE_SEEDS[role],
                stages=stages,
                iterations=iterations,
            )
            loo_rows.append(
                {
                    "heldout_index": heldout,
                    "train_indices": list(train),
                    **loo_fit["rows"][0],
                    "angle_norms": loo_fit["angle_norms"],
                    "angle_sha256": loo_fit["angle_sha256"],
                    "history": loo_fit["history"],
                }
            )
        loo_captures = [row["transport_capture"] for row in loo_rows]
        roles[role] = {
            "group": list(group),
            "fit_all": fit_all,
            "leave_one_out_rows": loo_rows,
            "leave_one_out_complete": len(heldouts) == len(group),
            "leave_one_out_minimum_capture": min(loo_captures),
            "leave_one_out_median_capture": statistics.median(loo_captures),
            "leave_one_out_maximum_capture": max(loo_captures),
        }
    capacity_pass = all(
        row["fit_all"]["minimum_capture"] >= 0.05
        and row["fit_all"]["median_capture"] >= 0.10
        for row in roles.values()
    )
    transfer_pass = (not preflight) and all(
        row["leave_one_out_minimum_capture"] >= 0.05
        and row["leave_one_out_median_capture"] >= 0.10
        for row in roles.values()
    )
    return (
        {
            "roles": roles,
            "capacity_pass": capacity_pass,
            "transfer_pass": transfer_pass,
            "retained": capacity_pass and transfer_pass,
        },
        role_angles,
        deployed_coefficients,
    )


def compact_checkpoint_bytes(
    prompt: torch.Tensor,
    role_angles: dict[str, tuple[torch.Tensor, torch.Tensor]],
    coefficients: torch.Tensor,
) -> bytes:
    chunks = [prompt.detach().half().cpu().contiguous().numpy().tobytes()]
    for role in ROLE_GROUPS:
        hidden, residual = role_angles[role]
        chunks.extend(
            (
                hidden.half().cpu().contiguous().numpy().tobytes(),
                residual.half().cpu().contiguous().numpy().tobytes(),
            )
        )
    chunks.append(coefficients.detach().half().cpu().contiguous().numpy().tobytes())
    payload = b"".join(chunks)
    expected = checkpoint_accounting()["fp16_checkpoint_bytes"]
    if len(payload) != expected:
        raise ValueError(f"compact checkpoint has {len(payload)} bytes, expected {expected}")
    return payload


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(421)
    shape = (16, 8)
    stages = 4
    atoms = tuple(normalized(torch.randn(shape, device=device)) for _ in range(6))
    targets: list[torch.Tensor] = [torch.empty(0)] * 6
    role_test_seeds: dict[str, int] = {}
    for role, group in ROLE_GROUPS.items():
        seed = 17 if role == "c_fc" else 29
        role_test_seeds[role] = seed
        hidden_permutations = make_stage_permutations(shape[0], stages, seed, device)
        residual_permutations = make_stage_permutations(shape[1], stages, seed + 1, device)
        hidden = 0.2 * torch.randn(stages, shape[0] // 2, device=device)
        residual = 0.2 * torch.randn(stages, shape[1] // 2, device=device)
        zero_hidden = torch.zeros_like(hidden)
        zero_residual = torch.zeros_like(residual)
        identity = fullcoverage_transport(
            atoms[group[0]],
            zero_hidden,
            zero_residual,
            hidden_permutations,
            residual_permutations,
        )
        if not torch.equal(identity, atoms[group[0]]):
            raise AssertionError("zero-angle transform is not exact identity")
        for index in group:
            targets[index] = fullcoverage_transport(
                atoms[index],
                hidden,
                residual,
                hidden_permutations,
                residual_permutations,
            )
    for role, group in ROLE_GROUPS.items():
        seed = role_test_seeds[role]
        fit, _, _ = fit_role_transport(
            atoms,
            tuple(targets),
            train_indices=group[:2],
            evaluation_indices=group,
            permutation_seed=seed,
            stages=stages,
            iterations=128,
        )
        if fit["minimum_capture"] < 0.90 or fit["maximum_norm_error"] > 1e-5:
            raise AssertionError({role: fit})
    accounting = checkpoint_accounting()
    if accounting["total_state_scalars"] != 566_040:
        raise AssertionError(accounting)
    return {"status": "passed", "accounting": accounting}


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
    if accounting["total_state_scalars"] != 566_040 or accounting["state_fraction"] > 0.01:
        raise ValueError("H41 accounting mismatch")
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 566_040:
        raise ValueError("plan/accounting mismatch")
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    model = build_dense_model(config, args.device)
    prompt, task_targets, prompt_manifest = make_prompt(
        model, config, prompt_length=PROMPT_LENGTH, device=args.device
    )
    joint_target, leading_fraction, target_manifest, w0_references = joint_leading_pc(
        args.trajectory_dir, parameters=FROZEN_PARAMETERS, device=args.device
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
    raw_atoms = tuple(
        zeropower_via_newtonschulz5(gradient, steps=5).detach()
        for gradient in gradients
    )
    split_targets = torch.split(joint_target, [weight.numel() for weight in initial_weights])
    atoms = tuple(
        canonicalize(parameter, atom)
        for parameter, atom in zip(FROZEN_PARAMETERS, raw_atoms, strict=True)
    )
    target_parts = tuple(
        canonicalize(parameter, part.reshape_as(weight))
        for parameter, part, weight in zip(
            FROZEN_PARAMETERS, split_targets, initial_weights, strict=True
        )
    )
    iterations = PREFLIGHT_ITERATIONS if args.preflight else BINDING_ITERATIONS
    transfer, role_angles, coefficients = role_transport_audit(
        atoms,
        target_parts,
        iterations=iterations,
        preflight=args.preflight,
    )
    transfer["leading_pc_energy_fraction"] = leading_fraction
    classification = (
        "PREFLIGHT"
        if args.preflight
        else ("RETAINED" if transfer["retained"] else "REJECTED")
    )

    checkpoint_path = output / "compact_checkpoint.bin"
    checkpoint_path.write_bytes(
        compact_checkpoint_bytes(prompt, role_angles, coefficients)
    )
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_seconds = time.time() - started
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_role_givens_transport_loo_v1",
        "classification": classification,
        "retained": transfer["retained"],
        "preflight": args.preflight,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": {
            **function_manifest,
            "task_atom": "NS5 first task gradient at W0",
            "canonical_shape": list(CANONICAL_SHAPE),
            "role_transports": "73 full-coverage Givens stages per role and axis",
            "singular_value_invariant": True,
            "persistent_dense_basis": False,
            "persistent_permutations": False,
        },
        "self_test": self_test(args.device),
        "transfer": transfer,
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
                runtime_seconds * BINDING_ITERATIONS / PREFLIGHT_ITERATIONS * 2
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
            "H41 is restricted to one orthogonal transport per MLP role.",
            "Orthogonal transport cannot change task-atom singular values except for one node scalar.",
            "A pass authorizes only a remaining-PC and transfer audit, never CE or scale-up.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": classification,
                "capacity_pass": transfer["capacity_pass"],
                "transfer_pass": transfer["transfer_pass"],
                "role_summaries": {
                    role: {
                        "fit_all_median_capture": row["fit_all"]["median_capture"],
                        "leave_one_out_median_capture": row[
                            "leave_one_out_median_capture"
                        ],
                    }
                    for role, row in transfer["roles"].items()
                },
                "runtime_seconds": runtime_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
