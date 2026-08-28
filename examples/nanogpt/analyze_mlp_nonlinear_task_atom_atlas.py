#!/usr/bin/env python3
"""H43 top-PC/LOO gate for a nonlinear task-atom-conditioned atlas."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_role_givens_transport_loo import (
    fullcoverage_transport,
)
from examples.nanogpt.analyze_mlp_role_wide_givens_atlas import (
    BINDING_COMPONENTS,
    BINDING_ITERATIONS,
    BRANCHES,
    PREFLIGHT_COMPONENTS,
    PREFLIGHT_ITERATIONS,
    PROMPT_LENGTH,
    ROLE_GROUPS,
    atlas_audit,
    checkpoint_accounting,
    compact_checkpoint_bytes,
    load_node_pc_inventory,
    make_branch_geometry,
    projection_metrics,
)
from examples.nanogpt.analyze_mlp_shared_separable_conditioner_loo import (
    CANONICAL_SHAPE,
    canonicalize,
    normalized,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program import (
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


FEATURE_SCALES = (0.5, 1.0, 2.0, 4.0)
FEATURE_BIASES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def feature_parameters(branch: int) -> tuple[float, float]:
    if branch <= 0:
        raise ValueError("branch zero is the exact linear anchor")
    offset = branch - 1
    return (
        FEATURE_SCALES[offset % len(FEATURE_SCALES)],
        FEATURE_BIASES[(offset // len(FEATURE_SCALES)) % len(FEATURE_BIASES)],
    )


def nonlinear_atlas_basis(
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
        if branch == 0:
            rows.append(atom.reshape(-1))
            continue
        scale, bias = feature_parameters(branch)
        ambient = math.sqrt(atom.numel()) * transported
        feature = F.gelu(scale * ambient + bias, approximate="none")
        feature = feature - feature.mean()
        feature = feature / feature.norm().clamp_min(1e-12)
        rows.append(feature.reshape(-1))
    return torch.stack(rows)


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(547)
    shape = (16, 8)
    branches = 8
    components = 4
    atom = normalized(torch.randn(shape, device=device))
    geometry = make_branch_geometry(
        shape, branches=branches, seed_base=43, device=device
    )
    hidden = 0.08 * torch.randn(branches, shape[0] // 2, device=device)
    residual = 0.08 * torch.randn(branches, shape[1] // 2, device=device)
    basis = nonlinear_atlas_basis(atom, hidden, residual, geometry)
    coefficients = torch.randn(components, branches, device=device)
    targets = coefficients @ basis
    targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-20)
    weights = torch.full((components,), 1.0 / components, device=device)
    weighted, metrics = projection_metrics(
        basis, targets, weights, relative_ridge=1e-8, differentiable=False
    )
    zeros_hidden = torch.zeros_like(hidden)
    zeros_residual = torch.zeros_like(residual)
    identity_branch = nonlinear_atlas_basis(
        atom, zeros_hidden, zeros_residual, geometry
    )[0].reshape(shape)
    if not torch.equal(identity_branch, atom):
        raise AssertionError("branch zero is not exact identity")
    if not torch.isfinite(basis).all():
        raise AssertionError("nonlinear feature atlas is not finite")
    feature_norms = basis[1:].norm(dim=1)
    if not torch.allclose(feature_norms, torch.ones_like(feature_norms), atol=1e-5):
        raise AssertionError({"feature_norms": feature_norms.tolist()})
    if float(weighted) < 0.90:
        raise AssertionError({"weighted_capture": float(weighted), "metrics": metrics})
    accounting = checkpoint_accounting()
    if accounting["total_state_scalars"] != 566_016:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "minimum_known_atlas_weighted_capture": float(weighted),
        "minimum_feature_norm": float(feature_norms.min()),
        "maximum_feature_norm": float(feature_norms.max()),
        "branch_zero_identity": True,
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
        raise ValueError("H43 accounting mismatch")
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
        basis_function=nonlinear_atlas_basis,
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
        "schema_version": "nanogpt_mlp_nonlinear_task_atom_atlas_v1",
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
            "role_atlas": "64 task-atom-conditioned standardized GELU feature branches",
            "feature_scales": list(FEATURE_SCALES),
            "feature_biases": list(FEATURE_BIASES),
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
            "H43 tests one exact fixed GELU scale/bias schedule around one task atom.",
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
