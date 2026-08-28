#!/usr/bin/env python3
"""H45 capacity gate for node-private task-modulated microcore atlases."""
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

from examples.nanogpt.analyze_mlp_private_microcore_atlas_helpers import (
    checkpoint_accounting,
    compact_checkpoint_bytes,
    self_test,
)
from examples.nanogpt.analyze_mlp_role_wide_givens_atlas import (
    BINDING_COMPONENTS,
    BINDING_ITERATIONS,
    BRANCHES,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    PREFLIGHT_COMPONENTS,
    PREFLIGHT_ITERATIONS,
    PROMPT_LENGTH,
    load_node_pc_inventory,
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
from examples.nanogpt.analyze_mlp_task_modulated_microcore_atlas import (
    initial_core,
    make_hash_geometry,
    microcore_basis,
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_virtual_lookahead_joint import (
    make_model_lookahead_program,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


CORE_SIZE = 96
DEPLOYED_SLOTS = (0, 1, 12, 13, 22, 23)
HASH_SEED_BASE = 877


def core_initialization_seed(deployed_slot: int) -> int:
    return 9001 + 97 * deployed_slot


def fit_private_core(
    atom: torch.Tensor,
    pcs: torch.Tensor,
    weights: torch.Tensor,
    *,
    deployed_slot: int,
    iterations: int,
    core_size: int = CORE_SIZE,
    branches: int = BRANCHES,
    learning_rate: float = LEARNING_RATE,
) -> tuple[dict[str, Any], torch.Tensor]:
    device = atom.device
    geometry = make_hash_geometry(
        tuple(atom.shape),
        core_size=core_size,
        branches=branches,
        seed_base=HASH_SEED_BASE,
        node_index=deployed_slot,
        device=device,
    )
    base_core = initial_core(
        core_size,
        seed=core_initialization_seed(deployed_slot),
        device=device,
    )
    core = torch.nn.Parameter(base_core.clone())
    optimizer = torch.optim.Adam([core], lr=learning_rate, weight_decay=0.0)
    history = []
    recorded_steps = {0, 1, 2, 3, 7, 15, 31, 63}
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        basis = microcore_basis(atom, core, geometry)
        objective, _ = projection_metrics(
            basis,
            pcs.flatten(1),
            weights,
            differentiable=True,
        )
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
    with torch.no_grad():
        learned_basis = microcore_basis(atom, core, geometry)
        _, learned = projection_metrics(
            learned_basis,
            pcs.flatten(1),
            weights,
            differentiable=False,
        )
        random_basis = microcore_basis(atom, base_core, geometry)
        _, random_base = projection_metrics(
            random_basis,
            pcs.flatten(1),
            weights,
            differentiable=False,
        )
        _, task_only = projection_metrics(
            learned_basis[:1],
            pcs.flatten(1),
            weights,
            differentiable=False,
        )
    return (
        {
            "deployed_slot": deployed_slot,
            "iterations": iterations,
            "branches": branches,
            "core_size": core_size,
            "learning_rate": learning_rate,
            "core_initialization_seed": core_initialization_seed(deployed_slot),
            "history": history,
            "learned": learned,
            "random_core_base": random_base,
            "task_atom_only": task_only,
            "weighted_capture_margin": learned["weighted_topk_capture"]
            - random_base["weighted_topk_capture"],
            "learned_core_norm": float(core.detach().norm()),
            "learned_core_sha256": tensor_sha256(core),
            "initial_core_sha256": tensor_sha256(base_core),
        },
        core.detach(),
    )


def private_core_audit(
    atoms: tuple[torch.Tensor, ...],
    pcs: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    iterations: int,
) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    rows = []
    learned_cores = {}
    for index, deployed_slot in enumerate(DEPLOYED_SLOTS):
        row, core = fit_private_core(
            atoms[index],
            pcs[index],
            weights[index],
            deployed_slot=deployed_slot,
            iterations=iterations,
        )
        rows.append({"index": index, **row})
        learned_cores[deployed_slot] = core
    weighted = [row["learned"]["weighted_topk_capture"] for row in rows]
    minimum_pc = [row["learned"]["minimum_component_capture"] for row in rows]
    margins = [row["weighted_capture_margin"] for row in rows]
    summary = {
        "minimum_weighted_topk_capture": min(weighted),
        "median_weighted_topk_capture": statistics.median(weighted),
        "maximum_weighted_topk_capture": max(weighted),
        "minimum_component_capture": min(minimum_pc),
        "minimum_random_core_margin": min(margins),
        "median_random_core_margin": statistics.median(margins),
    }
    retained = (
        summary["minimum_weighted_topk_capture"] >= 0.20
        and summary["minimum_component_capture"] >= 0.05
        and summary["minimum_random_core_margin"] >= 0.05
    )
    return {"rows": rows, "summary": summary, "retained": retained}, learned_cores


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
    if accounting["total_state_scalars"] != 541_440 or accounting["state_fraction"] > 0.01:
        raise ValueError("H45 accounting mismatch")
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_state_scalars"] != 541_440:
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
    audit, learned_cores = private_core_audit(
        atoms,
        pcs,
        weights,
        iterations=iterations,
    )
    classification = (
        "PREFLIGHT"
        if args.preflight
        else ("RETAINED" if audit["retained"] else "REJECTED")
    )
    checkpoint_path = output / "compact_checkpoint.bin"
    checkpoint_path.write_bytes(compact_checkpoint_bytes(prompt, learned_cores))
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_seconds = time.time() - started
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_private_microcore_atlas_v1",
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
            "decoder": "24 private 96x96 cores, 63 procedural task-modulated hash views per node, and one task anchor",
            "observed_deployed_slots": list(DEPLOYED_SLOTS),
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
            "H45 tests one exact private 96x96 core and procedural hash schedule.",
            "The first audit measures path capacity, not disjoint-stream generalization or LM CE.",
            "A pass authorizes only the frozen late/disjoint-stream audit.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": classification,
                "summary": audit["summary"],
                "rows": [
                    {
                        "index": row["index"],
                        "deployed_slot": row["deployed_slot"],
                        "weighted_capture": row["learned"]["weighted_topk_capture"],
                        "minimum_component_capture": row["learned"]["minimum_component_capture"],
                        "random_core_margin": row["weighted_capture_margin"],
                    }
                    for row in audit["rows"]
                ],
                "runtime_seconds": runtime_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
