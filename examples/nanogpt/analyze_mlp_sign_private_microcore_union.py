#!/usr/bin/env python3
"""H46 capacity gate for a global sign/private microcore union."""
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

import numpy as np
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
    load_node_pc_inventory,
    projection_metrics,
)
from examples.nanogpt.analyze_mlp_shared_separable_conditioner_loo import (
    CANONICAL_SHAPE,
    canonicalize,
    normalized,
)
from examples.nanogpt.analyze_mlp_shared_sign_preconditioner_loo import (
    sign_from_score,
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
from examples.nanogpt.analyze_mlp_task_modulated_microcore_atlas import (
    initial_core,
    make_hash_geometry,
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_virtual_lookahead_joint import (
    make_model_lookahead_program,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


CORE_SIZE = 64
DEPLOYED_MLP_MATRICES = 24
DEPLOYED_SLOTS = (0, 1, 12, 13, 22, 23)
HASH_SEED_BASE = 977
RANDOM_SIGN_SEED = 20261031
GLOBAL_BITPLANE_BITS = CANONICAL_SHAPE[0] * CANONICAL_SHAPE[1]


def core_initialization_seed(deployed_slot: int) -> int:
    return 10001 + 101 * deployed_slot


def checkpoint_accounting() -> dict[str, int | float]:
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    prompt_fp16_bytes = 2 * prompt_scalars
    bitplane_bytes = GLOBAL_BITPLANE_BITS // 8
    private_core_scalars = DEPLOYED_MLP_MATRICES * CORE_SIZE * CORE_SIZE
    private_core_fp16_bytes = 2 * private_core_scalars
    local_coefficient_scalars = DEPLOYED_MLP_MATRICES * BRANCHES
    local_coefficient_fp16_bytes = 2 * local_coefficient_scalars
    total_bytes = (
        prompt_fp16_bytes
        + bitplane_bytes
        + private_core_fp16_bytes
        + local_coefficient_fp16_bytes
    )
    dense_fp16_bytes = 2 * DENSE_MLP_SCALARS
    return {
        "prompt_scalars": prompt_scalars,
        "prompt_fp16_bytes": prompt_fp16_bytes,
        "global_bitplane_bits": GLOBAL_BITPLANE_BITS,
        "global_bitplane_bytes": bitplane_bytes,
        "deployed_private_cores": DEPLOYED_MLP_MATRICES,
        "core_size": CORE_SIZE,
        "private_core_scalars": private_core_scalars,
        "private_core_fp16_bytes": private_core_fp16_bytes,
        "branches": BRANCHES,
        "local_coefficient_scalars": local_coefficient_scalars,
        "local_coefficient_fp16_bytes": local_coefficient_fp16_bytes,
        "trainable_real_scalars": prompt_scalars
        + private_core_scalars
        + local_coefficient_scalars,
        "total_compact_checkpoint_bytes": total_bytes,
        "dense_mlp_fp16_denominator_bytes": dense_fp16_bytes,
        "checkpoint_byte_fraction": total_bytes / dense_fp16_bytes,
        "persistent_dense_fp16_basis_scalars": 0,
        "persistent_ambient_bitplanes": 1,
        "persistent_hash_index_scalars": 0,
        "persistent_generated_atom_scalars": 0,
    }


def packed_bitplane(mask: torch.Tensor) -> bytes:
    if tuple(mask.shape) != CANONICAL_SHAPE:
        raise ValueError(f"unexpected sign shape: {tuple(mask.shape)}")
    bits = (mask.detach().flatten().cpu().numpy() > 0).astype(np.uint8)
    payload = np.packbits(bits, bitorder="little").tobytes()
    if len(payload) != GLOBAL_BITPLANE_BITS // 8:
        raise ValueError("packed sign size mismatch")
    return payload


def global_sign_mask(
    atoms: tuple[torch.Tensor, ...], pcs: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    score = torch.zeros_like(atoms[0])
    for atom, node_pcs in zip(atoms, pcs, strict=True):
        score = score + atom * node_pcs[0]
    return sign_from_score(score)


def random_sign_mask(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_SIGN_SEED)
    mask = 2 * torch.randint(2, CANONICAL_SHAPE, generator=generator) - 1
    return mask.to(device=device, dtype=torch.float32)


def union_basis(
    atom: torch.Tensor,
    core: torch.Tensor,
    geometry: dict[str, Any],
    sign_mask: torch.Tensor,
) -> torch.Tensor:
    rows = [atom.reshape(-1), normalized(sign_mask * atom).reshape(-1)]
    task_sign = torch.where(atom >= 0, torch.ones_like(atom), -torch.ones_like(atom))
    for branch in range(2, BRANCHES):
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


def fit_private_union(
    atom: torch.Tensor,
    pcs: torch.Tensor,
    weights: torch.Tensor,
    *,
    sign_mask: torch.Tensor,
    random_mask: torch.Tensor,
    deployed_slot: int,
    iterations: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    device = atom.device
    geometry = make_hash_geometry(
        tuple(atom.shape),
        core_size=CORE_SIZE,
        branches=BRANCHES,
        seed_base=HASH_SEED_BASE,
        node_index=deployed_slot,
        device=device,
    )
    base_core = initial_core(
        CORE_SIZE,
        seed=core_initialization_seed(deployed_slot),
        device=device,
    )
    core = torch.nn.Parameter(base_core.clone())
    optimizer = torch.optim.Adam([core], lr=LEARNING_RATE, weight_decay=0.0)
    history = []
    recorded_steps = {0, 1, 2, 3, 7, 15, 31, 63}
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        basis = union_basis(atom, core, geometry, sign_mask)
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
        learned_basis = union_basis(atom, core, geometry, sign_mask)
        _, learned = projection_metrics(
            learned_basis, pcs.flatten(1), weights, differentiable=False
        )
        _, sign_only = projection_metrics(
            learned_basis[:2], pcs.flatten(1), weights, differentiable=False
        )
        private_only_basis = torch.cat((learned_basis[:1], learned_basis[2:]), dim=0)
        _, private_only = projection_metrics(
            private_only_basis, pcs.flatten(1), weights, differentiable=False
        )
        random_sign_basis = union_basis(atom, core, geometry, random_mask)
        _, random_sign = projection_metrics(
            random_sign_basis, pcs.flatten(1), weights, differentiable=False
        )
        random_core_basis = union_basis(atom, base_core, geometry, sign_mask)
        _, random_core = projection_metrics(
            random_core_basis, pcs.flatten(1), weights, differentiable=False
        )
    return (
        {
            "deployed_slot": deployed_slot,
            "iterations": iterations,
            "history": history,
            "learned": learned,
            "sign_only": sign_only,
            "private_only": private_only,
            "random_sign_union": random_sign,
            "random_core_union": random_core,
            "union_minus_sign_only": learned["weighted_topk_capture"]
            - sign_only["weighted_topk_capture"],
            "union_minus_private_only": learned["weighted_topk_capture"]
            - private_only["weighted_topk_capture"],
            "union_minus_random_sign": learned["weighted_topk_capture"]
            - random_sign["weighted_topk_capture"],
            "union_minus_random_core": learned["weighted_topk_capture"]
            - random_core["weighted_topk_capture"],
            "learned_core_sha256": tensor_sha256(core),
            "initial_core_sha256": tensor_sha256(base_core),
        },
        core.detach(),
    )


def union_audit(
    atoms: tuple[torch.Tensor, ...],
    pcs: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    iterations: int,
) -> tuple[dict[str, Any], torch.Tensor, dict[int, torch.Tensor]]:
    sign_mask = global_sign_mask(atoms, pcs)
    random_mask = random_sign_mask(atoms[0].device)
    rows = []
    cores = {}
    for index, deployed_slot in enumerate(DEPLOYED_SLOTS):
        row, core = fit_private_union(
            atoms[index],
            pcs[index],
            weights[index],
            sign_mask=sign_mask,
            random_mask=random_mask,
            deployed_slot=deployed_slot,
            iterations=iterations,
        )
        rows.append({"index": index, **row})
        cores[deployed_slot] = core
    weighted = [row["learned"]["weighted_topk_capture"] for row in rows]
    minimum_pc = [row["learned"]["minimum_component_capture"] for row in rows]
    sign_margins = [row["union_minus_sign_only"] for row in rows]
    private_margins = [row["union_minus_private_only"] for row in rows]
    summary = {
        "minimum_weighted_topk_capture": min(weighted),
        "median_weighted_topk_capture": statistics.median(weighted),
        "maximum_weighted_topk_capture": max(weighted),
        "minimum_component_capture": min(minimum_pc),
        "minimum_union_minus_sign_only": min(sign_margins),
        "median_union_minus_sign_only": statistics.median(sign_margins),
        "minimum_union_minus_private_only": min(private_margins),
        "median_union_minus_private_only": statistics.median(private_margins),
    }
    retained = (
        summary["minimum_weighted_topk_capture"] >= 0.20
        and summary["minimum_component_capture"] >= 0.05
        and summary["minimum_union_minus_sign_only"] >= 0.05
        and summary["minimum_union_minus_private_only"] >= 0.05
    )
    return {
        "global_sign_positive_fraction": float((sign_mask > 0).float().mean()),
        "global_sign_sha256": hashlib.sha256(packed_bitplane(sign_mask)).hexdigest(),
        "random_sign_positive_fraction": float((random_mask > 0).float().mean()),
        "random_sign_sha256": hashlib.sha256(packed_bitplane(random_mask)).hexdigest(),
        "rows": rows,
        "summary": summary,
        "retained": retained,
    }, sign_mask, cores


def compact_checkpoint_bytes(
    prompt: torch.Tensor,
    sign_mask: torch.Tensor,
    learned_cores: dict[int, torch.Tensor],
) -> bytes:
    chunks = [prompt.detach().half().cpu().contiguous().numpy().tobytes()]
    chunks.append(packed_bitplane(sign_mask))
    for deployed_slot in range(DEPLOYED_MLP_MATRICES):
        core = learned_cores.get(deployed_slot)
        if core is None:
            core = initial_core(
                CORE_SIZE,
                seed=core_initialization_seed(deployed_slot),
                device=prompt.device,
            )
        chunks.append(core.half().cpu().contiguous().numpy().tobytes())
    coefficients = torch.zeros(DEPLOYED_MLP_MATRICES, BRANCHES, dtype=torch.float16)
    coefficients[:, 0] = 1.0
    chunks.append(coefficients.numpy().tobytes())
    payload = b"".join(chunks)
    expected = checkpoint_accounting()["total_compact_checkpoint_bytes"]
    if len(payload) != expected:
        raise ValueError(f"checkpoint has {len(payload)} bytes, expected {expected}")
    return payload


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(571)
    shape = (16, 8)
    core_size = 4
    branches = BRANCHES
    atom = normalized(torch.randn(shape, device=device))
    sign_mask = sign_from_score(torch.randn(shape, device=device))
    core = initial_core(core_size, seed=127, device=device)
    geometry = make_hash_geometry(
        shape,
        core_size=core_size,
        branches=branches,
        seed_base=HASH_SEED_BASE,
        node_index=3,
        device=device,
    )
    basis = union_basis(atom, core, geometry, sign_mask)
    if not torch.equal(basis[0].reshape(shape), atom):
        raise AssertionError("branch zero is not exact identity")
    differentiable_core = core.clone().requires_grad_(True)
    differentiable_basis = union_basis(atom, differentiable_core, geometry, sign_mask)
    differentiable_basis[2:, :17].square().sum().backward()
    if (
        differentiable_core.grad is None
        or not torch.isfinite(differentiable_core.grad).all()
        or float(differentiable_core.grad.norm()) == 0.0
    ):
        raise AssertionError("union private core has no finite nonzero gradient")
    coefficients = torch.randn(4, branches, device=device)
    targets = coefficients @ basis
    targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-20)
    weights = torch.full((4,), 0.25, device=device)
    weighted, metrics = projection_metrics(
        basis, targets, weights, relative_ridge=1e-8, differentiable=False
    )
    if float(weighted) < 0.90:
        raise AssertionError({"weighted_capture": float(weighted), "metrics": metrics})
    accounting = checkpoint_accounting()
    if accounting["total_compact_checkpoint_bytes"] != 1_132_032:
        raise AssertionError(accounting)
    canonical_mask = torch.ones(CANONICAL_SHAPE, device=device)
    if len(packed_bitplane(canonical_mask)) != 294_912:
        raise AssertionError("bitpacking mismatch")
    return {
        "status": "passed",
        "minimum_known_union_weighted_capture": float(weighted),
        "branch_zero_identity": True,
        "finite_nonzero_private_core_gradient": True,
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
    if (
        accounting["total_compact_checkpoint_bytes"] != 1_132_032
        or accounting["checkpoint_byte_fraction"] > 0.01
    ):
        raise ValueError("H46 accounting mismatch")
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_compact_checkpoint_bytes"] != 1_132_032:
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
    audit, sign_mask, learned_cores = union_audit(
        atoms, pcs, weights, iterations=iterations
    )
    classification = (
        "PREFLIGHT"
        if args.preflight
        else ("RETAINED" if audit["retained"] else "REJECTED")
    )
    checkpoint_path = output / "compact_checkpoint.bin"
    checkpoint_path.write_bytes(
        compact_checkpoint_bytes(prompt, sign_mask, learned_cores)
    )
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_seconds = time.time() - started
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_sign_private_microcore_union_v1",
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
            "decoder": "one packed global sign task atom plus 24 private 64x64 microcore residual banks",
            "persistent_ambient_bitplanes": 1,
            "persistent_dense_fp16_basis": False,
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
            "H46 tests one exact global sign/private 64x64 core union.",
            "The first audit measures dense path capacity, not late/disjoint generalization or LM CE.",
            "Failure closes this exact hybrid without a sweep.",
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
                        "union_minus_sign_only": row["union_minus_sign_only"],
                        "union_minus_private_only": row["union_minus_private_only"],
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
