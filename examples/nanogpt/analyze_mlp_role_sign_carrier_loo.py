#!/usr/bin/env python3
"""H40 same-role LOO gate for two exact-budget packed sign carriers."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.nanogpt.analyze_mlp_shared_separable_conditioner_loo import optimal_scalar
from examples.nanogpt.analyze_mlp_shared_sign_preconditioner_loo import (
    CANONICAL_SHAPE,
    GLOBAL_BITPLANE_BITS,
    bitplane_sha256,
    canonicalize,
    normalized,
    sign_from_score,
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


PROMPT_LENGTH = 353
PROMPT_WIDTH = 768
DEPLOYED_MLP_MATRICES = 24
ROLE_GROUPS = {"c_fc": (0, 2, 4), "c_proj": (1, 3, 5)}
RANDOM_MASK_SEED = 20261013


def checkpoint_accounting() -> dict[str, int | float]:
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    prompt_bytes = 2 * prompt_scalars
    bitplane_bits = 2 * GLOBAL_BITPLANE_BITS
    bitplane_bytes = bitplane_bits // 8
    coefficient_bytes = 2 * DEPLOYED_MLP_MATRICES
    total_bytes = prompt_bytes + bitplane_bytes + coefficient_bytes
    dense_bytes = 2 * DENSE_MLP_SCALARS
    return {
        "prompt_scalars": prompt_scalars,
        "prompt_fp16_bytes": prompt_bytes,
        "role_bitplanes": 2,
        "role_bitplane_bits": bitplane_bits,
        "role_bitplane_packed_bytes": bitplane_bytes,
        "node_coefficient_scalars": DEPLOYED_MLP_MATRICES,
        "node_coefficient_fp16_bytes": coefficient_bytes,
        "total_compact_checkpoint_bytes": total_bytes,
        "dense_mlp_fp16_denominator_bytes": dense_bytes,
        "checkpoint_byte_fraction": total_bytes / dense_bytes,
        "fp16_equivalent_scalars": total_bytes / 2,
    }


def pack_mask(mask: torch.Tensor) -> bytes:
    bits = (mask.detach().flatten().cpu().numpy() > 0).astype(np.uint8)
    packed = np.packbits(bits, bitorder="little")
    expected = GLOBAL_BITPLANE_BITS // 8
    if packed.nbytes != expected:
        raise ValueError(f"packed mask has {packed.nbytes} bytes, expected {expected}")
    return packed.tobytes()


def role_sign_transfer(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    random_seed: int = RANDOM_MASK_SEED,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    if len(atoms) != 6 or len(targets) != 6:
        raise ValueError("H40 requires the frozen six-node inventory")
    atoms = tuple(normalized(value) for value in atoms)
    targets = tuple(normalized(value) for value in targets)
    for value in (*atoms, *targets):
        if tuple(value.shape) != CANONICAL_SHAPE:
            raise ValueError("H40 received a noncanonical node")
    masks: dict[str, torch.Tensor] = {}
    deployed_coefficients = torch.ones(
        DEPLOYED_MLP_MATRICES, device=atoms[0].device, dtype=torch.float32
    )
    deployed_indices = (0, 1, 12, 13, 22, 23)
    roles: dict[str, Any] = {}
    for role_offset, (role, group) in enumerate(ROLE_GROUPS.items()):
        all_score = sum(atoms[index] * targets[index] for index in group)
        all_mask = sign_from_score(all_score)
        masks[role] = all_mask
        generator = torch.Generator(device=atoms[0].device)
        generator.manual_seed(random_seed + role_offset)
        random_mask = torch.randint(
            0,
            2,
            CANONICAL_SHAPE,
            generator=generator,
            device=atoms[0].device,
            dtype=torch.int8,
        ).float().mul_(2.0).sub_(1.0)
        rows = []
        for heldout in group:
            train = tuple(index for index in group if index != heldout)
            train_score = sum(atoms[index] * targets[index] for index in train)
            loo_mask = sign_from_score(train_score)
            fit_all_base = all_mask * atoms[heldout]
            loo_base = loo_mask * atoms[heldout]
            coefficient = optimal_scalar(fit_all_base, targets[heldout])
            deployed_coefficients[deployed_indices[heldout]] = coefficient
            rows.append(
                {
                    "heldout_index": heldout,
                    "train_indices": list(train),
                    "fit_all_capture": squared_cosine(fit_all_base, targets[heldout]),
                    "leave_one_out_capture": squared_cosine(loo_base, targets[heldout]),
                    "raw_capture": squared_cosine(atoms[heldout], targets[heldout]),
                    "random_mask_capture": squared_cosine(
                        random_mask * atoms[heldout], targets[heldout]
                    ),
                    "fit_all_optimal_scalar": float(coefficient),
                    "leave_one_out_optimal_scalar": float(
                        optimal_scalar(loo_base, targets[heldout])
                    ),
                    "leave_one_out_positive_fraction": float(
                        (loo_mask > 0).float().mean()
                    ),
                    "leave_one_out_mask_sha256": bitplane_sha256(loo_mask),
                }
            )
        fit_all = [row["fit_all_capture"] for row in rows]
        loo = [row["leave_one_out_capture"] for row in rows]
        roles[role] = {
            "group": list(group),
            "rows": rows,
            "fit_all_minimum_capture": min(fit_all),
            "fit_all_median_capture": statistics.median(fit_all),
            "fit_all_maximum_capture": max(fit_all),
            "leave_one_out_minimum_capture": min(loo),
            "leave_one_out_median_capture": statistics.median(loo),
            "leave_one_out_maximum_capture": max(loo),
            "fit_all_positive_fraction": float((all_mask > 0).float().mean()),
            "fit_all_mask_sha256": bitplane_sha256(all_mask),
            "random_mask_positive_fraction": float((random_mask > 0).float().mean()),
            "random_mask_sha256": bitplane_sha256(random_mask),
            "random_mask_seed": random_seed + role_offset,
        }
    capacity_pass = all(
        row["fit_all_minimum_capture"] >= 0.05
        and row["fit_all_median_capture"] >= 0.10
        for row in roles.values()
    )
    transfer_pass = all(
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
        masks,
        deployed_coefficients,
    )


def compact_checkpoint_bytes(
    prompt: torch.Tensor,
    masks: dict[str, torch.Tensor],
    coefficients: torch.Tensor,
) -> bytes:
    prompt_bytes = prompt.detach().half().cpu().contiguous().numpy().tobytes()
    coefficient_bytes = (
        coefficients.detach().half().cpu().contiguous().numpy().tobytes()
    )
    payload = (
        prompt_bytes
        + pack_mask(masks["c_fc"])
        + pack_mask(masks["c_proj"])
        + coefficient_bytes
    )
    expected = checkpoint_accounting()["total_compact_checkpoint_bytes"]
    if len(payload) != expected:
        raise ValueError(f"compact checkpoint has {len(payload)} bytes, expected {expected}")
    return payload


def self_test(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(401)
    true_masks = {
        role: torch.where(
            torch.rand(CANONICAL_SHAPE, device=device) > 0.5,
            torch.tensor(1.0, device=device),
            torch.tensor(-1.0, device=device),
        )
        for role in ROLE_GROUPS
    }
    atoms = tuple(torch.randn(CANONICAL_SHAPE, device=device) for _ in range(6))
    targets_list: list[torch.Tensor] = [torch.empty(0)] * 6
    for role, group in ROLE_GROUPS.items():
        for index in group:
            targets_list[index] = (
                true_masks[role] * atoms[index] + 0.01 * torch.randn_like(atoms[index])
            )
    result, masks, coefficients = role_sign_transfer(atoms, tuple(targets_list))
    minimum = min(
        row["leave_one_out_minimum_capture"] for row in result["roles"].values()
    )
    if minimum < 0.99 or not result["retained"]:
        raise AssertionError(result)
    prompt = torch.randn(PROMPT_LENGTH, PROMPT_WIDTH, device=device)
    payload = compact_checkpoint_bytes(prompt, masks, coefficients)
    accounting = checkpoint_accounting()
    if len(payload) != 1_132_080 or accounting["checkpoint_byte_fraction"] > 0.01:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "minimum_leave_one_out_capture": minimum,
        "checkpoint_bytes": len(payload),
        "accounting": accounting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    if any(value is None for value in (args.config, args.plan, args.trajectory_dir, args.output)):
        parser.error("config, plan, trajectory, and output are required")
    assert args.config is not None and args.plan is not None
    assert args.trajectory_dir is not None and args.output is not None

    accounting = checkpoint_accounting()
    if accounting["total_compact_checkpoint_bytes"] != 1_132_080:
        raise ValueError("H40 accounting mismatch")
    if accounting["checkpoint_byte_fraction"] > 0.01:
        raise ValueError("H40 exceeds the one-percent checkpoint-byte budget")
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_compact_checkpoint_bytes"] != 1_132_080:
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
        zeropower_via_newtonschulz5(gradient, steps=5).detach() for gradient in gradients
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
    transfer, masks, coefficients = role_sign_transfer(atoms, target_parts)
    transfer["leading_pc_energy_fraction"] = leading_fraction
    classification = "RETAINED" if transfer["retained"] else "REJECTED"

    checkpoint_path = output / "compact_checkpoint.bin"
    checkpoint_path.write_bytes(compact_checkpoint_bytes(prompt, masks, coefficients))
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime_seconds = time.time() - started
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_role_sign_carrier_loo_v1",
        "classification": classification,
        "retained": transfer["retained"],
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": {
            **function_manifest,
            "task_atom": "NS5 first task gradient at W0",
            "canonical_shape": list(CANONICAL_SHAPE),
            "role_bitplanes": 2,
            "persistent_dense_float_basis": False,
        },
        "self_test": self_test(args.device),
        "transfer": transfer,
        "execution": {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
            "entrypoint": str(script),
            "entrypoint_sha256": sha256(script),
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "plan": str(args.plan),
            "plan_sha256": sha256(args.plan),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": runtime_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)},
            "compact_checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        },
        "limitations": [
            "H40 tests only two role-specific static polarity fields around one task atom.",
            "A pass authorizes a remaining-PC and transfer audit, never CE or scale-up.",
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
                        "fit_all_median_capture": row["fit_all_median_capture"],
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
