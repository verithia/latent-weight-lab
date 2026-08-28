#!/usr/bin/env python3
"""H33a leave-one-node-out gate for a shared sign-preconditioned task atom."""
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


PROMPT_LENGTH = 545
PROMPT_WIDTH = 768
CANONICAL_SHAPE = (3072, 768)
GLOBAL_BITPLANE_BITS = 3072 * 768
DEPLOYED_MLP_MATRICES = 24
RANDOM_MASK_SEED = 20261001


def checkpoint_accounting() -> dict[str, int | float]:
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    prompt_fp16_bytes = 2 * prompt_scalars
    bitplane_bytes = GLOBAL_BITPLANE_BITS // 8
    node_coefficient_fp16_bytes = 2 * DEPLOYED_MLP_MATRICES
    total_bytes = prompt_fp16_bytes + bitplane_bytes + node_coefficient_fp16_bytes
    dense_fp16_bytes = 2 * DENSE_MLP_SCALARS
    return {
        "prompt_scalars": prompt_scalars,
        "prompt_fp16_bytes": prompt_fp16_bytes,
        "global_bitplane_bits": GLOBAL_BITPLANE_BITS,
        "global_bitplane_bytes": bitplane_bytes,
        "node_coefficient_scalars": DEPLOYED_MLP_MATRICES,
        "node_coefficient_fp16_bytes": node_coefficient_fp16_bytes,
        "total_compact_checkpoint_bytes": total_bytes,
        "dense_mlp_fp16_denominator_bytes": dense_fp16_bytes,
        "checkpoint_byte_fraction": total_bytes / dense_fp16_bytes,
        "fp16_equivalent_scalars": total_bytes / 2,
    }


def canonicalize(parameter: str, value: torch.Tensor) -> torch.Tensor:
    canonical = value.T if parameter.endswith("c_proj.weight") else value
    if tuple(canonical.shape) != CANONICAL_SHAPE:
        raise ValueError(f"unexpected canonical shape for {parameter}: {canonical.shape}")
    return canonical


def normalized(value: torch.Tensor) -> torch.Tensor:
    work = value.detach().float()
    norm = work.norm()
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise ValueError("cannot normalize a zero or nonfinite node")
    return work / norm


def sign_from_score(score: torch.Tensor) -> torch.Tensor:
    return torch.where(score >= 0, torch.ones_like(score), -torch.ones_like(score))


def squared_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_work = left.float().flatten()
    right_work = right.float().flatten()
    denominator = left_work.square().sum() * right_work.square().sum()
    return float((left_work @ right_work).square() / denominator)


def bitplane_sha256(mask: torch.Tensor) -> str:
    bits = (mask.detach().flatten().cpu().numpy() > 0).astype(np.uint8)
    packed = np.packbits(bits, bitorder="little")
    if packed.nbytes != GLOBAL_BITPLANE_BITS // 8:
        raise ValueError("bitplane packing mismatch")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def shared_sign_transfer(
    atoms: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    random_seed: int,
) -> dict[str, Any]:
    if len(atoms) != len(targets) or len(atoms) < 2:
        raise ValueError("shared-sign audit requires matching multiple nodes")
    atom_rows = tuple(normalized(value) for value in atoms)
    target_rows = tuple(normalized(value) for value in targets)
    for value in (*atom_rows, *target_rows):
        if tuple(value.shape) != CANONICAL_SHAPE:
            raise ValueError("shared-sign audit received a noncanonical node")

    generator = torch.Generator(device=atom_rows[0].device)
    generator.manual_seed(random_seed)
    random_mask = torch.randint(
        0,
        2,
        CANONICAL_SHAPE,
        generator=generator,
        device=atom_rows[0].device,
        dtype=torch.int8,
    ).float().mul_(2.0).sub_(1.0)
    all_score = sum(atom * target for atom, target in zip(atom_rows, target_rows, strict=True))
    all_mask = sign_from_score(all_score)

    rows = []
    for heldout in range(len(atom_rows)):
        train_score = sum(
            atom_rows[index] * target_rows[index]
            for index in range(len(atom_rows))
            if index != heldout
        )
        mask = sign_from_score(train_score)
        rows.append(
            {
                "heldout_index": heldout,
                "leave_one_out_capture": squared_cosine(
                    mask * atom_rows[heldout], target_rows[heldout]
                ),
                "fit_all_capture": squared_cosine(
                    all_mask * atom_rows[heldout], target_rows[heldout]
                ),
                "raw_no_mask_capture": squared_cosine(
                    atom_rows[heldout], target_rows[heldout]
                ),
                "random_mask_capture": squared_cosine(
                    random_mask * atom_rows[heldout], target_rows[heldout]
                ),
                "leave_one_out_positive_fraction": float((mask > 0).float().mean()),
                "leave_one_out_mask_sha256": bitplane_sha256(mask),
            }
        )
    loo = [row["leave_one_out_capture"] for row in rows]
    return {
        "rows": rows,
        "minimum_leave_one_out_capture": min(loo),
        "median_leave_one_out_capture": statistics.median(loo),
        "maximum_leave_one_out_capture": max(loo),
        "fit_all_mask_positive_fraction": float((all_mask > 0).float().mean()),
        "fit_all_mask_sha256": bitplane_sha256(all_mask),
        "random_mask_positive_fraction": float((random_mask > 0).float().mean()),
        "random_mask_sha256": bitplane_sha256(random_mask),
        "random_mask_seed": random_seed,
    }


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(101)
    shared = torch.where(
        torch.rand(CANONICAL_SHAPE, device=device) > 0.5,
        torch.tensor(1.0, device=device),
        torch.tensor(-1.0, device=device),
    )
    atoms = tuple(torch.randn(CANONICAL_SHAPE, device=device) for _ in range(6))
    targets = tuple(shared * atom + 0.01 * torch.randn_like(atom) for atom in atoms)
    result = shared_sign_transfer(atoms, targets, random_seed=RANDOM_MASK_SEED)
    if result["minimum_leave_one_out_capture"] < 0.99:
        raise AssertionError(result)
    accounting = checkpoint_accounting()
    if accounting["total_compact_checkpoint_bytes"] != 1_132_080:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "minimum_leave_one_out_capture": result["minimum_leave_one_out_capture"],
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
    if accounting["checkpoint_byte_fraction"] > 0.01:
        raise ValueError("H33a exceeds the one-percent byte budget")
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    if plan["frozen_decoder"]["total_compact_checkpoint_bytes"] != 1_132_080:
        raise ValueError("plan/accounting mismatch")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    model = build_dense_model(config, args.device)
    prompt, targets, prompt_manifest = make_prompt(
        model,
        config,
        prompt_length=PROMPT_LENGTH,
        device=args.device,
    )
    joint_target, leading_fraction, target_manifest, w0_references = joint_leading_pc(
        args.trajectory_dir,
        parameters=FROZEN_PARAMETERS,
        device=args.device,
    )
    model_parameters = dict(model.named_parameters())
    w0_matches = {
        parameter: initialization_match(
            model_parameters[parameter], w0_references[parameter]
        )
        for parameter in FROZEN_PARAMETERS
    }
    if not all(bool(record["accepted"]) for record in w0_matches.values()):
        raise ValueError(f"model/trajectory W0 mismatch: {w0_matches}")

    _, loss_function, initial_weights, function_manifest = make_model_lookahead_program(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=targets,
        ns_steps=5,
        momentum=0.0,
    )
    gradient_function = torch.func.grad(loss_function, argnums=0)
    gradients = gradient_function(initial_weights, prompt)
    atoms_raw = tuple(
        zeropower_via_newtonschulz5(gradient, steps=5).detach()
        for gradient in gradients
    )
    split_targets = torch.split(
        joint_target,
        [weight.numel() for weight in initial_weights],
    )
    atoms = tuple(
        canonicalize(parameter, atom)
        for parameter, atom in zip(FROZEN_PARAMETERS, atoms_raw, strict=True)
    )
    target_parts = tuple(
        canonicalize(parameter, part.reshape_as(weight))
        for parameter, part, weight in zip(
            FROZEN_PARAMETERS, split_targets, initial_weights, strict=True
        )
    )
    transfer = shared_sign_transfer(atoms, target_parts, random_seed=RANDOM_MASK_SEED)
    minimum_gate = transfer["minimum_leave_one_out_capture"] >= 0.05
    median_gate = transfer["median_leave_one_out_capture"] >= 0.10
    retained = minimum_gate and median_gate
    transfer["minimum_gate"] = minimum_gate
    transfer["median_gate"] = median_gate
    transfer["retained"] = retained
    transfer["leading_pc_energy_fraction"] = leading_fraction

    torch.cuda.synchronize()
    accounting_path = output / "accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_shared_sign_preconditioner_loo_v1",
        "classification": "RETAINED" if retained else "REJECTED",
        "retained": retained,
        "plan": plan,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "target_manifest": target_manifest,
        "w0_storage_matches": w0_matches,
        "function_manifest": {
            **function_manifest,
            "task_atom": "NS5 first task gradient at W0",
            "canonical_shape": list(CANONICAL_SHAPE),
            "shared_bitplanes": 1,
            "persistent_dense_float_basis": False,
        },
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
            "runtime_seconds": time.time() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)}
        },
        "limitations": [
            "The learned sign controls are fitted to one registered joint PC, but every selection score is leave-one-node-out.",
            "A pass authorizes a tangent/transfer audit, never CE or scale-up.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "classification": metadata["classification"],
                "minimum_leave_one_out_capture": transfer["minimum_leave_one_out_capture"],
                "median_leave_one_out_capture": transfer["median_leave_one_out_capture"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
