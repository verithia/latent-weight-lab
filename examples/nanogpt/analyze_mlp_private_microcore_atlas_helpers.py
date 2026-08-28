"""Accounting, checkpoint, and self-test helpers for H45."""
from __future__ import annotations

from typing import Any

import torch

from examples.nanogpt.analyze_mlp_role_wide_givens_atlas import (
    BRANCHES,
    PROMPT_LENGTH,
    PROMPT_WIDTH,
    projection_metrics,
)
from examples.nanogpt.analyze_mlp_shared_separable_conditioner_loo import (
    normalized,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program import (
    DENSE_MLP_SCALARS,
)
from examples.nanogpt.analyze_mlp_task_modulated_microcore_atlas import (
    initial_core,
    make_hash_geometry,
    microcore_basis,
)


CORE_SIZE = 96
DEPLOYED_MLP_MATRICES = 24
OBSERVED_DEPLOYED_SLOTS = (0, 1, 12, 13, 22, 23)
HASH_SEED_BASE = 877


def core_initialization_seed(deployed_slot: int) -> int:
    return 9001 + 97 * deployed_slot


def checkpoint_accounting() -> dict[str, int | float]:
    prompt_scalars = PROMPT_LENGTH * PROMPT_WIDTH
    private_core_scalars = DEPLOYED_MLP_MATRICES * CORE_SIZE * CORE_SIZE
    local_coefficient_scalars = DEPLOYED_MLP_MATRICES * BRANCHES
    total_scalars = prompt_scalars + private_core_scalars + local_coefficient_scalars
    return {
        "prompt_scalars": prompt_scalars,
        "deployed_private_cores": DEPLOYED_MLP_MATRICES,
        "core_size": CORE_SIZE,
        "private_core_scalars": private_core_scalars,
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


def compact_checkpoint_bytes(
    prompt: torch.Tensor, learned_cores: dict[int, torch.Tensor]
) -> bytes:
    device = prompt.device
    chunks = [prompt.detach().half().cpu().contiguous().numpy().tobytes()]
    for deployed_slot in range(DEPLOYED_MLP_MATRICES):
        core = learned_cores.get(
            deployed_slot,
            initial_core(
                CORE_SIZE,
                seed=core_initialization_seed(deployed_slot),
                device=device,
            ),
        )
        chunks.append(core.half().cpu().contiguous().numpy().tobytes())
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
    torch.manual_seed(563)
    shape = (16, 8)
    core_size = 4
    branches = 8
    components = 4
    atom = normalized(torch.randn(shape, device=device))
    core = initial_core(core_size, seed=109, device=device)
    geometry = make_hash_geometry(
        shape,
        core_size=core_size,
        branches=branches,
        seed_base=HASH_SEED_BASE,
        node_index=12,
        device=device,
    )
    basis = microcore_basis(atom, core, geometry)
    regenerated = microcore_basis(
        atom,
        initial_core(core_size, seed=109, device=device),
        make_hash_geometry(
            shape,
            core_size=core_size,
            branches=branches,
            seed_base=HASH_SEED_BASE,
            node_index=12,
            device=device,
        ),
    )
    if not torch.equal(basis, regenerated):
        raise AssertionError("private microcore regeneration is not exact")
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
        raise AssertionError("private microcore basis has no finite nonzero gradient")
    coefficients = torch.randn(components, branches, device=device)
    targets = coefficients @ basis
    targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-20)
    weights = torch.full((components,), 1.0 / components, device=device)
    weighted, metrics = projection_metrics(
        basis, targets, weights, relative_ridge=1e-8, differentiable=False
    )
    if float(weighted) < 0.90:
        raise AssertionError({"weighted_capture": float(weighted), "metrics": metrics})
    accounting = checkpoint_accounting()
    if accounting["total_state_scalars"] != 541_440:
        raise AssertionError(accounting)
    prompt = torch.zeros(PROMPT_LENGTH, PROMPT_WIDTH, device=device)
    payload = compact_checkpoint_bytes(prompt, {})
    if len(payload) != accounting["fp16_checkpoint_bytes"]:
        raise AssertionError("private compact checkpoint byte mismatch")
    return {
        "status": "passed",
        "minimum_known_microcore_weighted_capture": float(weighted),
        "branch_zero_identity": True,
        "deterministic_regeneration": True,
        "finite_nonzero_core_gradient": True,
        "accounting": accounting,
    }
