from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_shared_sign_preconditioner_loo import (
    CANONICAL_SHAPE,
    RANDOM_MASK_SEED,
    checkpoint_accounting,
    shared_sign_transfer,
)


def test_checkpoint_accounting_is_exact_and_below_one_percent() -> None:
    accounting = checkpoint_accounting()
    assert accounting["prompt_scalars"] == 418_560
    assert accounting["global_bitplane_bits"] == 2_359_296
    assert accounting["global_bitplane_bytes"] == 294_912
    assert accounting["total_compact_checkpoint_bytes"] == 1_132_080
    assert accounting["checkpoint_byte_fraction"] < 0.01


def test_leave_one_out_recovers_a_truly_shared_sign_relation() -> None:
    torch.manual_seed(9)
    shared = torch.where(
        torch.rand(CANONICAL_SHAPE) > 0.5,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    atoms = tuple(torch.randn(CANONICAL_SHAPE) for _ in range(6))
    targets = tuple(shared * atom + 0.01 * torch.randn_like(atom) for atom in atoms)
    result = shared_sign_transfer(atoms, targets, random_seed=RANDOM_MASK_SEED)
    assert result["minimum_leave_one_out_capture"] > 0.99
    assert result["median_leave_one_out_capture"] > 0.99
    assert len(result["rows"]) == 6
    assert len(result["fit_all_mask_sha256"]) == 64
