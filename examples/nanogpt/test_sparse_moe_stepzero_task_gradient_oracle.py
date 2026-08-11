from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    family_overlap,
    reconstruct_gradient_family,
    row_span_overlap,
    stepzero_optimizer_action,
    walsh_token_mask,
)
from examples.nanogpt.muon import muon_update_batched
from examples.nanogpt.analyze_sparse_moe_paired_atom_oracle import (
    _state_chord,
    energy_recovery,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState


def _zero_state(experts: int = 2, hidden: int = 5, width: int = 4) -> LayerState:
    return LayerState(
        torch.zeros(experts, width),
        torch.zeros(experts, hidden, width),
        torch.zeros(experts, width, hidden),
    )


def test_coupled_gradient_bank_recovers_in_span_target() -> None:
    torch.manual_seed(41)
    left = _zero_state()
    directions = [
        LayerState(
            torch.randn_like(left.router),
            torch.randn_like(left.c_fc),
            torch.randn_like(left.c_proj),
        )
        for _ in range(4)
    ]
    router_coefficients = torch.randn(2, 4)
    pair_coefficients = torch.randn(2, 5, 4)
    target_router = torch.einsum(
        "ek,ker->er", router_coefficients, torch.stack([x.router for x in directions])
    )
    target_fc = torch.einsum(
        "ehk,kehd->ehd", pair_coefficients, torch.stack([x.c_fc for x in directions])
    )
    target_proj_atoms = torch.einsum(
        "ehk,kehd->ehd",
        pair_coefficients,
        torch.stack([x.c_proj.transpose(1, 2) for x in directions]),
    )
    target = (target_router, target_fc, target_proj_atoms)
    reconstructed, recoveries = reconstruct_gradient_family(
        left, target, directions, "coupled_four", 1e-10, "cpu"
    )
    actual = _state_chord(reconstructed, left)
    assert recoveries["paired_parameter_recovery"] > 0.999999
    assert recoveries["router_parameter_recovery"] > 0.999999
    assert energy_recovery(actual[1], target_fc) > 0.999999
    assert energy_recovery(actual[2], target_proj_atoms) > 0.999999


def test_identical_gradient_banks_have_unit_overlap() -> None:
    torch.manual_seed(43)
    bank = [
        LayerState(torch.randn(2, 4), torch.randn(2, 5, 4), torch.randn(2, 4, 5))
        for _ in range(4)
    ]
    assert family_overlap(bank, bank, "coupled_four") > 0.999999
    assert family_overlap(bank, bank, "separate_three_plus_three") > 0.999999


def test_orthogonal_row_banks_have_zero_overlap() -> None:
    left = torch.zeros(2, 1, 1, 4)
    right = torch.zeros_like(left)
    left[0, 0, 0, 0] = 1.0
    left[1, 0, 0, 1] = 1.0
    right[0, 0, 0, 2] = 1.0
    right[1, 0, 0, 3] = 1.0
    assert row_span_overlap(left, right) < 1e-8


def test_raw_optimizer_action_is_negative_gradient() -> None:
    torch.manual_seed(47)
    gradient = LayerState(torch.randn(2, 4), torch.randn(2, 5, 4), torch.randn(2, 4, 5))
    parameter = _zero_state(experts=2, hidden=5, width=4)
    action = stepzero_optimizer_action(
        gradient, parameter, direction_transform="raw"
    )
    torch.testing.assert_close(action.router, -gradient.router)
    torch.testing.assert_close(action.c_fc, -gradient.c_fc)
    torch.testing.assert_close(action.c_proj, -gradient.c_proj)


def test_muon_action_matches_batched_polar_and_first_adam_step() -> None:
    torch.manual_seed(53)
    gradient = LayerState(torch.randn(2, 4), torch.randn(2, 5, 4), torch.randn(2, 4, 5))
    parameter = LayerState(torch.randn(2, 4), torch.randn(2, 5, 4), torch.randn(2, 4, 5))
    action = stepzero_optimizer_action(
        gradient,
        parameter,
        direction_transform="muon_action",
        muon_ns_steps=5,
        weight_decay=0.1,
        adam_epsilon=1e-8,
    )
    expected_router = -gradient.router / (gradient.router.abs() + 1e-8)
    expected_fc = -muon_update_batched(gradient.c_fc, steps=5) - 0.1 * parameter.c_fc
    expected_proj = -muon_update_batched(gradient.c_proj, steps=5) - 0.1 * parameter.c_proj
    torch.testing.assert_close(action.router, expected_router)
    torch.testing.assert_close(action.c_fc, expected_fc)
    torch.testing.assert_close(action.c_proj, expected_proj)


def test_first_four_walsh_token_masks_are_orthogonal() -> None:
    masks = torch.stack([walsh_token_mask(2048, coordinate, "cpu") for coordinate in range(4)])
    gram = masks @ masks.T
    torch.testing.assert_close(gram, 2048 * torch.eye(4))


def test_jacobian_sketch_action_preserves_raw_gradient_orientation() -> None:
    torch.manual_seed(59)
    gradient = LayerState(torch.randn(2, 4), torch.randn(2, 5, 4), torch.randn(2, 4, 5))
    parameter = _zero_state(experts=2, hidden=5, width=4)
    action = stepzero_optimizer_action(
        gradient, parameter, direction_transform="jacobian_sketch"
    )
    torch.testing.assert_close(action.router, -gradient.router)
    torch.testing.assert_close(action.c_fc, -gradient.c_fc)
    torch.testing.assert_close(action.c_proj, -gradient.c_proj)
