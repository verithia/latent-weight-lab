from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState
from examples.nanogpt.analyze_sparse_moe_stepzero_kfac_maxbudget_oracle import (
    asymmetric_overlap,
    coordinate_compression,
    reconstruct_asymmetric_family,
    result_authorization,
)


def _state(scale: float) -> LayerState:
    router = scale * torch.arange(12, dtype=torch.float32).reshape(2, 6)
    c_fc = scale * torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)
    c_proj = scale * torch.arange(48, dtype=torch.float32).reshape(2, 6, 4)
    return LayerState(router, c_fc, c_proj)


def test_coordinate_compression_has_unique_200x_valid_seven_coordinate_cap() -> None:
    assert abs(coordinate_compression(3, 4) - 1536.0 / 7.0) < 1e-12
    assert coordinate_compression(3, 4) > 200.0
    assert coordinate_compression(4, 4) < 200.0


def test_asymmetric_reconstruction_uses_fourth_direction_only_for_output() -> None:
    left = _state(0.0)
    directions = [_state(float(index + 1)) for index in range(4)]
    fourth = directions[3]
    target = (
        torch.zeros_like(fourth.router),
        torch.zeros_like(fourth.c_fc),
        fourth.c_proj.clone(),
    )
    reconstructed, metrics = reconstruct_asymmetric_family(
        left,
        target,
        directions,
        incoming_rank=3,
        outgoing_rank=4,
        router_rank=3,
        ridge_ratio=1e-12,
        device="cpu",
    )
    assert reconstructed.c_fc.square().sum() == 0
    assert metrics["c_proj_parameter_recovery"] > 0.999999


def test_identical_banks_have_unit_asymmetric_overlap() -> None:
    bank = [_state(float(index + 1)) for index in range(4)]
    overlap = asymmetric_overlap(
        bank, bank, incoming_rank=3, outgoing_rank=4
    )
    assert overlap["incoming"] > 0.999999
    assert overlap["outgoing"] > 0.999999


def test_pass_still_does_not_authorize_fit_or_training() -> None:
    authorization = result_authorization(True)
    assert authorization["structured_basis_approximation_preregistration"]
    assert not authorization["dense_or_lora_basis"]
    assert not authorization["production_implementation"]
    assert not authorization["mfu_preflight"]
    assert not authorization["language_model_training"]
