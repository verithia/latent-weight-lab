from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_synthetic_muon_program import (
    initialization_match,
    latent_accounting,
    principal_component,
    selected_parameter_names,
    self_test,
)


def test_exact_h29a_accounting() -> None:
    record = latent_accounting(737, 768)
    assert record["prompt_scalars"] == 566_016
    assert record["amplitude_scalars"] == 24
    assert record["total_scalars"] == 566_040
    assert record["deployable_checkpoint_fp16_bytes"] == 1_132_080
    assert 0.00999 < float(record["deployable_scalar_fraction"]) <= 0.01


def test_principal_component_recovers_rank_one_path() -> None:
    direction = torch.randn(7, 5)
    coefficients = torch.linspace(-2, 2, 19)
    states = coefficients[:, None, None] * direction
    pc, fraction, _ = principal_component(states, anchored=False)
    cosine = abs(float((pc * direction).sum() / (pc.norm() * direction.norm())))
    assert cosine > 0.999999
    assert fraction > 0.999999


def test_mixed_hessian_polar_projection_self_test() -> None:
    record = self_test("cpu")
    assert float(record["path_energy_capture"]) > 0.999


def test_fp16_storage_roundtrip_is_accepted() -> None:
    reconstructed = torch.randn(64, 32, dtype=torch.float32)
    stored = reconstructed.to(torch.float16)
    record = initialization_match(reconstructed, stored)
    assert record["storage_roundtrip_bitwise_equal"] is True
    assert record["accepted"] is True


def test_cross_depth_parameter_inventory() -> None:
    assert selected_parameter_names([0, 11], ["c_fc", "c_proj"]) == [
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
