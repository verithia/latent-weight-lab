from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    PairedCoordinateField,
    channel_encoding,
    coordinate_count,
    decoder_parameter_count,
    fit_field,
    function_and_jvp,
    result_authorization,
)


def test_candidate_and_control_start_from_the_same_decoder() -> None:
    candidate = _small_field(seed=19)
    control = _small_field(seed=19)
    for left, right in zip(candidate.parameters(), control.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_registered_coordinate_budget_exceeds_200x() -> None:
    assert decoder_parameter_count(23, 64) == 5826
    per_layer = coordinate_count(
        experts=8,
        hidden_width=1536,
        code_width=6,
        decoder_input=23,
        decoder_hidden=64,
    )
    assert per_layer == 91844
    dense = 8 * 2 * 1536 * 768
    assert dense / per_layer > 205.5


def test_channel_encoding_is_fixed_and_finite() -> None:
    encoded = channel_encoding(17, 8, "cpu")
    assert encoded.shape == (17, 17)
    assert torch.isfinite(encoded).all()
    torch.testing.assert_close(encoded[0, 0], torch.tensor(-1.0))
    torch.testing.assert_close(encoded[-1, 0], torch.tensor(1.0))


def _small_field(seed: int = 7) -> PairedCoordinateField:
    return PairedCoordinateField(
        experts=2,
        input_width=8,
        hidden_width=12,
        code_width=3,
        encoding_frequencies=2,
        decoder_hidden_width=8,
        layer=0,
        tensor_layers=2,
        seed=seed,
        device="cpu",
        channel_chunk=4,
    )


def test_joint_materialization_is_finite_and_differentiable() -> None:
    field = _small_field()
    c_fc, c_proj_atoms = field.materialize()
    assert c_fc.shape == (2, 12, 8)
    assert c_proj_atoms.shape == (2, 12, 8)
    inputs = torch.randn(2, 5, 8)
    directions = torch.randn_like(inputs)
    output, jvp = field.function_and_jvp(inputs, directions)
    (output.square().mean() + jvp.square().mean()).backward()
    assert field.codes.grad is not None and torch.isfinite(field.codes.grad).all()
    assert all(parameter.grad is not None for parameter in field.decoder.parameters())


def test_analytic_jvp_matches_finite_difference() -> None:
    torch.manual_seed(9)
    inputs = torch.randn(2, 4, 8)
    directions = torch.randn_like(inputs)
    c_fc = torch.randn(2, 12, 8) * 0.1
    c_proj_atoms = torch.randn(2, 12, 8) * 0.1
    bias = torch.randn(2, 12) * 0.05
    output, jvp = function_and_jvp(inputs, directions, c_fc, c_proj_atoms, bias)
    epsilon = 1e-3
    plus, _ = function_and_jvp(inputs + epsilon * directions, directions, c_fc, c_proj_atoms, bias)
    minus, _ = function_and_jvp(inputs - epsilon * directions, directions, c_fc, c_proj_atoms, bias)
    torch.testing.assert_close(jvp, (plus - minus) / (2 * epsilon), atol=2e-4, rtol=2e-3)
    assert torch.isfinite(output).all()


def test_fit_reduces_joint_synthetic_objective() -> None:
    torch.manual_seed(11)
    field = _small_field()
    inputs = torch.randn(2, 24, 8)
    c_fc = torch.randn(2, 12, 8) * 0.1
    c_proj = torch.randn(2, 8, 12) * 0.1
    diagnostics = fit_field(
        field,
        inputs,
        c_fc,
        c_proj,
        steps=20,
        decoder_learning_rate=0.01,
        coordinate_learning_rate=0.02,
        decoder_weight_decay=0.0,
        code_weight_decay=0.0,
        gradient_clip=10.0,
        jvp_weight=0.1,
        probe_seed=13,
        train_decoder=True,
    )
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]
    assert math.isfinite(diagnostics["maximum_preclip_gradient_norm"])


def test_authorization_stops_before_training() -> None:
    passed = result_authorization(True)
    failed = result_authorization(False)
    assert passed["implementation"]
    assert passed["initialization_and_mapping_loss_shadow"]
    assert not passed["language_model_training"]
    assert not passed["mfu_preflight"]
    assert not failed["implementation"]
