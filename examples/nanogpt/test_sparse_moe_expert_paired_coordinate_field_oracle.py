from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_expert_paired_coordinate_field_oracle import (
    ExpertPairedCoordinateField,
    coordinate_count,
    decoder_parameter_count,
    result_authorization,
    validate_plan,
)
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    fit_field,
)


def _small_field(seed: int = 7) -> ExpertPairedCoordinateField:
    return ExpertPairedCoordinateField(
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


def test_registered_coordinate_budget_exceeds_200x() -> None:
    assert decoder_parameter_count(20, 56) == 4482
    per_layer = coordinate_count(
        experts=8,
        hidden_width=1536,
        code_width=3,
        decoder_input=20,
        decoder_hidden=56,
    )
    assert per_layer == 85024
    dense = 8 * 2 * 1536 * 768
    assert dense / per_layer > 221.98


def test_materialization_is_joint_finite_and_expert_specific() -> None:
    field = _small_field()
    c_fc, c_proj_atoms = field.materialize()
    assert c_fc.shape == (2, 12, 8)
    assert c_proj_atoms.shape == (2, 12, 8)
    inputs = torch.randn(2, 5, 8)
    directions = torch.randn_like(inputs)
    output, jvp = field.function_and_jvp(inputs, directions)
    (output[0].square().mean() + jvp[0].square().mean()).backward()
    assert torch.isfinite(field.decoder_weight_1.grad).all()
    assert float(field.decoder_weight_1.grad[0].abs().sum()) > 0.0
    assert float(field.decoder_weight_1.grad[1].abs().sum()) == 0.0


def test_same_seed_is_exact() -> None:
    left = _small_field(seed=19)
    right = _small_field(seed=19)
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_synthetic_fit_reduces_objective() -> None:
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


def test_preregistered_plan_is_hash_sealed() -> None:
    plan_path = (
        Path(__file__).parent
        / "configs"
        / "selection_artifacts"
        / "124m_sparse_moe_expert_paired_coordinate_field_oracle_plan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, plan_path)
    assert plan["identity"]["preregistration_git_commit"] == (
        "e6ff7423b49bd50f080014532f0c9e7c81200e87"
    )
    assert plan["candidate"]["paired_parameter_compression_ratio"] > 200.0
