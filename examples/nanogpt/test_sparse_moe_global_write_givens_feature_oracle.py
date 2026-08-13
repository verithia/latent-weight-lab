from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_global_write_givens_feature_oracle import (
    GlobalWriteGivensFeatures,
    apply_givens_stages,
    coordinate_count,
    fixed_matchings,
    validate_plan,
)


def _module(stages: int) -> GlobalWriteGivensFeatures:
    write, _ = torch.linalg.qr(torch.randn(8, 5))
    return GlobalWriteGivensFeatures(
        write_basis=write,
        tensor_layers=2,
        experts=2,
        feature_seed=7,
        matching_seed=11,
        stages=stages,
        device="cpu",
    )


def test_registered_coordinate_count_is_exact_and_above_200x() -> None:
    count = coordinate_count(
        rank=561, input_width=768, tensor_layers=12, experts=8, stages=4
    )
    assert count == 1130784
    assert 226492416 / count > 200.29


def test_live_count_includes_fixed_write_atlas() -> None:
    candidate = _module(4)
    control = _module(0)
    assert candidate.counted_coordinates() == coordinate_count(
        rank=5, input_width=8, tensor_layers=2, experts=2, stages=4
    )
    assert control.counted_coordinates() == coordinate_count(
        rank=5, input_width=8, tensor_layers=2, experts=2, stages=0
    )


def test_givens_stages_preserve_norm() -> None:
    torch.manual_seed(13)
    values = torch.randn(2, 7, 5)
    angles = torch.randn(2, 4, 2)
    matchings = fixed_matchings(5, 4, 17)
    rotated = apply_givens_stages(values, angles, matchings)
    torch.testing.assert_close(
        rotated.square().sum(-1), values.square().sum(-1),
        rtol=1e-5, atol=1e-5,
    )


def test_step_zero_output_and_jvp_are_exactly_zero() -> None:
    inputs = torch.randn(2, 6, 8)
    directions = torch.randn_like(inputs)
    for stages in (0, 4):
        module = _module(stages)
        output, jvp = module.function_and_jvp(inputs, directions, layer=1)
        assert torch.count_nonzero(output) == 0
        assert torch.count_nonzero(jvp) == 0


def test_feature_rows_have_fixed_norm() -> None:
    module = _module(4)
    expected = 0.02 * (8.0 ** 0.5)
    torch.testing.assert_close(
        module.feature_basis().norm(dim=-1),
        torch.full((5,), expected),
    )
    with torch.no_grad():
        module.raw_feature.mul_(9.0)
    torch.testing.assert_close(
        module.feature_basis().norm(dim=-1),
        torch.full((5,), expected),
    )


def test_analytic_jvp_matches_finite_difference() -> None:
    torch.manual_seed(19)
    module = _module(4)
    with torch.no_grad():
        module.output_gain.normal_(std=0.1)
        module.input_gain.normal_(mean=1.0, std=0.05)
        module.hidden_bias.normal_(std=0.05)
        module.angles.normal_(std=0.1)
    inputs = torch.randn(2, 4, 8)
    directions = torch.randn_like(inputs)
    _output, jvp = module.function_and_jvp(inputs, directions, layer=0)
    epsilon = 1e-3
    plus, _ = module.function_and_jvp(
        inputs + epsilon * directions, torch.zeros_like(inputs), layer=0
    )
    minus, _ = module.function_and_jvp(
        inputs - epsilon * directions, torch.zeros_like(inputs), layer=0
    )
    finite = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(jvp, finite, rtol=4e-3, atol=4e-5)


def test_same_seed_candidate_and_control_share_feature_initialization() -> None:
    candidate, control = _module(4), _module(0)
    for name in ("raw_feature", "input_gain", "hidden_bias", "output_gain"):
        torch.testing.assert_close(getattr(candidate, name), getattr(control, name))


def test_preregistered_plan_is_hash_sealed() -> None:
    plan_path = (
        Path(__file__).parent / "configs" / "selection_artifacts"
        / "124m_sparse_moe_global_write_givens_feature_oracle_plan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, plan_path)
    assert plan["identity"]["theory_preregistration_git_commit"] == (
        "f1db3a0b24a214013ca2ed041a9d3ce790cdff3e"
    )
