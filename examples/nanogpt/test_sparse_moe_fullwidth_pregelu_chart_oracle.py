from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_fullwidth_pregelu_chart_oracle import (
    FullWidthPreGeluChart,
    coordinate_count,
    procedural_signs,
    validate_plan,
)


def _module(*, learn_angles: bool) -> FullWidthPreGeluChart:
    write, _ = torch.linalg.qr(torch.randn(8, 5))
    return FullWidthPreGeluChart(
        write_basis=write,
        hidden_width=12,
        padded_width=16,
        tensor_layers=2,
        experts=2,
        feature_seed=7,
        pre_matching_seed=11,
        post_matching_seed=13,
        procedural_map_seed=17,
        learn_angles=learn_angles,
        device="cpu",
    )


def test_registered_coordinate_count_is_exact() -> None:
    assert coordinate_count(
        rank=480,
        input_width=768,
        hidden_width=1536,
        tensor_layers=12,
        experts=8,
        pre_stages=1,
        post_stages=1,
    ) == 1_124_352
    assert coordinate_count(
        rank=480,
        input_width=768,
        hidden_width=1536,
        tensor_layers=12,
        experts=8,
        pre_stages=0,
        post_stages=0,
    ) == 1_078_272


def test_live_count_includes_write_atlas_and_excludes_procedural_signs() -> None:
    candidate, control = _module(learn_angles=True), _module(learn_angles=False)
    assert candidate.counted_coordinates() == coordinate_count(
        rank=5,
        input_width=8,
        hidden_width=12,
        tensor_layers=2,
        experts=2,
        pre_stages=1,
        post_stages=1,
    )
    assert control.counted_coordinates() == coordinate_count(
        rank=5,
        input_width=8,
        hidden_width=12,
        tensor_layers=2,
        experts=2,
        pre_stages=0,
        post_stages=0,
    )
    assert "procedural_signs" not in candidate.state_dict()


def test_procedural_signs_are_replayable_and_node_distinct() -> None:
    first = procedural_signs(
        tensor_layers=2, experts=2, padded_width=16, base_seed=19
    )
    second = procedural_signs(
        tensor_layers=2, experts=2, padded_width=16, base_seed=19
    )
    torch.testing.assert_close(first, second)
    assert set(first.unique().tolist()) == {-1, 1}
    assert not torch.equal(first[0, 0, 0], first[1, 1, 0])
    assert not torch.equal(first[0, 0, 0], first[0, 0, 1])


def test_candidate_and_control_have_exact_same_nonangle_initial_state() -> None:
    candidate, control = _module(learn_angles=True), _module(learn_angles=False)
    for name in (
        "raw_feature",
        "input_gain",
        "hidden_bias",
        "output_gain",
        "write_basis",
        "procedural_signs",
    ):
        torch.testing.assert_close(getattr(candidate, name), getattr(control, name))
    assert candidate.angles is not None
    assert control.angles is None


def test_step_zero_output_and_jvp_are_exactly_zero() -> None:
    inputs = torch.randn(2, 6, 8)
    directions = torch.randn_like(inputs)
    for learn_angles in (False, True):
        module = _module(learn_angles=learn_angles)
        output, jvp = module.function_and_jvp(inputs, directions, layer=1)
        assert torch.count_nonzero(output) == 0
        assert torch.count_nonzero(jvp) == 0


def test_analytic_jvp_matches_finite_difference() -> None:
    torch.manual_seed(23)
    module = _module(learn_angles=True)
    with torch.no_grad():
        module.output_gain.normal_(std=0.1)
        module.input_gain.normal_(mean=1.0, std=0.05)
        module.hidden_bias.normal_(std=0.05)
        module.pre_angles.normal_(std=0.1)
        module.post_angles.normal_(std=0.1)
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
    torch.testing.assert_close(jvp, finite, rtol=8e-3, atol=8e-5)


def test_procedural_fullwidth_map_has_requested_dimensions() -> None:
    module = _module(learn_angles=True)
    inputs = torch.randn(2, 3, 8)
    compact = torch.randn(2, 3, 5)
    tangent = torch.randn_like(compact)
    expanded, expanded_jvp = module._fht_pair(
        compact,
        tangent,
        layer=0,
        selected=slice(None),
        input_sign_index=0,
        output_sign_index=1,
        output_width=12,
        scale=module.expansion_scale,
    )
    assert expanded.shape == expanded_jvp.shape == (2, 3, 12)
    output, output_jvp = module.function_and_jvp(
        inputs, torch.randn_like(inputs), layer=0
    )
    assert output.shape == output_jvp.shape == (2, 3, 8)


def test_preregistered_plan_and_helper_inventory_are_hash_sealed() -> None:
    plan_path = (
        Path(__file__).parent / "configs" / "selection_artifacts"
        / "124m_sparse_moe_fullwidth_pregelu_chart_oracle_plan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, plan_path)
    assert plan["identity"]["theory_preregistration_git_commit"] == (
        "2f21fe2cee4623bffe5650be1d7ede9af31193b1"
    )
