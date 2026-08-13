from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_shared_nonlinear_dictionary_oracle import (
    FAMILY_ORDER,
    SharedNonlinearDictionary,
    coordinate_count,
    result_authorization,
    validate_plan,
)


def _small(family: str, rank: int = 3) -> SharedNonlinearDictionary:
    return SharedNonlinearDictionary(
        family=family,
        rank=rank,
        tensor_layers=2,
        experts=2,
        input_width=8,
        seed=17,
        device="cpu",
    )


def test_registered_coordinate_counts_are_exact() -> None:
    expected = {
        "global_shared_rank619": 1129056,
        "layer_shared_rank60": 1123200,
        "expert_local_rank7": 1034208,
    }
    ranks = dict(zip(FAMILY_ORDER, (619, 60, 7)))
    for family in FAMILY_ORDER:
        assert coordinate_count(
            family=family, rank=ranks[family], tensor_layers=12,
            experts=8, input_width=768,
        ) == expected[family]
        assert 226492416 / expected[family] >= 200.0


def test_live_small_modules_match_coordinate_accounting() -> None:
    for family in FAMILY_ORDER:
        module = _small(family)
        assert module.compact_parameter_count() == coordinate_count(
            family=family, rank=3, tensor_layers=2, experts=2, input_width=8
        )


def test_step_zero_output_and_jvp_are_exactly_zero() -> None:
    inputs = torch.randn(2, 5, 8)
    directions = torch.randn_like(inputs)
    for family in FAMILY_ORDER:
        module = _small(family)
        output, jvp = module.function_and_jvp(
            inputs, directions, layer=1
        )
        assert torch.count_nonzero(output) == 0
        assert torch.count_nonzero(jvp) == 0


def test_analytic_jvp_matches_finite_difference() -> None:
    torch.manual_seed(23)
    for family in FAMILY_ORDER:
        module = _small(family)
        with torch.no_grad():
            module.output_gain.normal_(std=0.1)
            module.input_gain.normal_(mean=1.0, std=0.05)
            module.hidden_bias.normal_(std=0.05)
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


def test_single_expert_matches_batched_slice() -> None:
    module = _small("expert_local_rank7")
    with torch.no_grad():
        module.output_gain.normal_(std=0.1)
    inputs = torch.randn(2, 4, 8)
    directions = torch.randn_like(inputs)
    batched = module.function_and_jvp(inputs, directions, layer=1)
    single = module.function_and_jvp(
        inputs[1:2], directions[1:2], layer=1, expert=1
    )
    torch.testing.assert_close(single[0], batched[0][1:2])
    torch.testing.assert_close(single[1], batched[1][1:2])


def test_no_dense_base_and_authorization_stops_before_training() -> None:
    module = _small("global_shared_rank619")
    names = set(dict(module.named_parameters()))
    assert names == {
        "feature_basis", "write_basis", "input_gain", "hidden_bias",
        "output_gain",
    }
    passed = result_authorization(True)
    assert passed["production_implementation"]
    assert passed["initialization_and_mapping_loss_shadow"]
    assert not passed["language_model_training"]
    assert not passed["larger_rung"]


def test_preregistered_plan_is_hash_sealed() -> None:
    plan_path = (
        Path(__file__).parent / "configs" / "selection_artifacts"
        / "124m_sparse_moe_shared_nonlinear_dictionary_oracle_plan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, plan_path)
    assert plan["identity"]["theory_preregistration_git_commit"] == (
        "f4374394458824176e16be553e64ea963698b002"
    )
