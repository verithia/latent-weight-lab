from __future__ import annotations

import json
import math
from pathlib import Path


PLAN_PATH = (
    Path(__file__).parent
    / "configs"
    / "selection_artifacts"
    / "124m_sparse_moe_fullwidth_pregelu_chart_oracle_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_candidate_coordinate_accounting_is_exact_and_above_200x() -> None:
    plan = _plan()
    source, candidate = plan["source"], plan["candidate"]
    rank = candidate["rank"]
    nodes = source["tensor_layers"] * source["num_experts"]
    expected = (
        2 * rank * source["input_width"]
        + nodes
        * (
            candidate["givens_pairs_per_stage"]
            + candidate["hidden_width"]
            + candidate["hidden_width"]
            + candidate["givens_pairs_per_stage"]
            + rank
        )
    )
    assert expected == candidate["total_coordinates_all_layers"] == 1_124_352
    ratio = source["dense_paired_parameters_all_layers"] / expected
    assert math.isclose(ratio, candidate["paired_parameter_compression_ratio"])
    assert ratio >= plan["candidate_gates"]["paired_parameter_compression_ratio_min"]


def test_control_removes_only_givens_coordinates() -> None:
    plan = _plan()
    candidate, control = plan["candidate"], plan["location_control"]
    removed = (
        candidate["pre_gelu_angle_coordinates"]
        + candidate["post_gelu_angle_coordinates"]
    )
    assert candidate["total_coordinates_all_layers"] - removed == (
        control["total_coordinates_all_layers"]
    )
    assert control["paired_parameter_compression_ratio"] > 200.0


def test_procedural_widths_and_scales_are_frozen() -> None:
    plan = _plan()
    source = plan["source"]
    candidate = plan["candidate"]
    assert source["procedural_padded_width"] == 2048
    assert candidate["rank"] == 480
    assert candidate["hidden_width"] == source["expert_hidden_width"] == 1536
    assert candidate["fixed_dense_map_storage"] is False
    assert math.isclose(candidate["feature_scale"], 0.02 * math.sqrt(768))
    assert "sqrt(2048/480)" in plan["theory"]["procedural_expansion"]
    assert "sqrt(2048/1536)" in plan["theory"]["procedural_contraction"]


def test_plan_is_fail_closed_before_implementation() -> None:
    plan = _plan()
    assert plan["status"] == "theory_preregistered_before_implementation_or_candidate_values"
    assert plan["identity"]["theory_preregistration_git_commit"] is None
    assert plan["identity"]["entrypoint_sha256"] is None
    assert plan["identity"]["helper_sha256"] is None
    assert plan["authorization"] == {
        "implement_zero_update_oracle": True,
        "run_after_tests_identity_and_exact_runtime_gate": True,
        "causal_chart_implementation": False,
        "language_model_training": False,
        "larger_rung": False,
        "full_attention_work": False,
        "automatic_retry_or_sweep": False,
    }


def test_primary_and_escape_rules_do_not_select_on_heldout_values() -> None:
    plan = _plan()
    assert plan["decision_rule"]["no_automatic_promotion"] is True
    assert "Do not choose between candidate and control on heldout values" in (
        plan["control_escape_gate"]["rule"]
    )
    assert plan["candidate_gates"][
        "candidate_minus_location_control_recovery_mean_min_each_bank"
    ] == 0.05
    assert plan["candidate_gates"][
        "candidate_minus_location_control_jvp_recovery_mean_min_each_bank"
    ] == 0.05
