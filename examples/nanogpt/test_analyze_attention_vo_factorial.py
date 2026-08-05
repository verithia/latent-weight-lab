import copy
import json

import pytest

from examples.nanogpt.analyze_attention_vo_factorial import PLAN, analyze


def _plan() -> dict:
    plan = json.loads(PLAN.read_text())
    plan["fixed_curves"] = {
        "steps": [1, 2, 3, 4],
        "qk_only_L00": [1.0] * 4,
        "qk_plus_v_L10": [1.02] * 4,
        "qk_plus_cproj_L01": "test",
        "qk_plus_v_plus_cproj_L11": [1.12] * 4,
    }
    return plan


def _result(plan: dict, curve: list[float]) -> dict:
    identity = plan["identity"]
    return {
        "run": {
            "config_sha256": identity["active_parent_config_sha256"],
            "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
            "fixed_eval_indices_sha256": identity["fixed_eval_indices_sha256"],
            "exit_code": 0,
            "classification": "clean",
            "fixed_evaluations": [
                {"step": step, "validation_ce": value}
                for step, value in zip([1, 2, 3, 4], curve)
            ],
        }
    }


def test_factorial_math_and_cproj_gate() -> None:
    plan = _plan()
    result = analyze(plan, _result(plan, [1.08] * 4))
    assert result["effects"]["v_shapley_ce"] == pytest.approx([0.03] * 4)
    assert result["effects"]["cproj_shapley_ce"] == pytest.approx([0.09] * 4)
    assert result["effects"]["v_by_cproj_interaction_ce"] == pytest.approx(
        [0.02] * 4
    )
    assert result["effects"]["identity_residual"] == pytest.approx([0.0] * 4)
    assert result["decision"]["authorize_cproj_ratio10"] is True
    assert result["decision_evidence"]["cproj_gate"] is True


def test_positive_interaction_can_authorize_capacity() -> None:
    plan = _plan()
    plan["fixed_curves"]["qk_plus_v_L10"] = [1.01] * 4
    plan["fixed_curves"]["qk_plus_v_plus_cproj_L11"] = [1.08] * 4
    result = analyze(plan, _result(plan, [1.02] * 4))
    assert result["decision_evidence"]["terminal_cproj_shapley_ce"] < 0.05
    assert result["decision_evidence"]["terminal_interaction_ce"] == pytest.approx(
        0.05
    )
    assert result["decision_evidence"]["interaction_gate"] is True
    assert result["decision"]["authorize_cproj_ratio10"] is True


def test_small_effect_rejects_capacity_branch() -> None:
    plan = _plan()
    plan["fixed_curves"]["qk_plus_v_L10"] = [1.03] * 4
    plan["fixed_curves"]["qk_plus_v_plus_cproj_L11"] = [1.04] * 4
    result = analyze(plan, _result(plan, [1.02] * 4))
    assert result["decision"]["authorize_cproj_ratio10"] is False
    assert result["decision"]["selected_branch"] == "no_cproj_capacity_run"


def test_identity_mismatch_and_incomplete_curve_are_rejected() -> None:
    plan = _plan()
    wrong = _result(plan, [1.08] * 4)
    wrong["run"]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="config_sha256"):
        analyze(plan, wrong)

    incomplete = _result(plan, [1.08] * 4)
    incomplete["run"]["fixed_evaluations"].pop()
    with pytest.raises(ValueError, match="fixed evaluation steps differ"):
        analyze(plan, incomplete)


def test_nonclean_result_is_rejected() -> None:
    plan = _plan()
    failed = copy.deepcopy(_result(plan, [1.08] * 4))
    failed["run"]["classification"] = "failed"
    with pytest.raises(ValueError, match="not a clean terminal run"):
        analyze(plan, failed)
