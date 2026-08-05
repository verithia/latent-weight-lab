import copy
import json

import pytest

from examples.nanogpt.analyze_attention_cproj_capacity import (
    CANDIDATE_CONFIG,
    PARENT,
    PLAN,
    _sha256,
    analyze,
)


def _fixtures() -> tuple[dict, dict]:
    plan = json.loads(PLAN.read_text())
    parent = json.loads(PARENT.read_text())
    return plan, parent


def _candidate(plan: dict, curve: list[float]) -> dict:
    identity = plan["identity"]
    return {
        "run": {
            "config_sha256": _sha256(CANDIDATE_CONFIG),
            "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
            "fixed_eval_indices_sha256": identity["fixed_eval_indices_sha256"],
            "exit_code": 0,
            "classification": "clean",
            "fixed_evaluations": [
                {"step": step, "validation_ce": value}
                for step, value in zip([594, 1188, 1782, 2373], curve)
            ],
        }
    }


def test_registered_improvement_promotes_full_attention_transfer() -> None:
    plan, parent = _fixtures()
    result = analyze(plan, parent, _candidate(plan, [4.10, 3.755, 3.60, 3.52]))
    assert result["decision_evidence"]["terminal_improvement_ce"] == pytest.approx(
        0.0318
    )
    assert result["decision_evidence"]["early_worsened_steps"] == []
    assert result["decision"]["promote_cproj_ratio10_to_full_attention"] is True


def test_terminal_shortfall_rejects_transfer() -> None:
    plan, parent = _fixtures()
    result = analyze(plan, parent, _candidate(plan, [4.10, 3.755, 3.60, 3.54]))
    assert result["decision_evidence"]["terminal_gate"] is False
    assert result["decision"]["classification"] == "REJECT_CPROJ_RATIO10_TRANSFER"


def test_two_material_early_regressions_reject_transfer() -> None:
    plan, parent = _fixtures()
    result = analyze(plan, parent, _candidate(plan, [4.13, 3.78, 3.60, 3.52]))
    assert result["decision_evidence"]["early_worsened_steps"] == [594, 1188]
    assert result["decision_evidence"]["early_gate"] is False
    assert result["decision"]["promote_cproj_ratio10_to_full_attention"] is False


def test_one_material_early_regression_is_permitted() -> None:
    plan, parent = _fixtures()
    result = analyze(plan, parent, _candidate(plan, [4.13, 3.76, 3.60, 3.52]))
    assert result["decision_evidence"]["early_worsened_steps"] == [594]
    assert result["decision"]["promote_cproj_ratio10_to_full_attention"] is True


def test_identity_and_terminal_state_are_fail_closed() -> None:
    plan, parent = _fixtures()
    candidate = _candidate(plan, [4.10, 3.755, 3.60, 3.52])
    candidate["run"]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate.run.config_sha256"):
        analyze(plan, parent, candidate)

    failed = _candidate(plan, [4.10, 3.755, 3.60, 3.52])
    failed["run"]["classification"] = "failed"
    with pytest.raises(ValueError, match="not a clean terminal run"):
        analyze(plan, parent, failed)


def test_incomplete_or_nonfinite_curve_is_rejected() -> None:
    plan, parent = _fixtures()
    incomplete = _candidate(plan, [4.10, 3.755, 3.60, 3.52])
    incomplete["run"]["fixed_evaluations"].pop()
    with pytest.raises(ValueError, match="fixed evaluation steps differ"):
        analyze(plan, parent, incomplete)

    nonfinite = _candidate(plan, [4.10, 3.755, 3.60, float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        analyze(plan, parent, nonfinite)


def test_parent_identity_is_also_checked() -> None:
    plan, parent = _fixtures()
    wrong_parent = copy.deepcopy(parent)
    wrong_parent["run"]["dataset_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent.run.dataset_manifest_sha256"):
        analyze(
            plan,
            wrong_parent,
            _candidate(plan, [4.10, 3.755, 3.60, 3.52]),
        )
