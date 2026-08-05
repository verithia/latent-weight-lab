import copy
import json

import pytest

from examples.nanogpt.analyze_attention_qkv_only_20tpp import (
    CONFIG,
    DATASET_MANIFEST_SHA256,
    FIXED_EVAL_INDICES_SHA256,
    FIXED_STEPS,
    MFU_RESULT,
    PLAN,
    analyze,
    sha256,
)


def candidate(curve: list[float]) -> dict:
    mfu = json.loads(MFU_RESULT.read_text())
    return {
        "run": {
            "config_sha256": sha256(CONFIG),
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "mfu_certificate_sha256": mfu["passing_preflight"]["certificate_sha256"],
            "provenance_sha256": "1" * 64,
            "status_sha256": "2" * 64,
            "log_sha256": "3" * 64,
            "checkpoint_sha256": "4" * 64,
            "exit_code": 0,
            "classification": "clean",
            "fixed_evaluations": [
                {"step": step, "validation_ce": value}
                for step, value in zip(FIXED_STEPS, curve)
            ],
        }
    }


def fixtures() -> tuple[dict, dict]:
    return json.loads(PLAN.read_text()), json.loads(MFU_RESULT.read_text())


def test_candidate_inside_both_registered_margins_passes() -> None:
    plan, mfu = fixtures()
    result = analyze(plan, mfu, candidate([4.0, 3.6, 3.3, 3.17]))
    assert result["decision"]["confirmed"] is True
    assert result["comparisons"]["terminal_delta_to_dense_ce"] == pytest.approx(0.0153)
    assert result["decision"]["classification"] == "QKV_ONLY_PARTIAL_ATTENTION_CONFIRMED_20TPP"


def test_dense_margin_failure_rejects_even_if_better_than_full_attention() -> None:
    plan, mfu = fixtures()
    result = analyze(plan, mfu, candidate([4.0, 3.6, 3.3, 3.18]))
    assert result["decision_evidence"]["dense_gate"] is False
    assert result["decision"]["confirmed"] is False


def test_full_attention_margin_failure_is_independent() -> None:
    plan, mfu = fixtures()
    modified = copy.deepcopy(plan)
    modified["matched_results"]["dense_124m_20tpp"]["terminal_validation_ce"] = 3.0
    modified["terminal_gate"]["maximum_validation_ce_delta_to_dense"] = 0.30
    result = analyze(modified, mfu, candidate([4.0, 3.6, 3.3, 3.23]))
    assert result["decision_evidence"]["dense_gate"] is True
    assert result["decision_evidence"]["full_attention_gate"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config_sha256", "0" * 64),
        ("dataset_manifest_sha256", "0" * 64),
        ("fixed_eval_indices_sha256", "0" * 64),
        ("mfu_certificate_sha256", "0" * 64),
    ],
)
def test_identity_mismatch_fails_closed(field: str, value: str) -> None:
    plan, mfu = fixtures()
    wrong = candidate([4.0, 3.6, 3.3, 3.17])
    wrong["run"][field] = value
    with pytest.raises(ValueError, match=field):
        analyze(plan, mfu, wrong)


def test_nonclean_incomplete_and_nonfinite_runs_fail_closed() -> None:
    plan, mfu = fixtures()
    failed = candidate([4.0, 3.6, 3.3, 3.17])
    failed["run"]["classification"] = "failed"
    with pytest.raises(ValueError, match="not a clean terminal run"):
        analyze(plan, mfu, failed)

    incomplete = candidate([4.0, 3.6, 3.3, 3.17])
    incomplete["run"]["fixed_evaluations"].pop()
    with pytest.raises(ValueError, match="fixed evaluation steps differ"):
        analyze(plan, mfu, incomplete)

    nonfinite = candidate([4.0, 3.6, 3.3, float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        analyze(plan, mfu, nonfinite)


def test_mfu_result_must_remain_passing_and_config_matched() -> None:
    plan, mfu = fixtures()
    bad = copy.deepcopy(mfu)
    bad["classification"] = "REJECTED"
    with pytest.raises(ValueError, match="did not pass"):
        analyze(plan, bad, candidate([4.0, 3.6, 3.3, 3.17]))

    bad = copy.deepcopy(mfu)
    bad["config"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="immutable config"):
        analyze(plan, bad, candidate([4.0, 3.6, 3.3, 3.17]))
