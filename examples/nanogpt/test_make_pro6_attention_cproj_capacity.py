import json
from pathlib import Path

from examples.nanogpt.make_pro6_attention_cproj_capacity import (
    PLAN,
    PARENT,
    PROJECTION,
    QK,
    ROOT,
    authorize,
    build,
)


def test_cproj_capacity_candidate_changes_one_scientific_field() -> None:
    parent = json.loads(PARENT.read_text())
    candidate = build(parent)
    assert candidate["block_fht_latent_ratios"] == {
        QK: 0.01,
        PROJECTION: 0.10,
    }
    metadata = {
        "candidate_scope",
        "confirmation_slot",
        "hpo_stage",
        "ladder_role",
        "ladder_slot",
        "launch_ready",
        "launch_block_reason",
        "operator_override",
        "out_dir",
        "practical_equivalence_policy",
        "resolved_from_template",
    }
    scientific_changes = {
        key
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key) and key not in metadata
    }
    assert scientific_changes == {"block_fht_latent_ratios"}


def test_cproj_capacity_candidate_is_blocked_pending_factorial() -> None:
    candidate = build(json.loads(Path(PARENT).read_text()))
    assert candidate["launch_ready"] is False
    assert "factorial" in candidate["launch_block_reason"]
    assert "watchdog" in candidate["practical_equivalence_policy"]


def test_cproj_capacity_builder_rejects_a_mutated_parent() -> None:
    parent = json.loads(PARENT.read_text())
    parent["block_fht_latent_ratios"] = {PROJECTION: 0.02}
    try:
        build(parent)
    except ValueError as error:
        assert "must not already override" in str(error)
    else:
        raise AssertionError("mutated parent was accepted")


def _decision(*, authorized: bool = True) -> dict:
    plan = json.loads(PLAN.read_text())
    return {
        "schema_version": "mai_124m_attention_vo_factorial_result_v1",
        "source_plan": str(PLAN.relative_to(ROOT)),
        "identity": plan["identity"],
        "decision": {
            "authorize_cproj_ratio10": authorized,
            "selected_branch": (
                "qk_cproj_only_cprojratio10"
                if authorized
                else "no_cproj_capacity_run"
            ),
        },
    }


def test_authorize_requires_matching_positive_factorial_decision() -> None:
    plan = json.loads(PLAN.read_text())
    candidate = build(json.loads(PARENT.read_text()))
    resolved = authorize(
        candidate,
        _decision(),
        plan,
        decision_path="examples/nanogpt/configs/selection_artifacts/result.json",
        decision_sha256="a" * 64,
    )
    assert resolved["launch_ready"] is True
    assert resolved["launch_block_reason"] is None
    assert resolved["factorial_decision_artifact_sha256"] == "a" * 64
    assert candidate["launch_ready"] is False


def test_authorize_rejects_negative_or_wrong_identity_decision() -> None:
    plan = json.loads(PLAN.read_text())
    candidate = build(json.loads(PARENT.read_text()))
    try:
        authorize(
            candidate,
            _decision(authorized=False),
            plan,
            decision_path="result.json",
            decision_sha256="a" * 64,
        )
    except ValueError as error:
        assert "does not authorize" in str(error)
    else:
        raise AssertionError("negative decision was accepted")

    wrong = _decision()
    wrong["identity"] = dict(wrong["identity"])
    wrong["identity"]["active_parent_config_sha256"] = "0" * 64
    try:
        authorize(
            candidate,
            wrong,
            plan,
            decision_path="result.json",
            decision_sha256="a" * 64,
        )
    except ValueError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("wrong identity was accepted")
