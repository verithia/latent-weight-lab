import json
from pathlib import Path

from examples.nanogpt.make_pro6_attention_cproj_capacity import (
    PARENT,
    PROJECTION,
    QK,
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
