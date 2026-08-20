import hashlib
import json

from examples.nanogpt.make_pro6_full_replacement_cproj_error_feedback_124m_20tpp import (
    BASE,
    FAILED_PARENT,
    OUTPUT,
    PLAN,
    QK_20TPP,
    V_REPAIR_20TPP,
    build_config,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_scientific_change_is_attention_cproj_feedback() -> None:
    base = json.loads(BASE.read_text())
    candidate = build_config()
    metadata_changes = {
        "candidate_scope",
        "confirmation_slot",
        "confirmation_source",
        "hpo_stage",
        "ladder_role",
        "mfu_preflight_certificate",
        "out_dir",
        "practical_equivalence_policy",
        "recipe_resolution_dependency",
        "recipe_resolution_stage",
        "registered_resume_protocol",
        "resolved_from_template",
        "selection_endpoint",
    }
    differing = {key for key in candidate if candidate.get(key) != base.get(key)}
    assert differing == metadata_changes | {
        "block_fht_attn_cproj_int8_lattice_error_feedback"
    }
    assert candidate["block_fht_attn_cproj_int8_lattice"] is True
    assert candidate["block_fht_attn_cproj_int8_lattice_error_feedback"] is True
    assert candidate["block_fht_attn_v_int8_lattice_error_feedback"] is True
    assert candidate["block_fht_mlp_int8_lattice_error_feedback"] is True
    assert candidate["max_iters"] == 9489
    assert candidate["eval_interval"] == 2373


def test_plan_pins_late_horizon_attribution_gate() -> None:
    plan = json.loads(PLAN.read_text())
    assert sha256(OUTPUT) == plan["candidate"]["config_sha256"]
    assert sha256(FAILED_PARENT) == plan["evidence"]["failed_parent"]["sha256"]
    assert sha256(QK_20TPP) == plan["evidence"]["qk_only_20tpp"]["sha256"]
    assert sha256(V_REPAIR_20TPP) == plan["evidence"]["isolated_repaired_v_20tpp"]["sha256"]
    gate = plan["frozen_gate"]
    assert gate["maximum_delta_to_qk_only_at_every_fixed_evaluation_ce"] == 0.02
    assert gate["minimum_improvement_over_parent_at_step_7119_ce"] == 0.008
    assert gate["minimum_improvement_over_parent_at_step_9489_ce"] == 0.008
    assert gate["terminal_practical_maximum_validation_ce"] == 3.1647
    assert plan["candidate"]["additional_persistent_optimizer_state_bytes"] == 14155776
    assert plan["authorization"]["larger_model"] is False
