import hashlib
import json

from examples.nanogpt.make_pro6_full_replacement_repaired_v_124m_20tpp import (
    BASE,
    OUTPUT,
    PLAN,
    QK_20TPP,
    RESULT_5TPP,
    V_REPAIR_20TPP,
    build_config,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_horizon_transfer_changes_no_scientific_structure() -> None:
    base = json.loads(BASE.read_text())
    candidate = build_config()
    allowed = {
        "max_iters",
        "lr_decay_iters",
        "warmup_iters",
        "eval_interval",
        "planned_tokens",
        "scheduled_tokens",
        "planned_tpp",
        "scheduled_tpp",
        "candidate_scope",
        "confirmation_slot",
        "confirmation_source",
        "dense_fixed_validation_curve",
        "hpo_stage",
        "ladder_role",
        "mfu_preflight_certificate",
        "monitoring_policy",
        "out_dir",
        "practical_equivalence_policy",
        "recipe_resolution_dependency",
        "recipe_resolution_stage",
        "selection_endpoint",
        "resolved_from_template",
    }
    differing = {key for key in candidate if candidate.get(key) != base.get(key)}
    assert differing <= allowed
    assert candidate["max_iters"] == 9489
    assert candidate["lr_decay_iters"] == 9489
    assert candidate["warmup_iters"] == 94
    assert candidate["eval_interval"] == 2373
    assert candidate["block_fht_attn_v_int8_lattice_error_feedback"] is True
    assert candidate["block_fht_mlp_int8_lattice_error_feedback"] is True
    assert not candidate.get("block_fht_attn_cproj_int8_lattice_error_feedback", False)


def test_plan_pins_terminal_and_curve_gates() -> None:
    plan = json.loads(PLAN.read_text())
    assert sha256(OUTPUT) == plan["candidate"]["config_sha256"]
    evidence = plan["evidence"]
    assert sha256(RESULT_5TPP) == evidence["complete_replacement_5tpp"]["sha256"]
    assert sha256(QK_20TPP) == evidence["qk_only_20tpp"]["sha256"]
    assert sha256(V_REPAIR_20TPP) == evidence["isolated_repaired_v_20tpp"]["sha256"]
    gate = plan["terminal_gate"]
    assert gate["zero_gap_closure_validation_ce"] == 3.1547
    assert gate["practical_acceptance_maximum_validation_ce"] == 3.1647
    assert gate["maximum_delta_to_qk_only_at_every_fixed_evaluation_ce"] == 0.02
    assert gate["maximum_delta_to_isolated_repaired_v_at_every_fixed_evaluation_ce"] == 0.02
    assert plan["authorization"]["larger_model"] is False
