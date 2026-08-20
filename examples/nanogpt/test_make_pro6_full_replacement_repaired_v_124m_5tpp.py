import hashlib
import json

from examples.nanogpt.make_pro6_full_replacement_repaired_v_124m_5tpp import (
    BASE,
    DENSE_REPLAY,
    FULL_REPLACEMENT_PARENT,
    OUTPUT,
    PLAN,
    V_REPAIR_5TPP,
    V_REPAIR_20TPP,
    build_config,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_v_representation_changes_from_passed_combined_parent() -> None:
    parent = json.loads(BASE.read_text())
    candidate = build_config()
    assert candidate["block_fht_targets"] == ["attn.c_attn.qk_headwise"]
    assert candidate["block_fht_attn_cayley_targets"] == [
        "attn.c_attn.qk_headwise"
    ]
    assert candidate["block_fht_attn_cayley_ranks"] == {
        "attn.c_attn.qk_headwise": 64
    }
    assert candidate["block_fht_output_gain_targets"] == [
        "attn.c_attn.qk_headwise"
    ]
    assert candidate["block_fht_attn_v_int8_lattice"] is True
    assert candidate["block_fht_attn_v_int8_lattice_block_size"] == 4096
    assert candidate["block_fht_attn_v_int8_lattice_seed"] == 161804
    assert candidate["block_fht_attn_v_int8_lattice_error_feedback"] is True

    for key in (
        "block_fht_attn_cproj_int8_lattice",
        "block_fht_attn_cproj_int8_lattice_block_size",
        "block_fht_attn_cproj_int8_lattice_seed",
        "block_fht_mlp_int8_lattice_targets",
        "block_fht_mlp_int8_lattice_block_size",
        "block_fht_mlp_int8_lattice_seed",
        "block_fht_mlp_int8_lattice_error_feedback",
        "learning_rate",
        "min_lr",
        "max_iters",
        "lr_decay_iters",
        "warmup_iters",
        "eval_interval",
        "optimizer",
        "muon_momentum",
        "muon_ns_steps",
        "weight_decay",
        "batch_size",
        "gradient_accumulation_steps",
        "model_seed",
        "train_data_seed",
        "data_manifest_sha256",
        "fixed_eval_index_spec_sha256",
    ):
        assert candidate[key] == parent[key]
    assert not candidate.get(
        "block_fht_attn_cproj_int8_lattice_error_feedback", False
    )
    assert candidate["mfu_preflight_required"] is True
    assert candidate["mfu_min_fraction"] >= 0.20


def test_plan_pins_evidence_and_freezes_dual_closure_gate() -> None:
    plan = json.loads(PLAN.read_text())
    assert sha256(OUTPUT) == plan["candidate"]["config_sha256"]
    for path, key in (
        (FULL_REPLACEMENT_PARENT, "full_replacement_cayley_v_parent"),
        (V_REPAIR_5TPP, "v_repair_5tpp"),
        (V_REPAIR_20TPP, "v_repair_20tpp"),
        (DENSE_REPLAY, "ordinary_dense_replay"),
    ):
        assert sha256(path) == plan["evidence"][key]["sha256"]
    gate = plan["terminal_gate"]
    assert gate["zero_gap_closure_validation_ce"] == 3.5402
    assert gate["practical_acceptance_maximum_validation_ce"] == 3.5502
    assert gate["maximum_delta_to_ordinary_dense_ce"] == 0.01
    assert (
        gate[
            "maximum_delta_to_cayley_v_combined_parent_at_every_fixed_evaluation_ce"
        ]
        == 0.02
    )
    assert plan["authorization"]["automatic_20tpp"] is False
    assert plan["authorization"]["larger_model"] is False
