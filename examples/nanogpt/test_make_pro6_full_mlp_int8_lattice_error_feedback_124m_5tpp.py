from examples.nanogpt.make_pro6_full_mlp_int8_lattice_error_feedback_124m_5tpp import (
    OUTPUT,
    build_config,
    build_plan,
)

import json


def test_generated_config_is_same_recipe_five_tpp_transfer() -> None:
    config = build_config()
    assert json.loads(OUTPUT.read_text()) == config
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
    ]
    assert config["block_fht_attn_cproj_int8_lattice"] is True
    assert config["block_fht_mlp_int8_lattice_targets"] == [
        "mlp.c_fc",
        "mlp.c_proj",
    ]
    assert config["block_fht_mlp_int8_lattice_error_feedback"] is True
    assert config["max_iters"] == config["lr_decay_iters"] == 2373
    assert config["warmup_iters"] == 23
    assert config["eval_interval"] == 594
    assert config["planned_tpp"] == 5.0
    assert config["scheduled_tokens"] == 622_067_712
    assert config["learning_rate"] == 0.0024
    assert config["min_lr"] == 0.00024
    assert config["mfu_min_fraction"] == 0.2
    assert config["checkpoint_wall_clock_seconds"] == 7200
    assert config["out_dir"].endswith("errorfeedback_5tpp/scientific")


def test_plan_requires_new_mfu_gate_and_freezes_near_dense_endpoint() -> None:
    plan = build_plan("0" * 64)
    assert plan["status"] == "registered_before_exact_config_mfu_and_training"
    assert plan["authorization"]["exact_config_mfu_passed"] is False
    assert plan["authorization"]["one_124m_5tpp_training_after_mfu_pass"] is False
    assert plan["terminal_gate"]["maximum_terminal_validation_ce"] == 3.5602
    assert plan["terminal_gate"]["maximum_delta_to_full_attention_parent_ce"] == 0.0103
    assert plan["authorization"]["larger_model"] is False
