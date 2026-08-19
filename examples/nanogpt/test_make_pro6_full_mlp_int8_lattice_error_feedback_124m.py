from examples.nanogpt.make_pro6_full_mlp_int8_lattice_error_feedback_124m import (
    build_config,
    build_plan,
)


def test_candidate_changes_only_the_registered_temporal_codec() -> None:
    config = build_config()
    assert config["block_fht_mlp_int8_lattice_targets"] == [
        "mlp.c_fc",
        "mlp.c_proj",
    ]
    assert config["block_fht_mlp_int8_lattice_error_feedback"] is True
    assert config["block_fht_attn_cproj_int8_lattice"] is True
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
    ]
    assert config["max_iters"] == config["lr_decay_iters"] == 238
    assert config["planned_tpp"] == 0.5
    assert config["mfu_min_fraction"] == 0.2
    accounting = config["mlp_int8_lattice_representation"]
    assert accounting["persistent_codec_bytes"] == 56_650_752
    assert accounting["fp16_optimizer_error_feedback_bytes"] == 113_246_208
    assert accounting["storage_reduction"] > 3.99


def test_plan_blocks_training_until_exact_config_mfu_passes() -> None:
    plan = build_plan("0" * 64)
    assert plan["status"] == "registered_before_exact_config_mfu_and_training"
    assert plan["authorization"]["exact_config_mfu_passed"] is False
    assert plan["authorization"]["one_124m_0p5tpp_training"] is False
    assert plan["terminal_gate"]["maximum_terminal_validation_ce"] == 5.3117
    assert plan["update_rule"]["attention_cproj_feedback"] is False
    assert plan["update_rule"]["extra_training_state_bytes"] == 113_246_208
