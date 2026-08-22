from examples.nanogpt.make_pro6_pair_vq_lazy_retraction_124m_5tpp import (
    ADAPTIVE_MOMENTUM_BYTES_MAX,
    DENSE_PARENT,
    OUTPUT,
    PER_STEP_RESULT,
    PLAN,
    REGISTRATION,
    SHORT_RESULT,
    build_config,
    build_registration,
    sha256,
)


def test_5tpp_registration_is_qualified_and_fail_closed() -> None:
    registration = build_registration()
    assert registration["qualification"]["short_passed"] is True
    assert registration["qualification"]["short_result_sha256"] == sha256(
        SHORT_RESULT
    )
    assert registration["causal_comparator"]["per_step_result_sha256"] == sha256(
        PER_STEP_RESULT
    )
    frozen = registration["frozen_test"]
    assert frozen["horizon_updates"] == 2373
    assert frozen["terminal_validation_ce_maximum"] == 3.4958
    assert frozen["fixed_model_compute_equivalent_penalty_maximum"] == 1.10
    assert frozen["minimum_exact_config_mfu_fraction"] == 0.20
    assert registration["decision_policy"]["automatic_sweep"] is False
    assert registration["decision_policy"]["automatic_scale_up"] is False


def test_5tpp_config_changes_only_registered_retraction_cadence() -> None:
    registration = build_registration()
    config = build_config(sha256(REGISTRATION) if REGISTRATION.exists() else "pending")
    assert config["scientific_parent_sha256"] == sha256(DENSE_PARENT)
    assert config["qualification_dependency_sha256"] == sha256(SHORT_RESULT)
    assert config["theory_plan_sha256"] == sha256(PLAN)
    assert config["max_iters"] == 2373
    assert config["eval_interval"] == 594
    assert config["model_seed"] == 1337
    assert config["train_data_seed"] == 20260714
    assert config["block_fht_targets"] == ["attn.c_attn.qk_headwise"]
    assert config["block_fht_mlp_pair_vq"] is True
    assert config["block_fht_mlp_pair_vq_code_refresh_interval"] == 8
    assert config["block_fht_mlp_pair_vq_lazy_retraction_interval"] == 8
    assert config["block_fht_mlp_pair_vq_lazy_retraction_forced_steps"] == [
        594,
        1188,
        1782,
        2373,
    ]
    assert "mfu_preflight_pair_vq_persistent_training_bytes_exact" not in config
    assert config["persistent_momentum_bytes_max"] == ADAPTIVE_MOMENTUM_BYTES_MAX
    assert OUTPUT.name in config["literal_command"]
    gate = config["endpoint_gate"]
    assert gate["terminal_candidate_validation_ce_max"] == 3.4958
    assert gate["fixed_model_compute_equivalent_penalty_max"] == 1.10
    assert gate["automatic_rerun"] is False
    assert gate["automatic_sweep"] is False
    assert gate["automatic_horizon_transfer"] is False
    assert gate["automatic_scale_up"] is False
