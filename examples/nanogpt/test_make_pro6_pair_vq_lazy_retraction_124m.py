from __future__ import annotations

import json

from examples.nanogpt.make_pro6_pair_vq_lazy_retraction_124m import (
    ADAPTIVE_MOMENTUM_BYTES_MAX,
    MFU_RESULT,
    OUTPUT,
    PLAN,
    build_config,
    sha256,
)


def test_lazy_retraction_endpoint_changes_only_registered_mechanism() -> None:
    config = build_config()
    assert config["scientific_parent_sha256"] == sha256(PLAN)
    assert config["max_iters"] == 238
    assert config["eval_interval"] == 60
    assert config["model_seed"] == 1337
    assert config["train_data_seed"] == 20260714
    assert config["block_fht_targets"] == ["attn.c_attn.qk_headwise"]
    assert config["block_fht_mlp_pair_vq"] is True
    assert config.get(
        "block_fht_mlp_pair_vq_targets", ["mlp.c_fc", "mlp.c_proj"]
    ) == ["mlp.c_fc", "mlp.c_proj"]
    assert config["block_fht_mlp_pair_vq_code_refresh_interval"] == 8
    assert config["block_fht_mlp_pair_vq_forward_visible_feedback"] is True
    assert config["block_fht_mlp_pair_vq_fp16_ambient_momentum"] is True
    assert (
        config["block_fht_mlp_pair_vq_fp16_reserved_escape_granularity"]
        == "adaptive_block"
    )
    assert config["block_fht_mlp_pair_vq_lazy_retraction_interval"] == 8
    assert config["block_fht_mlp_pair_vq_lazy_retraction_forced_steps"] == [
        60,
        120,
        180,
        238,
    ]
    assert "persistent_training_bytes_exact" not in config
    assert config["persistent_momentum_bytes_max"] == ADAPTIVE_MOMENTUM_BYTES_MAX


def test_lazy_retraction_endpoint_is_fail_closed() -> None:
    config = build_config()
    gate = config["endpoint_gate"]
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] >= 0.20
    assert gate["terminal_candidate_validation_ce_max"] == 5.4110
    assert gate["compact_checkpoint_required"] is True
    assert gate["automatic_rerun"] is False
    assert gate["automatic_horizon_transfer"] is False
    assert gate["automatic_scale_up"] is False
    assert gate["automatic_sweep"] is False
    assert OUTPUT.name in config["literal_command"]


def test_lazy_retraction_exact_config_mfu_is_sealed() -> None:
    result = json.loads(MFU_RESULT.read_text())
    assert result["passed"] is True
    assert result["config"]["sha256"] == sha256(OUTPUT)
    assert result["preflight"]["timed_updates"] == 8
    assert result["native_block_fht_extension"]["loaded"] is True
    assert result["measurement"]["mfu_fraction"] >= 0.20
    assert result["stability"]["all_logged_losses_finite"] is True
