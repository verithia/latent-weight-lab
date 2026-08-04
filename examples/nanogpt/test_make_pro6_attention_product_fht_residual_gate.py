from __future__ import annotations

import json

from examples.nanogpt.make_pro6_attention_product_fht_residual_gate import (
    CERTIFICATE,
    EXPECTED_SOURCE_CONFIG_SHA256,
    OUTPUT_CONFIG,
    PHASE_BOUNDARIES,
    PROBE_LAYERS,
    PROBE_STEPS,
    SOURCE_CONFIG,
    json_bytes,
    make_config,
    make_plan,
    sha256_bytes,
    sha256_file,
)


def test_replay_preserves_scientific_recipe() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    candidate = make_config(source)
    assert sha256_file(SOURCE_CONFIG) == EXPECTED_SOURCE_CONFIG_SHA256
    for key in (
        "method",
        "optimizer",
        "learning_rate",
        "min_lr",
        "weight_decay",
        "max_iters",
        "batch_size",
        "gradient_accumulation_steps",
        "model_seed",
        "train_data_seed",
        "eval_seed",
    ):
        assert candidate[key] == source[key]
    assert candidate["trajectory_snapshot_interval"] == 60
    assert candidate["trajectory_snapshot_targets"] == [
        "attn.c_attn",
        "attn.c_proj",
    ]
    assert candidate["trajectory_snapshot_layers"] == PROBE_LAYERS
    assert candidate["optimizer_probe_steps"] == PROBE_STEPS
    assert candidate["optimizer_probe_layers"] == PROBE_LAYERS
    assert candidate["mfu_preflight_certificate"] == str(CERTIFICATE)


def test_plan_binds_replay_and_freezes_oracle_gate() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config_sha256 = sha256_bytes(json_bytes(make_config(source)))
    plan = make_plan(config_sha256)
    assert plan["identity"]["candidate_config_sha256"] == config_sha256
    assert plan["replay"]["phase_boundaries"] == PHASE_BOUNDARIES
    assert plan["performance_gate"]["minimum_mfu_fraction"] == 0.2
    assert plan["performance_gate"]["include_diagnostic_io"] is True
    assert plan["oracle"]["factors"] == [6, 12]
    assert plan["authorization"]["watchdog"] is False
    assert plan["decision_rule"]["promote_at_most_one"] is True
    assert "aggregate recovery >= 0.10" in plan["decision_rule"]["promote"]
    assert plan["identity"]["candidate_config"] == str(
        OUTPUT_CONFIG.relative_to(OUTPUT_CONFIG.parents[3])
    )
