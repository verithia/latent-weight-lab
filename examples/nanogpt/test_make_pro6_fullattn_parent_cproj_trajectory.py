from __future__ import annotations

import json

from examples.nanogpt.make_pro6_fullattn_parent_cproj_trajectory import (
    CERTIFICATE,
    EXPECTED_SOURCE_CONFIG_SHA256,
    OUTPUT_CONFIG,
    SNAPSHOT_LAYERS,
    SOURCE_CONFIG,
    json_bytes,
    make_config,
    make_plan,
    sha256_bytes,
    sha256_file,
)


def test_matched_parent_config_preserves_scientific_recipe() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    candidate = make_config(source)
    assert sha256_file(SOURCE_CONFIG) == EXPECTED_SOURCE_CONFIG_SHA256
    for key in (
        "method",
        "block_fht_targets",
        "block_fht_latent_ratio",
        "block_fht_layers",
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
    assert candidate["trajectory_snapshot_interval"] == 1
    assert candidate["trajectory_snapshot_targets"] == ["mlp.c_proj"]
    assert candidate["trajectory_snapshot_layers"] == SNAPSHOT_LAYERS
    assert candidate["trajectory_snapshot_dtype"] == "float32"
    assert candidate["mfu_preflight_certificate"] == str(CERTIFICATE)


def test_plan_binds_inputs_and_freezes_parent_specific_decision() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config_sha256 = sha256_bytes(json_bytes(make_config(source)))
    plan = make_plan(config_sha256)
    assert plan["identity"]["candidate_config_sha256"] == config_sha256
    assert plan["performance_gate"]["minimum_mfu_fraction"] == 0.2
    assert plan["performance_gate"]["include_diagnostic_io"] is True
    assert plan["authorization"]["language_model_candidate_training"] is False
    assert plan["endpoint_oracle"]["selection_order"] == [
        "hidden88_output32_full_carry",
        "hidden88_output64_full_carry",
    ]
    assert plan["decision_rule"]["new_training_before_classification"] is False
    assert plan["identity"]["candidate_config"] == str(
        OUTPUT_CONFIG.relative_to(OUTPUT_CONFIG.parents[3])
    )
