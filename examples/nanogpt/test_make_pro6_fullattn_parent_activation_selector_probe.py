from __future__ import annotations

import json

from examples.nanogpt.make_pro6_fullattn_parent_activation_selector_probe import (
    CERTIFICATE,
    EXPECTED_SOURCE_CONFIG_SHA256,
    MATCHED_PARENT_ENDPOINT_RESULT_SHA256,
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


def test_probe_config_preserves_matched_parent_recipe() -> None:
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
    assert candidate["trajectory_snapshot_interval"] == 60
    assert candidate["trajectory_snapshot_all_parameters"] is True
    assert candidate["trajectory_snapshot_targets"] == []
    assert candidate["trajectory_snapshot_layers"] is None
    assert candidate["optimizer_probe_steps"] == PROBE_STEPS
    assert candidate["optimizer_probe_layers"] == PROBE_LAYERS
    assert candidate["optimizer_probe_targets"] == ["mlp.c_proj"]
    assert candidate["optimizer_probe_dtype"] == "float32"
    assert candidate["mfu_preflight_certificate"] == str(CERTIFICATE)


def test_plan_freezes_equal_budget_holdout_decision_before_analysis() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config_sha256 = sha256_bytes(json_bytes(make_config(source)))
    plan = make_plan(config_sha256)
    assert plan["identity"]["candidate_config_sha256"] == config_sha256
    assert (
        plan["identity"]["matched_parent_endpoint_result_sha256"]
        == MATCHED_PARENT_ENDPOINT_RESULT_SHA256
    )
    assert plan["acquisition"]["phase_boundaries"] == PHASE_BOUNDARIES
    assert plan["performance_gate"]["include_diagnostic_io"] is True
    assert plan["selector_analysis"]["parameter_updates"] == 0
    assert (
        plan["selector_analysis"]["fit_window"]["seed"]
        != plan["selector_analysis"]["holdout_window"]["seed"]
    )
    assert (
        plan["selector_analysis"]["shared_chart"]["output_stages"] == 32
    )
    assert (
        plan["selector_analysis"]["shared_chart"]["coordinate_count_per_layer"]
        == 147456
    )
    assert plan["decision_rule"]["language_model_training_authorized_by_this_plan"] is False
