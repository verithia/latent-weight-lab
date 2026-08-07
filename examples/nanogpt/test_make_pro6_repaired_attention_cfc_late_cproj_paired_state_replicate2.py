from __future__ import annotations

import json

from examples.nanogpt.make_pro6_repaired_attention_cfc_late_cproj_paired_state_replicate2 import (
    NONSCIENTIFIC_CONFIG_KEYS,
    SOURCE_CONFIG,
    changed_config_keys,
    make_config,
    prediction_intervals,
)


def test_prediction_intervals_are_frozen_from_four_prior_runs() -> None:
    intervals = prediction_intervals()
    assert intervals["1188"]["historical_values"] == [
        3.8510, 3.8554, 3.8530, 3.8596
    ]
    assert abs(intervals["1188"]["mean"] - 3.85475) < 1e-12
    assert abs(intervals["1188"]["prediction_half_width"] - 0.0131649076) < 1e-9
    assert intervals["2373"]["lower"] < 3.640025 < intervals["2373"]["upper"]


def test_confirmatory_config_changes_only_non_scientific_keys() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    candidate = make_config(source)
    changed = changed_config_keys(source, candidate)
    assert changed
    assert changed <= NONSCIENTIFIC_CONFIG_KEYS
    assert candidate["optimizer_probe_steps"] == source["optimizer_probe_steps"]
    assert candidate["model_seed"] == source["model_seed"]
    assert candidate["train_data_seed"] == source["train_data_seed"]
