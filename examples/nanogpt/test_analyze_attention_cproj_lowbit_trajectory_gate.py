from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_cproj_lowbit_trajectory_gate import (
    classify_candidate,
    encode_blocks,
    phase_name,
)


def test_binary_codec_is_deterministic_least_squares() -> None:
    values = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])
    codes, scales, decoded = encode_blocks(
        values, codec="binary", block_size=4, ternary_threshold_rms=0.6
    )
    codes2, scales2, decoded2 = encode_blocks(
        values, codec="binary", block_size=4, ternary_threshold_rms=0.6
    )
    assert torch.equal(codes, codes2)
    assert torch.equal(scales, scales2)
    assert torch.equal(decoded, decoded2)
    torch.testing.assert_close(decoded, torch.tensor([[[2.5, -2.5, 2.5, -2.5]]]))


def test_registered_phase_partition() -> None:
    boundaries = [600, 1200, 1800, 2373]
    assert phase_name(60, boundaries) == "phase_0"
    assert phase_name(1200, boundaries) == "phase_1"
    assert phase_name(2373, boundaries) == "phase_3"


def test_classification_requires_state_chord_decode_and_storage() -> None:
    summary = {
        "state": {"fixed_scale_recovery": 0.90},
        "endpoint_state": {"fixed_scale_recovery": 0.90},
        "minimum_layer_state_recovery": 0.80,
        "minimum_snapshot_state_recovery": 0.80,
        "chord": {"fixed_scale_recovery": 0.70, "cosine": 0.85},
        "minimum_phase_chord_recovery": 0.50,
        "minimum_layer_chord_recovery": 0.50,
        "deterministic_decode": True,
        "storage": {"storage_ratio": 0.12},
    }
    thresholds = {
        "state_aggregate_recovery_minimum": 0.55,
        "state_endpoint_recovery_minimum": 0.55,
        "state_minimum_layer_recovery": 0.45,
        "state_minimum_snapshot_recovery": 0.45,
        "chord_aggregate_recovery_minimum": 0.30,
        "chord_aggregate_cosine_minimum": 0.65,
        "chord_minimum_phase_recovery": 0.10,
        "chord_minimum_layer_recovery": 0.10,
        "maximum_storage_ratio": 0.13,
    }
    passed, checks = classify_candidate(summary, thresholds)
    assert passed and all(checks.values())
    summary["chord"]["fixed_scale_recovery"] = 0.20
    passed, checks = classify_candidate(summary, thresholds)
    assert not passed and not checks["chord_aggregate"]
