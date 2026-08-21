from __future__ import annotations

import torch

from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    classify_endpoint,
    pair_metrics,
    summarize,
)


def test_pair_metrics_exact_and_zero_prediction() -> None:
    target = torch.tensor([[1.0, -2.0]])
    exact = pair_metrics(target, target)
    assert exact["explained_target_energy"] == 1.0
    assert exact["relative_residual_rms"] == 0.0
    zero = pair_metrics(target, torch.zeros_like(target))
    assert abs(zero["explained_target_energy"]) < 1e-7
    assert abs(zero["relative_residual_rms"] - 1.0) < 1e-7


def test_summarize_energy_weights_records() -> None:
    one = pair_metrics(torch.tensor([1.0]), torch.tensor([1.0]))
    two = pair_metrics(torch.tensor([2.0]), torch.tensor([0.0]))
    summary = summarize([one, two])
    assert summary["mean_explained_target_energy"] == 0.5
    assert abs(summary["global_explained_target_energy"] - 0.2) < 1e-7


def test_classification_is_fail_closed() -> None:
    good = {
        "mean_explained_target_energy": 0.95,
        "minimum_explained_target_energy": 0.8,
    }
    bad_output = {
        "mean_explained_target_energy": 0.89,
        "minimum_explained_target_energy": 0.8,
    }
    bad_jvp = {
        "mean_explained_target_energy": 0.79,
        "minimum_explained_target_energy": 0.6,
    }
    kwargs = {
        "output_mean_threshold": 0.9,
        "output_minimum_threshold": 0.75,
        "jvp_mean_threshold": 0.8,
        "jvp_minimum_threshold": 0.5,
    }
    assert classify_endpoint(bad_output, good, **kwargs) == "ENDPOINT_VALUE_MISMATCH"
    assert classify_endpoint(good, bad_jvp, **kwargs) == "INPUT_JACOBIAN_MISMATCH"
    assert classify_endpoint(good, good, **kwargs) == "UPSTREAM_OR_TEMPORAL_COMPOUNDING"
