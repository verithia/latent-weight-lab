from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_replay import (
    ARMS,
    aggregate_rows,
    cell_metrics,
    cosine_lr,
)


def test_cosine_lr_matches_registered_warmup_and_endpoints() -> None:
    kwargs = dict(
        learning_rate=2.4e-3,
        min_lr=2.4e-4,
        warmup_iters=10,
        decay_iters=238,
    )
    assert math.isclose(cosine_lr(0, **kwargs), 2.4e-3 / 11)
    assert math.isclose(cosine_lr(10, **kwargs), 2.4e-3)
    assert math.isclose(cosine_lr(238, **kwargs), 2.4e-4)


def test_row_gram_metric_distinguishes_left_from_right_transport() -> None:
    start = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    angle = torch.tensor(0.3)
    rotation = torch.stack(
        (
            torch.stack((angle.cos(), -angle.sin())),
            torch.stack((angle.sin(), angle.cos())),
        )
    )
    dense_end = rotation @ start
    right_only = start @ rotation
    exact = cell_metrics(start, dense_end, dense_end, [1.0], torch.zeros_like(start))
    missed = cell_metrics(start, dense_end, right_only, [1.0], torch.zeros_like(start))
    assert exact["row_gram_recovery"] > 0.999999
    assert missed["row_gram_recovery"] < exact["row_gram_recovery"]


def synthetic_rows(output32: float, output64: float) -> list[dict]:
    recovery = {
        "hidden88_decay0p5": 0.4,
        "hidden88_decay1p0": 0.4,
        "hidden96_decay0p5": 0.5,
        "hidden104_decay0p5": 0.5,
        "hidden88_output32_decay0p5": output32,
        "hidden88_output64_decay0p5": output64,
    }
    gram_error = {
        name: (0.7 if "output" in name else 1.0) for name in recovery
    }
    rows = []
    for layer in range(5):
        for phase in range(4):
            for arm in ARMS:
                rows.append(
                    {
                        "arm": arm.name,
                        "layer": layer,
                        "phase_start": phase,
                        "phase_end": phase + 1,
                        "chord_energy": 1.0,
                        "endpoint_error_energy": 1.0 - recovery[arm.name],
                        "endpoint_recovery": recovery[arm.name],
                        "endpoint_cosine": 1.0,
                        "row_gram_chord_energy": 1.0,
                        "row_gram_error_energy": gram_error[arm.name],
                        "row_gram_recovery": 1.0 - gram_error[arm.name],
                        "mean_requested_update_recovery": 0.5,
                        "terminal_feedback_fro": 0.1,
                    }
                )
    return rows


def test_smallest_passing_output_arm_is_selected() -> None:
    result = aggregate_rows(synthetic_rows(0.56, 0.60))
    assert result["decision"] == "SELECT_OUTPUT32_FOR_PRODUCTION_PREFLIGHT"


def test_subthreshold_output_arms_are_rejected() -> None:
    result = aggregate_rows(synthetic_rows(0.54, 0.54))
    assert result["decision"] == "REJECT_ADDITIVE_SPARSE_OUTPUT_TRANSPORT"
