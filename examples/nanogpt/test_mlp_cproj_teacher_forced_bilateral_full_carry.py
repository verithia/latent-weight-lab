from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_full_carry import (
    CONTROL,
    aggregate_rows,
    fit_output_pass,
)


def test_output_pass_reduces_left_rotation_residual_and_preserves_column_gram() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(4, 8, generator=generator)
    angle = 0.01
    rotation = torch.eye(4)
    rotation[0, 0] = torch.cos(torch.tensor(angle))
    rotation[0, 1] = -torch.sin(torch.tensor(angle))
    rotation[1, 0] = torch.sin(torch.tensor(angle))
    rotation[1, 1] = torch.cos(torch.tensor(angle))
    target = rotation @ weight - weight
    updated = fit_output_pass(
        weight, target, stages=1, neighbors=2, seed=5
    )
    assert (target - (updated - weight)).square().sum() < target.square().sum()
    assert torch.allclose(
        updated.T @ updated,
        weight.T @ weight,
        atol=2e-5,
        rtol=2e-5,
    )


def _row(
    arm: str,
    layer: int,
    step: int,
    endpoint_error: float,
    gram_error: float,
    feedback: float,
):
    return {
        "arm": arm,
        "layer": layer,
        "score_step": step,
        "chord_energy": 1.0,
        "endpoint_error_energy": endpoint_error,
        "endpoint_recovery": 1.0 - endpoint_error,
        "endpoint_cosine": 1.0,
        "row_gram_chord_energy": 1.0,
        "row_gram_error_energy": gram_error,
        "row_gram_recovery": 1.0 - gram_error,
        "mean_requested_update_recovery": -2.0 if arm == CONTROL else -1.0,
        "terminal_feedback_fro": feedback,
    }


def _rows(output32_error: float, output32_gram: float, output32_feedback: float):
    rows = []
    for step in (60, 120, 180, 238):
        for layer in range(5):
            rows.extend(
                [
                    _row(CONTROL, layer, step, 0.2, 0.2, 10.0),
                    _row(
                        "hidden88_output32_full_carry",
                        layer,
                        step,
                        output32_error,
                        output32_gram,
                        output32_feedback,
                    ),
                    _row(
                        "hidden88_output64_full_carry",
                        layer,
                        step,
                        0.1,
                        0.1,
                        8.0,
                    ),
                ]
            )
    return rows


def test_selects_smallest_passing_output_depth() -> None:
    result = aggregate_rows(_rows(0.16, 0.16, 9.0))
    assert result["comparisons"]["hidden88_output32_full_carry"][
        "passed"
    ] is True
    assert result["selected_arm"] == "hidden88_output32_full_carry"


def test_falls_through_to_output64() -> None:
    result = aggregate_rows(_rows(0.18, 0.18, 9.6))
    assert result["comparisons"]["hidden88_output32_full_carry"][
        "passed"
    ] is False
    assert result["comparisons"]["hidden88_output64_full_carry"][
        "passed"
    ] is True
    assert result["selected_arm"] == "hidden88_output64_full_carry"
