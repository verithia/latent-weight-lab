from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_mlp_cproj_teacher_forced_global_plane_carry import (
    CONTROL,
    _exact_plane_rotation,
    aggregate_rows,
    fit_global_planes,
)


def test_exact_plane_rotation_preserves_row_gram_and_reduces_residual() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(5, 8, generator=generator)
    target_update = torch.randn(5, 8, generator=generator) * 0.03
    u = torch.randn(8, generator=generator)
    u = u / u.norm()
    v = torch.randn(8, generator=generator)
    v = v - torch.dot(u, v) * u
    v = v / v.norm()
    updated, angle, recovery = _exact_plane_rotation(
        weight, target_update, u, v
    )
    assert math.isfinite(angle)
    assert recovery >= -1e-6
    assert torch.allclose(
        updated @ updated.T,
        weight @ weight.T,
        atol=2e-5,
        rtol=2e-5,
    )
    assert (target_update - (updated - weight)).square().sum() <= (
        target_update.square().sum() * (1.0 + 1e-6)
    )


def test_task_conditioned_plane_recovers_a_pure_right_rotation() -> None:
    weight = torch.eye(8)
    angle = torch.tensor(0.2)
    rotation = torch.eye(8)
    rotation[:2, :2] = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ]
    )
    target_update = weight @ rotation - weight
    updated, recovery, angles = fit_global_planes(
        weight,
        target_update,
        planes=1,
        power_iterations=2,
        seed=11,
    )
    assert len(angles) == 1
    assert recovery > 0.99999
    assert torch.allclose(updated, weight + target_update, atol=1e-5, rtol=1e-5)


def _row(arm: str, layer: int, step: int, error: float, feedback: float):
    return {
        "arm": arm,
        "layer": layer,
        "score_step": step,
        "chord_energy": 1.0,
        "endpoint_error_energy": error,
        "endpoint_recovery": 1.0 - error,
        "endpoint_cosine": 1.0,
        "row_gram_chord_energy": 1.0,
        "row_gram_error_energy": 0.5,
        "row_gram_recovery": 0.5,
        "mean_requested_update_recovery": (
            -2.0 if arm == CONTROL else -1.0
        ),
        "mean_global_plane_update_recovery": 0.0 if arm == CONTROL else 0.1,
        "terminal_feedback_fro": feedback,
    }


def _rows(global4_error: float, global4_feedback: float):
    rows = []
    for step in (60, 120, 180, 238):
        for layer in range(5):
            rows.extend(
                [
                    _row(CONTROL, layer, step, 0.2, 10.0),
                    _row(
                        "hidden88_plus_global4_full_carry",
                        layer,
                        step,
                        global4_error,
                        global4_feedback,
                    ),
                    _row(
                        "hidden88_plus_global8_full_carry",
                        layer,
                        step,
                        0.1,
                        8.0,
                    ),
                ]
            )
    return rows


def test_selects_smallest_passing_global_plane_count() -> None:
    result = aggregate_rows(_rows(0.16, 9.0))
    assert result["comparisons"][
        "hidden88_plus_global4_full_carry"
    ]["passed"] is True
    assert result["selected_arm"] == "hidden88_plus_global4_full_carry"


def test_falls_through_to_global8() -> None:
    result = aggregate_rows(_rows(0.18, 9.6))
    assert result["comparisons"][
        "hidden88_plus_global4_full_carry"
    ]["passed"] is False
    assert result["comparisons"][
        "hidden88_plus_global8_full_carry"
    ]["passed"] is True
    assert result["selected_arm"] == "hidden88_plus_global8_full_carry"
