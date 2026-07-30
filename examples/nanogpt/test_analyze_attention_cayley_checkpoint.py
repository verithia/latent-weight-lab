from __future__ import annotations

import pytest
import torch

from examples.nanogpt.analyze_attention_cayley_checkpoint import (
    frame_motion_metrics,
    seeded_right_frame,
)


def test_seeded_frame_is_reproducible_and_column_normalized() -> None:
    first = seeded_right_frame(32, 2, 17)
    second = seeded_right_frame(32, 2, 17)
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=0), torch.ones(2))


def test_frame_motion_detects_unchanged_and_rotated_right_subspace() -> None:
    initial = torch.eye(8)[:, :2]
    left = torch.eye(8)[:, 2:4]
    unchanged = frame_motion_metrics(initial, initial, left)
    rotated = frame_motion_metrics(initial, torch.eye(8)[:, 4:6], left)
    assert unchanged["right_principal_cosine_min"] == 1.0
    assert unchanged["right_chordal_distance"] == 0.0
    assert rotated["right_principal_cosine_mean"] == 0.0
    assert rotated["right_chordal_distance"] == pytest.approx(2**0.5)
    assert unchanged["left_energy_outside_initial_right"] == 1.0
