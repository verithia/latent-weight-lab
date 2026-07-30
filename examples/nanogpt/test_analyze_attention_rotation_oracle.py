from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_rotation_oracle import (
    low_rank_right_recovery,
    positive_line_recovery,
    right_orthogonal_direction,
)


def test_positive_line_recovery_rejects_wrong_sign() -> None:
    target = torch.tensor([1.0, 2.0])
    assert positive_line_recovery(target, target) == 1.0
    assert positive_line_recovery(target, -target) == 0.0


def test_rank_two_skew_recovers_rank_two_right_orbit_direction() -> None:
    torch.manual_seed(20260730)
    weight, _ = torch.linalg.qr(
        torch.randn(6, 6, dtype=torch.float64)
    )
    first = torch.randn(6, dtype=torch.float64)
    second = torch.randn(6, dtype=torch.float64)
    skew = torch.outer(first, second) - torch.outer(second, first)
    gradient = weight @ skew
    recovery = low_rank_right_recovery(
        weight,
        gradient,
        [2, 4],
    )
    assert recovery[2] > 0.999999
    assert recovery[4] > 0.999999
    assert positive_line_recovery(
        gradient,
        right_orthogonal_direction(weight, gradient),
    ) > 0.0
