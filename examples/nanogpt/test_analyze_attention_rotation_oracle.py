from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_rotation_oracle import (
    low_rank_bilateral_span_recovery,
    low_rank_left_recovery,
    low_rank_right_recovery,
    left_orthogonal_direction,
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


def test_rank_two_skew_recovers_rank_two_left_orbit_direction() -> None:
    torch.manual_seed(20260731)
    weight, _ = torch.linalg.qr(
        torch.randn(6, 6, dtype=torch.float64)
    )
    first = torch.randn(6, dtype=torch.float64)
    second = torch.randn(6, dtype=torch.float64)
    skew = torch.outer(first, second) - torch.outer(second, first)
    gradient = skew @ weight
    recovery = low_rank_left_recovery(
        weight,
        gradient,
        [2, 4],
    )
    assert recovery[2] > 0.999999
    assert recovery[4] > 0.999999
    assert positive_line_recovery(
        gradient,
        left_orthogonal_direction(weight, gradient),
    ) > 0.0


def test_bilateral_span_recovers_mixed_left_and_right_orbits() -> None:
    torch.manual_seed(20260801)
    weight = torch.diag(
        torch.tensor([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
    ).double()
    vectors = [torch.randn(6, dtype=torch.float64) for _ in range(4)]
    left_skew = (
        torch.outer(vectors[0], vectors[1])
        - torch.outer(vectors[1], vectors[0])
    )
    right_skew = (
        torch.outer(vectors[2], vectors[3])
        - torch.outer(vectors[3], vectors[2])
    )
    gradient = left_skew @ weight + weight @ right_skew
    recovery = low_rank_bilateral_span_recovery(
        weight,
        gradient,
        [2, 4],
    )
    assert recovery[2] > 0.0
    assert recovery[2] >= max(
        low_rank_left_recovery(weight, gradient, [2])[2],
        low_rank_right_recovery(weight, gradient, [2])[2],
    )
    assert recovery[4] >= recovery[2]
