from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    direction_metrics,
    right_orthogonal_tangent,
    span_recovery,
)


def test_direction_metrics_exact_positive_direction() -> None:
    target = torch.randn(5, 7)
    result = direction_metrics(target, 3.0 * target)
    assert result["cosine"] > 0.999999
    assert result["positive_step_line_recovery"] > 0.999999
    assert abs(float(result["optimal_signed_scale"]) - 1.0 / 3.0) < 1e-7


def test_right_tangent_preserves_row_gram_to_first_order() -> None:
    generator = torch.Generator().manual_seed(13)
    weight = torch.randn(4, 9, generator=generator)
    direction = torch.randn(4, 9, generator=generator)
    tangent = right_orthogonal_tangent(weight, direction)
    gram_derivative = tangent @ weight.T + weight @ tangent.T
    assert float(gram_derivative.abs().max()) < 2e-5


def test_direction_span_recovers_two_component_target() -> None:
    first = torch.zeros(3, 4)
    second = torch.zeros(3, 4)
    first[0, 0] = 1.0
    second[2, 3] = 1.0
    target = 2.0 * first - 0.5 * second
    assert span_recovery(target, [first, second]) > 0.999999
