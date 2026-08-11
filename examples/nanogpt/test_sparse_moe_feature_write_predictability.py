from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_feature_write_predictability import (
    direction_cosine_squared,
    family_prediction,
    optimal_plane,
    optimal_radial,
    partition_indices,
    recovery,
    ridge_predict,
)


def test_partition_is_deterministic_and_disjoint() -> None:
    fit_a, score_a = partition_indices(16, 12, 17)
    fit_b, score_b = partition_indices(16, 12, 17)
    assert torch.equal(fit_a, fit_b)
    assert torch.equal(score_a, score_b)
    assert set(fit_a.tolist()).isdisjoint(score_a.tolist())
    assert sorted(torch.cat((fit_a, score_a)).tolist()) == list(range(16))


def test_optimal_radial_recovers_signed_rays() -> None:
    generator = torch.Generator().manual_seed(3)
    direction = torch.randn(2, 7, 5, generator=generator)
    scale = torch.randn(2, 7, 1, generator=generator)
    target = direction * scale
    predicted = optimal_radial(direction, target)
    assert torch.allclose(predicted, target, atol=2e-5, rtol=2e-5)
    assert recovery(predicted, target) > 0.99999


def test_optimal_plane_recovers_two_stream_span() -> None:
    generator = torch.Generator().manual_seed(5)
    current = torch.randn(2, 9, 6, generator=generator)
    motion = torch.randn(2, 9, 6, generator=generator)
    coefficients = torch.randn(2, 9, 2, generator=generator)
    target = coefficients[..., :1] * current + coefficients[..., 1:] * motion
    predicted = optimal_plane(current, motion, target)
    assert recovery(predicted, target) > 0.99999


def test_ridge_predict_generalizes_known_linear_map() -> None:
    generator = torch.Generator().manual_seed(7)
    fit = torch.randn(3, 40, 8, generator=generator)
    score = torch.randn(3, 12, 8, generator=generator)
    matrix = torch.randn(3, 8, 6, generator=generator)
    target = fit @ matrix
    expected = score @ matrix
    predicted, diagnostics = ridge_predict(fit, target, score, 1e-8)
    assert torch.allclose(predicted, expected, atol=2e-4, rtol=2e-4)
    assert diagnostics["effective_condition_maximum"] >= 1.0


def test_dense_motion_family_recovers_unseen_linear_relation() -> None:
    generator = torch.Generator().manual_seed(11)
    current = torch.randn(2, 48, 6, generator=generator)
    motion = torch.randn(2, 48, 6, generator=generator)
    features = torch.cat(
        (
            current / current.norm(dim=-1, keepdim=True),
            motion / motion.norm(dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    matrix = torch.randn(2, 12, 6, generator=generator)
    target = features @ matrix
    fit, score = partition_indices(48, 36, 13)
    prediction, _ = family_prediction(
        "dense_current_motion_ridge",
        current,
        motion,
        target,
        fit,
        score,
        1e-8,
    )
    scored_target = target.index_select(1, score)
    assert recovery(prediction, scored_target) > 0.999
    assert direction_cosine_squared(prediction, scored_target) > 0.999
