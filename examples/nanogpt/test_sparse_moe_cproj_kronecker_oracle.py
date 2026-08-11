from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cproj_context_modulated_fht_oracle import (
    ExpertFrame,
)
from examples.nanogpt.analyze_sparse_moe_cproj_kronecker_oracle import (
    KroneckerShape,
    apply_delta,
    fit_functional_factors,
    materialize_kronecker,
    parameter_recovery,
    truncated_kronecker_svd,
)


def test_materialization_matches_torch_kron() -> None:
    generator = torch.Generator().manual_seed(7)
    group = torch.randn(2, 2, 3, 4, generator=generator)
    channel = torch.randn(2, 2, 5, 6, generator=generator)
    actual = materialize_kronecker(group, channel)
    expected = torch.stack(
        [
            sum(
                (torch.kron(group[e, r], channel[e, r]) for r in range(2)),
                torch.zeros(15, 24),
            )
            for e in range(2)
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_truncated_svd_exactly_recovers_rank_two_kronecker_sum() -> None:
    generator = torch.Generator().manual_seed(11)
    shape = KroneckerShape(3, 4, 5, 2)
    group = torch.randn(2, 2, 3, 5, generator=generator)
    channel = torch.randn(2, 2, 4, 2, generator=generator)
    target = materialize_kronecker(group, channel)
    fitted_group, fitted_channel = truncated_kronecker_svd(target, shape, 2)
    reconstructed = materialize_kronecker(fitted_group, fitted_channel)
    assert parameter_recovery(reconstructed, target) > 1.0 - 1e-10


def test_one_kronecker_term_can_have_full_ordinary_matrix_rank() -> None:
    group = torch.eye(3).reshape(1, 1, 3, 3)
    channel = torch.eye(4).reshape(1, 1, 4, 4)
    matrix = materialize_kronecker(group, channel)[0]
    assert int(torch.linalg.matrix_rank(matrix)) == 12


def test_registered_shape_is_271x_at_rank_two() -> None:
    shape = KroneckerShape(24, 32, 48, 32)
    coordinates = shape.coordinates_per_expert(2)
    assert coordinates == 4352
    assert shape.output_width * shape.input_width / coordinates == 271.05882352941177


def test_functional_refinement_reduces_synthetic_error() -> None:
    generator = torch.Generator().manual_seed(19)
    shape = KroneckerShape(2, 2, 2, 2)
    target_group = torch.randn(1, 1, 2, 2, generator=generator)
    target_channel = torch.randn(1, 1, 2, 2, generator=generator)
    target_delta = materialize_kronecker(target_group, target_channel)
    hidden = torch.randn(32, 4, generator=generator)
    frames = [
        ExpertFrame(
            tokens=torch.arange(32),
            probabilities=torch.ones(32),
            hidden=hidden,
        )
    ]
    target = apply_delta(frames, target_delta, 32)
    initial_group = torch.randn(1, 1, 2, 2, generator=generator) * 0.2
    initial_channel = torch.randn(1, 1, 2, 2, generator=generator) * 0.2
    fitted_group, fitted_channel, diagnostics = fit_functional_factors(
        frames,
        target,
        initial_group,
        initial_channel,
        steps=120,
        learning_rate=0.05,
        gradient_clip=10.0,
    )
    fitted = apply_delta(
        frames, materialize_kronecker(fitted_group, fitted_channel), 32
    )
    assert diagnostics["best_relative_error_squared"] < 1e-4
    assert parameter_recovery(fitted, target) > 0.999
