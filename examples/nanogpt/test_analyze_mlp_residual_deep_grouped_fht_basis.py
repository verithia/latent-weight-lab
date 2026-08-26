from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_deep_grouped_fht_basis import (
    DeepGroupedProductFHT,
    parse_depth_groups,
)
from examples.nanogpt.analyze_mlp_residual_multibranch_fht_basis import (
    coordinate_vjp,
)
from latent_weight_lab import ProductFHTLinear


def test_parse_depth_groups_requires_equal_state() -> None:
    assert parse_depth_groups("5:1,10:2,20:4,40:8") == [
        (5, 1),
        (10, 2),
        (20, 4),
        (40, 8),
    ]
    try:
        parse_depth_groups("5:1,8:2")
    except ValueError:
        pass
    else:
        raise AssertionError("unequal state candidates must be rejected")


def test_equal_depth_group_ratio_has_equal_state() -> None:
    counts = []
    for depth, group in ((2, 1), (4, 2), (8, 4)):
        module = DeepGroupedProductFHT(
            5,
            6,
            depth=depth,
            group_size=group,
            seed=3,
            weight_std=0.02,
        )
        counts.append(module.trainable_scalar_count)
    assert counts == [22, 22, 22]


def test_group_one_matches_production_product_fht() -> None:
    reference = ProductFHTLinear(
        5,
        6,
        factors=3,
        seed=13,
        weight_std=0.02,
        weight_space_muon=False,
        natural_gradient=True,
    )
    candidate = DeepGroupedProductFHT(
        5,
        6,
        depth=3,
        group_size=1,
        seed=13,
        weight_std=0.02,
    )
    with torch.no_grad():
        values = torch.randn_like(reference.product_log_diagonals) * 0.04
        gain = torch.randn_like(reference.product_output_log_gain) * 0.04
        reference.product_log_diagonals.copy_(values)
        reference.product_output_log_gain.copy_(gain)
        candidate.grouped_log_diagonals.copy_(values)
        candidate.shared_output_log_gain.copy_(gain)
    torch.testing.assert_close(candidate.weight(), reference._live_weight())
    direction = torch.randn(candidate.trainable_scalar_count)
    diagonal, output = candidate.split_coordinates(direction)
    torch.testing.assert_close(
        candidate.jvp(direction),
        reference._weight_jvp_from_factors(diagonal, output),
    )


def test_grouped_jvp_matches_finite_difference_and_vjp() -> None:
    torch.manual_seed(8)
    module = DeepGroupedProductFHT(
        5,
        6,
        depth=4,
        group_size=2,
        seed=17,
        weight_std=0.02,
    )
    with torch.no_grad():
        for coordinate in module.coordinate_tensors:
            coordinate.normal_(std=0.03)
    direction = torch.randn(module.trainable_scalar_count)
    analytic = module.jvp(direction)
    epsilon = 1e-3
    grouped_direction, output_direction = module.split_coordinates(direction)
    originals = [coordinate.detach().clone() for coordinate in module.coordinate_tensors]
    with torch.no_grad():
        module.grouped_log_diagonals.add_(grouped_direction, alpha=epsilon)
        module.shared_output_log_gain.add_(output_direction, alpha=epsilon)
        plus = module.weight().clone()
        for coordinate, original in zip(
            module.coordinate_tensors, originals, strict=True
        ):
            coordinate.copy_(original)
        module.grouped_log_diagonals.add_(grouped_direction, alpha=-epsilon)
        module.shared_output_log_gain.add_(output_direction, alpha=-epsilon)
        minus = module.weight().clone()
        for coordinate, original in zip(
            module.coordinate_tensors, originals, strict=True
        ):
            coordinate.copy_(original)
    finite = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite, rtol=4e-3, atol=4e-5)
    target = torch.randn(6, 5)
    torch.testing.assert_close(
        torch.sum(analytic * target),
        torch.dot(direction, coordinate_vjp(module, target)),
        rtol=2e-5,
        atol=2e-6,
    )
