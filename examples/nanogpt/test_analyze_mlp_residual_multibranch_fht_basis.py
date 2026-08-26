from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_multibranch_fht_basis import (
    MultiBranchProductFHT,
    TARGET_FIT_OFFSETS,
    coordinate_vjp,
    parse_topologies,
)


def test_parse_topologies_requires_equal_total_depth() -> None:
    assert parse_topologies("5;3+2;2+2+1;1+1+1+1+1") == [
        (5,),
        (3, 2),
        (2, 2, 1),
        (1, 1, 1, 1, 1),
    ]
    try:
        parse_topologies("5;2+2")
    except ValueError:
        pass
    else:
        raise AssertionError("unequal depth partitions must be rejected")


def test_equal_total_depth_has_equal_state() -> None:
    counts = []
    for topology in ((3,), (2, 1), (1, 1, 1)):
        module = MultiBranchProductFHT(
            5, 6, branch_depths=topology, seed=7, weight_std=0.02
        )
        counts.append(module.trainable_scalar_count)
    assert counts == [30, 30, 30]


def test_target_fit_offsets_are_independent_of_target_filtering() -> None:
    assert TARGET_FIT_OFFSETS == {"mlp.c_fc": 0, "mlp.c_proj": 1}


def test_exact_jvp_matches_finite_difference() -> None:
    torch.manual_seed(4)
    module = MultiBranchProductFHT(
        5, 6, branch_depths=(2, 1), seed=9, weight_std=0.02
    )
    with torch.no_grad():
        for coordinate in module.coordinate_tensors:
            coordinate.normal_(std=0.03)
    direction = torch.randn(module.trainable_scalar_count)
    analytic = module.jvp(direction)
    epsilon = 1e-3
    originals = [coordinate.detach().clone() for coordinate in module.coordinate_tensors]
    branch_directions, output_direction = module.split_coordinates(direction)
    with torch.no_grad():
        for coordinate, delta in zip(
            module.branch_log_diagonals, branch_directions, strict=True
        ):
            coordinate.add_(delta, alpha=epsilon)
        module.shared_output_log_gain.add_(output_direction, alpha=epsilon)
        plus = module.weight().clone()
        for coordinate, original in zip(
            module.coordinate_tensors, originals, strict=True
        ):
            coordinate.copy_(original)
        for coordinate, delta in zip(
            module.branch_log_diagonals, branch_directions, strict=True
        ):
            coordinate.add_(delta, alpha=-epsilon)
        module.shared_output_log_gain.add_(output_direction, alpha=-epsilon)
        minus = module.weight().clone()
        for coordinate, original in zip(
            module.coordinate_tensors, originals, strict=True
        ):
            coordinate.copy_(original)
    finite = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite, rtol=3e-3, atol=3e-5)


def test_jvp_vjp_adjoint_identity() -> None:
    torch.manual_seed(8)
    module = MultiBranchProductFHT(
        5, 6, branch_depths=(2, 1), seed=11, weight_std=0.02
    )
    direction = torch.randn(module.trainable_scalar_count)
    target = torch.randn(6, 5)
    left = torch.sum(module.jvp(direction) * target)
    right = torch.dot(direction, coordinate_vjp(module, target))
    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
