from __future__ import annotations

import torch

from examples.nanogpt.analyze_shared_mlp_layer_gradient_conflict import (
    boundary_group_recoveries,
    common_update_energy_fraction,
    contiguous_group_recoveries,
    gradient_cosine,
)


def test_gradient_cosine_aligned_and_opposed() -> None:
    vector = torch.tensor([1.0, 2.0])
    assert abs(gradient_cosine((vector,), (vector,)) - 1.0) < 1e-7
    assert abs(gradient_cosine((vector,), (-vector,)) + 1.0) < 1e-7


def test_common_update_energy_fraction_detects_cancellation() -> None:
    vector = torch.tensor([1.0, -3.0])
    assert abs(common_update_energy_fraction([(vector,), (vector,)]) - 1.0) < 1e-7
    assert common_update_energy_fraction([(vector,), (-vector,)]) < 1e-7


def test_contiguous_partitions_recover_private_layers() -> None:
    gradients = [(torch.tensor([float(index + 1)]),) for index in range(4)]
    assert contiguous_group_recoveries(gradients, 4) == [1.0, 1.0, 1.0, 1.0]
    try:
        contiguous_group_recoveries(gradients, 3)
    except ValueError as error:
        assert "divide" in str(error)
    else:
        raise AssertionError("expected invalid grouping to fail")


def test_boundary_partitions_support_unequal_groups() -> None:
    gradients = [(torch.tensor([float(index + 1)]),) for index in range(6)]
    values = boundary_group_recoveries(gradients, (1, 3, 6))
    assert len(values) == 3
    assert values[0] == 1.0
    try:
        boundary_group_recoveries(gradients, (2, 5))
    except ValueError as error:
        assert "layer count" in str(error)
    else:
        raise AssertionError("expected incomplete boundaries to fail")
