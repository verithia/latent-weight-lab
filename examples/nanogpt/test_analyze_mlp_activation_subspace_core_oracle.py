from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_activation_subspace_core_oracle import (
    fixed_subspace_core_metrics,
)


def test_fixed_subspace_core_recovers_exact_orthogonal_motion() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(12, 8, generator=generator)
    basis = torch.linalg.qr(
        torch.randn(8, 3, generator=generator)
    ).Q
    core = torch.linalg.qr(
        torch.randn(3, 3, generator=generator)
    ).Q
    target = source + (source @ basis @ core - source @ basis) @ basis.T
    result = fixed_subspace_core_metrics(source, target, basis)
    assert result["orthogonal_core_recovery"] > 0.999999
    assert result["projection_upper_recovery"] > 0.999999


def test_fixed_subspace_core_rejects_orthogonal_complement_motion() -> None:
    source = torch.eye(4)
    basis = torch.eye(4)[:, :2]
    target = source.clone()
    target[:, 2:] = target[:, 2:] @ torch.tensor(
        [[0.0, -1.0], [1.0, 0.0]]
    )
    result = fixed_subspace_core_metrics(source, target, basis)
    assert abs(float(result["orthogonal_core_recovery"])) < 1e-12
    assert abs(float(result["projection_upper_recovery"])) < 1e-12
