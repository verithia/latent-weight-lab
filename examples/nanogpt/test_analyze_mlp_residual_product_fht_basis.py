from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_product_fht_basis import (
    cg_project,
    coordinate_jvp,
    deterministic_weighted_mixture,
)
from latent_weight_lab import ProductFHTLinear


def test_cg_recovers_a_known_chart_tangent() -> None:
    module = ProductFHTLinear(
        4,
        4,
        factors=2,
        seed=19,
        weight_std=0.02,
        weight_space_muon=False,
    )
    torch.manual_seed(7)
    coordinate = torch.randn(module.trainable_scalar_count)
    target = coordinate_jvp(module, coordinate)
    result = cg_project(
        module,
        target,
        maximum_iterations=64,
        relative_tolerance=1e-7,
        damping_ratio=1e-8,
    )
    assert result["cg_projection_capture"] > 0.999


def test_weighted_mixture_is_deterministic_and_normalized() -> None:
    basis = torch.stack((torch.eye(3), torch.ones(3, 3)))
    probabilities = torch.tensor([0.8, 0.2])
    first = deterministic_weighted_mixture(
        basis, probabilities, update=4, width=3, seed=11
    )
    second = deterministic_weighted_mixture(
        basis, probabilities, update=4, width=3, seed=11
    )
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(), torch.tensor(1.0))
