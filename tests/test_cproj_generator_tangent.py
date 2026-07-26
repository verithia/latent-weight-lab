from __future__ import annotations

import torch

from examples.nanogpt.analyze_cproj_generator_tangent import (
    cgls_generator_tangent,
    tangent_adjoint,
    tangent_apply,
)
from latent_weight_lab import BlockFHTLinear


def _tiny_module() -> BlockFHTLinear:
    torch.manual_seed(20260726)
    return BlockFHTLinear(
        4,
        3,
        latent_dim=4,
        layers=2,
        seed=17,
        latent_init_std=0.02,
        weight_scale=0.25,
    )


def test_generator_tangent_adjoint_identity() -> None:
    module = _tiny_module()
    hidden = torch.randn(7, 4)
    delta = torch.randn(4)
    output_probe = torch.randn(7, 3)
    left = torch.sum(tangent_apply(module, hidden, delta) * output_probe)
    right = torch.dot(delta, tangent_adjoint(module, hidden, output_probe))
    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_cgls_recovers_an_exact_generator_tangent_target() -> None:
    module = _tiny_module()
    hidden = torch.randn(16, 4)
    true_delta = torch.randn(4)
    target = tangent_apply(module, hidden, true_delta)
    holdout_hidden = torch.randn(9, 4)
    holdout_target = tangent_apply(module, holdout_hidden, true_delta)
    fitted, rows = cgls_generator_tangent(
        module,
        hidden,
        target,
        iterations=12,
        record_iterations={1, 2, 4, 8, 12},
        relative_tolerance=1e-10,
        holdout_hidden=holdout_hidden,
        holdout_target=holdout_target,
    )
    prediction = tangent_apply(module, hidden, fitted)
    torch.testing.assert_close(prediction, target, rtol=2e-4, atol=2e-5)
    assert rows[-1]["explained_energy"] > 0.999999
    assert rows[-1]["holdout_explained_energy"] > 0.999999
    assert all(
        later["explained_energy"] >= earlier["explained_energy"] - 1e-7
        for earlier, later in zip(rows, rows[1:])
    )
