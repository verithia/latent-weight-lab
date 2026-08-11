from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_functional_tt_oracle import (
    CanonicalTT,
    initialization_relative_error,
    left_canonicalize,
)
from examples.nanogpt.analyze_sparse_moe_cfc_global_tt_oracle import (
    materialize_tt,
)


def _cores() -> list[torch.Tensor]:
    torch.manual_seed(3)
    return [
        torch.randn(1, 2, 2),
        torch.randn(2, 3, 3),
        torch.randn(3, 2, 1),
    ]


def test_left_canonicalization_preserves_tensor() -> None:
    cores = _cores()
    canonical = left_canonicalize(cores)
    torch.testing.assert_close(
        materialize_tt(canonical),
        materialize_tt(cores),
        atol=2e-5,
        rtol=2e-5,
    )
    assert initialization_relative_error(cores, canonical) < 1e-10


def test_canonical_module_matches_canonical_initialization() -> None:
    canonical = left_canonicalize(_cores())
    module = CanonicalTT(canonical)
    torch.testing.assert_close(
        materialize_tt(module.canonical_cores()),
        materialize_tt(canonical),
        atol=2e-5,
        rtol=2e-5,
    )


def test_canonical_module_has_finite_nonzero_gradients() -> None:
    module = CanonicalTT(_cores())
    loss = materialize_tt(module.canonical_cores()).square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.norm()) > 0
        for parameter in module.parameters()
    )
