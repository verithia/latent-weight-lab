from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_quadratic_image_oracle import (
    quadratic_component_cotangents,
    quadratic_jvp,
    quadratic_map,
)


def test_quadratic_jvp_matches_autograd() -> None:
    torch.manual_seed(3)
    z = torch.randn(5, dtype=torch.float64, requires_grad=True)
    dz = torch.randn_like(z)
    a = torch.randn(17, 5, dtype=torch.float64)
    b = torch.randn(17, 5, dtype=torch.float64)
    c = torch.randn(17, 5, dtype=torch.float64)
    fn = lambda value: quadratic_map(a @ value, b @ value, c @ value, 0.7, 0.2, 0.9)
    expected = torch.autograd.functional.jvp(fn, z, dz)[1]
    actual = quadratic_jvp(b @ z, c @ z, a @ dz, b @ dz, c @ dz, 0.7, 0.2, 0.9)
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_quadratic_component_cotangents_are_adjoint() -> None:
    torch.manual_seed(4)
    first = torch.randn(19, dtype=torch.float64)
    second = torch.randn(19, dtype=torch.float64)
    da = torch.randn(19, dtype=torch.float64)
    db = torch.randn(19, dtype=torch.float64)
    dc = torch.randn(19, dtype=torch.float64)
    cotangent = torch.randn(19, dtype=torch.float64)
    tangent = quadratic_jvp(first, second, da, db, dc, 0.8, 0.3, 0.75)
    wa, wb, wc = quadratic_component_cotangents(cotangent, first, second, 0.8, 0.3, 0.75)
    left = (cotangent * tangent).sum()
    right = (wa * da).sum() + (wb * db).sum() + (wc * dc).sum()
    assert torch.allclose(left, right, atol=1e-11, rtol=1e-11)
