from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_reciprocal_gradient_factors import (
    factor_pair,
    paired_parameter_names,
)


FC = "transformer.h.6.mlp.c_fc.weight"
PROJ = "transformer.h.6.mlp.c_proj.weight"


def inventory() -> dict[str, dict[str, list[torch.Tensor]]]:
    generator = torch.Generator().manual_seed(17)
    fc = [torch.randn((12, 4), generator=generator) for _ in range(3)]
    proj = [value.T.contiguous() for value in fc]
    zeros_fc = [torch.zeros_like(value) for value in fc]
    zeros_proj = [torch.zeros_like(value) for value in proj]
    return {
        FC: {"gradient": fc, "weight": zeros_fc},
        PROJ: {"gradient": proj, "weight": zeros_proj},
    }


def test_pair_names_require_transpose_shapes() -> None:
    values = inventory()
    assert paired_parameter_names(values) == (FC, PROJ)
    values[PROJ]["gradient"][0] = torch.zeros((5, 12))
    try:
        paired_parameter_names(values)
    except ValueError as error:
        assert "transposes" in str(error)
    else:
        raise AssertionError("shape mismatch must fail")


def test_factor_pair_exposes_reciprocal_coordinate_spaces() -> None:
    fields = factor_pair(inventory(), factor_rank=3, device="cpu")
    assert set(fields) == {"hidden", "residual"}
    assert fields["hidden"]["first"][0].shape == (12, 3)
    assert fields["hidden"]["second"][0].shape == (12, 3)
    assert fields["residual"]["first"][0].shape == (4, 3)
    assert fields["residual"]["second"][0].shape == (4, 3)
    # A transposed paired gradient has identical reciprocal singular frames.
    for space in fields.values():
        for first, second in zip(space["first"], space["second"], strict=True):
            projection = first.T @ second
            torch.testing.assert_close(
                projection.T @ projection,
                torch.eye(3),
                atol=1e-5,
                rtol=1e-5,
            )
