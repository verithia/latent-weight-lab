from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_fht_block_skew_tangent import (
    coordinate_dot,
)
from examples.nanogpt.analyze_attention_persistent_givens_tangent import (
    FixedGivensSide,
    PersistentGivensTangent,
)
from examples.nanogpt.model import LearnedGivensOutputMix


def test_givens_side_matches_finite_difference() -> None:
    torch.manual_seed(19)
    weight = torch.randn(8, 8, dtype=torch.float64)
    mixer = LearnedGivensOutputMix(8, 2, 23).to(dtype=torch.float64)
    for side in ("input", "output"):
        chart = FixedGivensSide(
            weight=weight,
            side=side,
            permutations=mixer.permutations,
        )
        coordinates = torch.randn_like(chart.zeros()) * 0.1
        epsilon = 1e-6
        with torch.no_grad():
            mixer.angles.copy_((epsilon * coordinates).reshape(-1))
            moved = (
                mixer(weight.T).T
                if side == "output"
                else weight - (mixer(weight) - weight)
            )
            finite = (moved - weight) / epsilon
        assert torch.allclose(
            chart.jvp(coordinates), finite, atol=2e-6, rtol=2e-6
        )


def test_persistent_givens_adjoint() -> None:
    torch.manual_seed(29)
    weight = torch.randn(8, 8, dtype=torch.float64)
    input_mix = LearnedGivensOutputMix(8, 2, 31)
    output_mix = LearnedGivensOutputMix(8, 2, 37)
    chart = PersistentGivensTangent(
        weight=weight,
        sides=("input", "output"),
        permutations={
            "input": input_mix.permutations,
            "output": output_mix.permutations,
        },
        stages=2,
    )
    coordinates = tuple(torch.randn_like(value) for value in chart.zeros())
    direction = torch.randn_like(weight)
    left = (chart.jvp(coordinates) * direction).sum()
    right = coordinate_dot(coordinates, chart.adjoint(direction))
    assert torch.allclose(left, right, atol=2e-10, rtol=2e-10)
