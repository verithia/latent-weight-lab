from __future__ import annotations

import torch

from examples.nanogpt.analyze_pair_vq_side_localization import (
    _dominance,
    decode_pair_vq_side,
)


def test_decode_pair_vq_side_restores_shapes() -> None:
    codebooks = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]],
        ],
        dtype=torch.bfloat16,
    )
    codes = torch.tensor([[0, 2, 1], [2, 0, 1]], dtype=torch.uint8)
    decoded = decode_pair_vq_side(
        codebooks=codebooks,
        codes=codes,
        shapes=[torch.Size((2, 3)), torch.Size((2, 3))],
        device="cpu",
    )
    expected = torch.tensor(
        [
            [[1.0, 2.0, 5.0], [6.0, 3.0, 4.0]],
            [[-5.0, -6.0, -1.0], [-2.0, -3.0, -4.0]],
        ]
    )
    torch.testing.assert_close(decoded, expected)


def test_dominance_requires_ratio_and_full_gap_fraction() -> None:
    assert _dominance(c_fc_gap=0.01, c_proj_gap=0.07, both_gap=0.08)[
        "dominant_side"
    ] == "c_proj"
    assert _dominance(c_fc_gap=0.04, c_proj_gap=0.04, both_gap=0.08)[
        "dominant_side"
    ] == "neither"
    assert _dominance(c_fc_gap=0.03, c_proj_gap=0.01, both_gap=0.08)[
        "dominant_side"
    ] == "neither"
