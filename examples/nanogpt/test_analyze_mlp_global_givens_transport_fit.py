from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_global_givens_transport_fit import (
    fit_global_givens_transport,
    parse_cells,
)
from examples.nanogpt.model import LearnedGivensOutputMix


def test_fit_recovers_exact_small_givens_flow() -> None:
    torch.manual_seed(7)
    source = torch.randn(32, 8)
    expected = LearnedGivensOutputMix(8, 2, 11)
    with torch.no_grad():
        expected.angles.normal_(std=0.1)
        target = expected(source)
    result = fit_global_givens_transport(
        source,
        target,
        stages=2,
        seed=11,
        steps=500,
        learning_rate=0.03,
    )
    assert result["endpoint_recovery"] > 0.999
    assert result["coordinates"] == 8


def test_parse_cells_requires_unique_pairs() -> None:
    assert parse_cells("0:0,3:60") == [(0, 0), (3, 60)]
    try:
        parse_cells("0:0,0:0")
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate cell was accepted")
