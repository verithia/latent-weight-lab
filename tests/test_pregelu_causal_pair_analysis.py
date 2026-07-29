import math

import pytest
import torch

from examples.nanogpt.analyze_pregelu_causal_pair import (
    paired_stats,
    pregelu_stats,
)


def test_paired_stats_reports_relative_geometry() -> None:
    name = "transformer.h.0.mlp.c_fc.weight"
    parent = {name: torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    candidate = {name: torch.tensor([[1.0, 1.0], [0.0, 1.0]])}
    result = paired_stats(parent, candidate, (".mlp.c_fc.weight",))
    assert result["coordinate_count"] == 4
    assert result["difference_rms"] == pytest.approx(0.5)
    assert result["difference_over_parent_norm"] == pytest.approx(
        1.0 / math.sqrt(2.0)
    )


def test_pregelu_stats_preserves_per_layer_evidence() -> None:
    candidate = {
        (
            "transformer.h.0.mlp."
            "pregelu_block_rotation.coordinates"
        ): torch.tensor([0.0, 3.0, 4.0]),
        (
            "transformer.h.1.mlp."
            "pregelu_block_rotation.coordinates"
        ): torch.tensor([0.0, 0.0, 0.0]),
    }
    result = pregelu_stats(candidate)
    assert result["tensor_count"] == 2
    assert result["coordinate_count"] == 6
    assert result["rms"] == pytest.approx(math.sqrt(25.0 / 6.0))
    assert result["max_abs"] == 4.0
    assert len(result["per_layer"]) == 2
