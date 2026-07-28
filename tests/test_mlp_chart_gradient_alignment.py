from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_mlp_chart_gradient_alignment import (
    alignment_rows,
    chart_config,
    parse_float_list,
    vector_alignment,
)
from examples.nanogpt.model import GPTConfig


def test_vector_alignment_distinguishes_aligned_orthogonal_and_opposed() -> None:
    reference = torch.tensor([1.0, 0.0])

    aligned = vector_alignment(reference, torch.tensor([2.0, 0.0]))
    orthogonal = vector_alignment(reference, torch.tensor([0.0, 3.0]))
    opposed = vector_alignment(reference, torch.tensor([-4.0, 0.0]))

    assert math.isclose(aligned["cosine"], 1.0)
    assert math.isclose(orthogonal["cosine"], 0.0)
    assert math.isclose(opposed["cosine"], -1.0)


def test_alignment_rows_preserve_global_group_and_layer_scopes() -> None:
    left = {
        "layer.0.hidden_rotation": torch.tensor([1.0, 0.0]),
        "layer.0.hidden_gain": torch.tensor([0.0]),
        "layer.0.output_rotation": torch.tensor([0.0, 1.0]),
        "layer.0.output_gain": torch.tensor([1.0]),
        "layer.3.hidden_rotation": torch.tensor([1.0, 0.0]),
        "layer.3.hidden_gain": torch.tensor([0.0]),
        "layer.3.output_rotation": torch.tensor([0.0, 1.0]),
        "layer.3.output_gain": torch.tensor([1.0]),
    }
    right = {key: value.clone() for key, value in left.items()}

    rows = alignment_rows(
        left,
        right,
        comparison="test",
        split="fit",
        initialization=0.0,
    )

    assert len(rows) == 1 + 4 + 2 * 5
    assert rows[0]["scope"] == "global"
    assert math.isclose(float(rows[0]["cosine"]), 1.0, rel_tol=1e-6)
    assert {
        row["group"] for row in rows if row["scope"] == "group"
    } == {
        "hidden_rotation",
        "hidden_gain",
        "output_rotation",
        "output_gain",
    }


def test_parse_float_list() -> None:
    assert parse_float_list("0,0.125") == [0.0, 0.125]


def test_chart_config_uses_only_model_configuration_fields() -> None:
    configured = chart_config(vars(GPTConfig()), 0.125)

    assert configured.block_fht_mlp_hidden_block_rotation_stages == 2
    assert configured.block_fht_mlp_output_block_rotation_stages == 4
    assert configured.block_fht_mlp_residual_output_log_gain_init == 0.125
