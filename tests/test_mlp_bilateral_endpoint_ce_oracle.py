from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_bilateral_endpoint_ce_oracle import (
    capture_chart_state,
    combine_chart_states,
    evaluate_model_ce,
    parse_layers,
    restore_chart_state,
)
from examples.nanogpt.model import GPT, GPTConfig


def charted_tiny_model() -> GPT:
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=1,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=("mlp.c_proj",),
            block_fht_latent_ratio=0.25,
            block_fht_mlp_hidden_block_rotation_stages=1,
            block_fht_mlp_hidden_block_rotation_size=4,
            block_fht_mlp_hidden_block_rotation_basis_size=8,
            block_fht_mlp_hidden_gain=True,
            block_fht_mlp_output_block_rotation_stages=1,
            block_fht_mlp_output_block_rotation_size=4,
            block_fht_mlp_output_block_rotation_basis_size=8,
            block_fht_mlp_residual_output_gain=True,
        )
    )


def test_parse_layers_requires_unique_nonempty_values() -> None:
    assert parse_layers("0,3,6") == [0, 3, 6]


def test_chart_state_can_select_one_fitted_layer() -> None:
    model = charted_tiny_model()
    layers = [0, 1]
    initial = capture_chart_state(model, layers)
    with torch.no_grad():
        model.transformer.h[1].mlp.hidden_log_gain.add_(0.25)
    fitted = capture_chart_state(model, layers)
    selected = combine_chart_states(initial, fitted, [1])

    restore_chart_state(model, layers, selected)
    restored = capture_chart_state(model, layers)

    assert torch.equal(
        restored["layer.0.hidden_gain"],
        initial["layer.0.hidden_gain"],
    )
    assert torch.equal(
        restored["layer.1.hidden_gain"],
        fitted["layer.1.hidden_gain"],
    )


def test_evaluate_model_ce_is_finite_and_flushes_cache() -> None:
    torch.manual_seed(31)
    model = charted_tiny_model()
    batches = [torch.randint(0, 32, (2, 8))]

    loss = evaluate_model_ce(model, batches, "cpu")

    assert torch.isfinite(torch.tensor(loss))
    for block in model.transformer.h:
        assert block.mlp._cached_charted_cproj_weight is None
