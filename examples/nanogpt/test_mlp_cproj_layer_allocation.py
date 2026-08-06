from __future__ import annotations

import pytest
import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_matched_givens import MuonMatchedGivensLinear


def config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "block_size": 8,
        "vocab_size": 32,
        "n_layer": 4,
        "n_head": 2,
        "n_embd": 8,
        "bias": False,
        "block_fht": True,
        "block_fht_targets": ("mlp.c_proj",),
        "block_fht_mlp_cproj_muon_matched_givens": True,
        "block_fht_mlp_cproj_muon_matched_givens_layers": (2, 3),
        "block_fht_mlp_cproj_muon_matched_givens_stages": 2,
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages": 1,
        "block_fht_mlp_cproj_muon_matched_givens_neighbors": 4,
    }
    values.update(overrides)
    return GPTConfig(**values)


def test_selected_layers_are_procedural_and_others_are_dense() -> None:
    model = GPT(config())
    assert [
        isinstance(block.mlp.c_proj, MuonMatchedGivensLinear)
        for block in model.transformer.h
    ] == [False, False, True, True]
    assert all(
        isinstance(model.transformer.h[layer].mlp.c_proj, torch.nn.Linear)
        for layer in (0, 1)
    )
    dense_std = torch.stack(
        [
            model.transformer.h[layer].mlp.c_proj.weight.std()
            for layer in (0, 1)
        ]
    )
    assert torch.all(dense_std > 0)
    inputs = torch.randint(0, 32, (2, 8))
    _, loss = model(inputs, inputs)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    for layer in (0, 1):
        grad = model.transformer.h[layer].mlp.c_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all()


def test_empty_layer_mask_preserves_historical_all_layer_behavior() -> None:
    model = GPT(
        config(block_fht_mlp_cproj_muon_matched_givens_layers=())
    )
    assert all(
        isinstance(block.mlp.c_proj, MuonMatchedGivensLinear)
        for block in model.transformer.h
    )


@pytest.mark.parametrize("layers", [(1, 1), (-1,), (4,), (True,)])
def test_invalid_layer_masks_fail_closed(layers: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="layer IDs"):
        GPT(config(block_fht_mlp_cproj_muon_matched_givens_layers=layers))


def test_layer_mask_requires_enabled_chart() -> None:
    with pytest.raises(ValueError, match="require the c_proj chart"):
        GPT(
            config(
                block_fht_mlp_cproj_muon_matched_givens=False,
                block_fht_mlp_cproj_muon_matched_givens_layers=(2, 3),
            )
        )
