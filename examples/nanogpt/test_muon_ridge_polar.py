from __future__ import annotations

import math

import pytest
import torch

from examples.nanogpt.model import GPT, GPTConfig, MultiOptimizer
from examples.nanogpt.muon import Muon, muon_update
from examples.nanogpt.muon_pair_vq import MuonPairVQ


def test_zero_ridge_preserves_native_muon_path_exactly() -> None:
    request = torch.randn(7, 19, generator=torch.Generator().manual_seed(7))
    native = muon_update(request, steps=5)
    explicit_zero = muon_update(request, steps=5, polar_ridge=0.0)
    assert torch.equal(explicit_zero, native)


@pytest.mark.parametrize("shape", [(8, 32), (32, 8)])
def test_ridge_polar_restores_the_original_muon_frobenius_scale(
    shape: tuple[int, int],
) -> None:
    request = torch.randn(*shape, generator=torch.Generator().manual_seed(11))
    update = muon_update(request, steps=5, polar_ridge=0.25)
    columns = request.numel() / request.shape[0]
    rectangular_scale = max(1.0, request.shape[0] / columns) ** 0.5
    expected = math.sqrt(float(min(shape))) * rectangular_scale
    assert float(update.float().norm()) == pytest.approx(expected, rel=1e-5)


def test_ridge_polar_rejects_invalid_values() -> None:
    request = torch.randn(4, 8)
    for value in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            muon_update(request, steps=5, polar_ridge=value)


def test_dense_mlp_matrices_are_the_only_dense_ridge_group() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=1,
            n_embd=8,
        )
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_mlp_polar_ridge=0.25,
    )
    assert isinstance(optimizer, MultiOptimizer)
    ridge_optimizers = [
        item
        for item in optimizer.optimizers
        if isinstance(item, Muon)
        and float(item.param_groups[0]["polar_ridge"]) > 0.0
    ]
    assert len(ridge_optimizers) == 1
    actual = {
        id(parameter)
        for group in ridge_optimizers[0].param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.endswith(".mlp.c_fc.weight")
        or name.endswith(".mlp.c_proj.weight")
    }
    assert actual == expected


def test_pair_vq_optimizer_receives_the_same_mlp_ridge() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=1,
            n_embd=8,
            block_fht=True,
            block_fht_targets=(),
            block_fht_mlp_pair_vq=True,
            block_fht_mlp_pair_vq_neighbor_candidates=16,
            block_fht_mlp_pair_vq_code_refresh_interval=8,
        )
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_mlp_polar_ridge=0.25,
    )
    pair_optimizers = [
        item for item in optimizer.optimizers if isinstance(item, MuonPairVQ)
    ]
    assert len(pair_optimizers) == 1
    assert pair_optimizers[0].param_groups[0]["polar_ridge"] == 0.25
