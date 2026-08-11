from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from examples.nanogpt.model import (
    GPT,
    GPTConfig,
    HeadwiseLinear,
    MultiOptimizer,
    SparseMoEMLP,
)
from latent_weight_lab import BlockFHTLinear
from examples.nanogpt.muon import Muon, muon_update, muon_update_batched


def tiny_config(**overrides) -> GPTConfig:
    values = dict(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_hidden_multiplier=2,
    )
    values.update(overrides)
    return GPTConfig(**values)


def scalar_reference(module: SparseMoEMLP, x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1, module.n_embd)
    _logits, indices, probabilities = module._route(flat)
    output = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(module.top_k):
            expert = int(indices[token, slot])
            hidden = F.gelu(
                F.linear(flat[token], module.expert_c_fc[expert])
            )
            value = F.linear(hidden, module.expert_c_proj[expert])
            output[token] += probabilities[token, slot] * value
    return output.reshape_as(x)


def optimizer_parameter_ids(optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_dropless_batched_dispatch_matches_scalar_complete_experts() -> None:
    torch.manual_seed(7)
    module = SparseMoEMLP(tiny_config(n_layer=1), layer_id=0)
    x = torch.randn(3, 5, 8)
    expected = scalar_reference(module, x)
    actual = module(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert module.last_assignment_count == x.shape[0] * x.shape[1] * 2
    assert int(module.last_expert_counts.sum()) == module.last_assignment_count
    assert module.expert_c_fc.shape == (4, 16, 8)
    assert module.expert_c_proj.shape == (4, 8, 16)


def test_router_tie_break_and_selected_probabilities_are_deterministic() -> None:
    module = SparseMoEMLP(tiny_config(n_layer=1), layer_id=0)
    with torch.no_grad():
        module.router.weight.zero_()
    logits, indices, probabilities = module._route(torch.zeros(6, 8))
    assert torch.equal(indices, torch.tensor([[0, 1]]).expand(6, 2))
    torch.testing.assert_close(probabilities, torch.full((6, 2), 0.5))
    assert torch.count_nonzero(logits) == 0


def test_batched_muon_matches_independent_matrix_updates() -> None:
    torch.manual_seed(11)
    update = torch.randn(4, 6, 3)
    expected = torch.stack(
        [muon_update(matrix, steps=5) for matrix in update]
    )
    actual = muon_update_batched(update, steps=5)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_muon_owns_expert_pairs_and_adamw_owns_router() -> None:
    model = GPT(tiny_config())
    optimizer = model.configure_optimizers(
        0.1,
        2.4e-3,
        (0.95, 0.925),
        "cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_adamw_lr_scale=1.0,
    )
    assert isinstance(optimizer, MultiOptimizer)
    muon = next(item for item in optimizer.optimizers if isinstance(item, Muon))
    adamw = next(
        item
        for item in optimizer.optimizers
        if isinstance(item, torch.optim.AdamW)
    )
    muon_ids = optimizer_parameter_ids(muon)
    adamw_ids = optimizer_parameter_ids(adamw)
    for block in model.transformer.h:
        assert isinstance(block.mlp, SparseMoEMLP)
        assert id(block.mlp.expert_c_fc) in muon_ids
        assert id(block.mlp.expert_c_proj) in muon_ids
        assert id(block.mlp.router.weight) in adamw_ids
        assert id(block.mlp.router.weight) not in muon_ids


def test_active_and_stored_counts_match_constructed_model() -> None:
    config = tiny_config()
    model = GPT(config)
    stats = model.moe_parameter_stats()
    stored = sum(parameter.numel() for parameter in model.parameters())
    single_expert = 2 * config.moe_expert_hidden_multiplier * config.n_embd**2
    inactive = (
        config.n_layer
        * (config.moe_num_experts - config.moe_top_k)
        * single_expert
    )
    assert stats["stored"] == stored
    assert stats["active"] == stored - inactive
    assert stats["router"] == (
        config.n_layer * config.moe_num_experts * config.n_embd
    )


def test_eval_reports_ce_while_training_adds_router_objectives() -> None:
    torch.manual_seed(13)
    config = tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, idx.shape)
    model.eval()
    logits, eval_loss = model(idx, targets)
    expected_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    torch.testing.assert_close(eval_loss, expected_ce)
    model.train()
    _logits, train_loss = model(idx, targets)
    assert train_loss > expected_ce


def test_state_and_optimizer_roundtrip_preserve_next_step() -> None:
    torch.manual_seed(17)
    config = tiny_config()
    model = GPT(config)
    optimizer = model.configure_optimizers(
        0.0, 1e-3, (0.9, 0.95), "cpu", optimizer="muon"
    )
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, idx.shape)
    _logits, loss = model(idx, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    restored = GPT(config)
    restored.load_state_dict(model.state_dict())
    restored_optimizer = restored.configure_optimizers(
        0.0, 1e-3, (0.9, 0.95), "cpu", optimizer="muon"
    )
    restored_optimizer.load_state_dict(optimizer.state_dict())
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected, expected_loss = model(idx, targets)
        actual, actual_loss = restored(idx, targets)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0.0, rtol=0.0)


def test_legacy_block_fht_helpers_are_noops_for_sparse_moe() -> None:
    model = GPT(tiny_config())
    model.prepare_block_fht_cache(dtype=torch.float32)
    assert model.finalize_product_fht_pullback_probes() == []
    suspended = model.suspend_block_fht_cache()
    model.restore_block_fht_cache(suspended)
    model.flush_block_fht_cache()


def qk_mapped_moe_config(**overrides) -> GPTConfig:
    values = dict(
        block_fht=True,
        block_fht_targets=("attn.c_attn.qk_headwise",),
        block_fht_output_gain_targets=("attn.c_attn.qk_headwise",),
        block_fht_attn_cayley_targets=("attn.c_attn.qk_headwise",),
        block_fht_attn_cayley_output_targets=(
            "attn.c_attn.qk_headwise",
        ),
        block_fht_attn_cayley_bilateral_targets=(
            "attn.c_attn.qk_headwise",
        ),
        block_fht_attn_cayley_ranks={
            "attn.c_attn.qk_headwise": 4,
        },
    )
    values.update(overrides)
    return tiny_config(**values)


def test_attention_only_block_fht_composes_with_dense_complete_experts() -> None:
    torch.manual_seed(23)
    model = GPT(qk_mapped_moe_config())
    for block in model.transformer.h:
        assert isinstance(block.mlp, SparseMoEMLP)
        assert isinstance(block.attn.c_attn_qk_headwise, HeadwiseLinear)
        assert all(
            isinstance(head, BlockFHTLinear)
            for head in block.attn.c_attn_qk_headwise.heads
        )
        assert isinstance(block.attn.c_attn_v, nn.Linear)
        assert isinstance(block.attn.c_proj, nn.Linear)
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, idx.shape)
    logits, loss = model(idx, targets)
    assert torch.isfinite(logits).all()
    assert loss is not None and torch.isfinite(loss)


@pytest.mark.parametrize("target", ["mlp.c_fc", "mlp.c_proj"])
def test_sparse_moe_rejects_every_block_fht_mlp_target(target: str) -> None:
    with pytest.raises(
        ValueError,
        match="permits BlockFHT only for attention targets",
    ):
        GPT(
            qk_mapped_moe_config(
                block_fht_targets=(
                    "attn.c_attn.qk_headwise",
                    target,
                )
            )
        )


def test_mixed_qk_moe_optimizer_ownership_is_disjoint() -> None:
    model = GPT(qk_mapped_moe_config())
    optimizer = model.configure_optimizers(
        0.1,
        2.4e-3,
        (0.9, 0.95),
        "cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_adamw_lr_scale=0.3,
        block_fht_attn_cayley_lr_scale=10.0 / 3.0,
    )
    assert isinstance(optimizer, MultiOptimizer)
    muon_ids = set().union(
        *(
            optimizer_parameter_ids(item)
            for item in optimizer.optimizers
            if isinstance(item, Muon)
        )
    )
    adamw_ids = set().union(
        *(
            optimizer_parameter_ids(item)
            for item in optimizer.optimizers
            if isinstance(item, torch.optim.AdamW)
        )
    )
    for block in model.transformer.h:
        assert id(block.mlp.expert_c_fc) in muon_ids
        assert id(block.mlp.expert_c_proj) in muon_ids
        assert id(block.attn.c_attn_v.weight) in muon_ids
        assert id(block.attn.c_proj.weight) in muon_ids
        assert id(block.mlp.router.weight) in adamw_ids
        for head in block.attn.c_attn_qk_headwise.heads:
            assert id(head.generator.latent) in adamw_ids
            assert id(head.generator.latent) not in muon_ids
