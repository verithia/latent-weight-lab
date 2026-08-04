from __future__ import annotations

import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_matched_givens import (
    MuonMatchedGivens,
    MuonMatchedGivensLinear,
)


def _fake_matching(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
):
    del direction, neighbors, seed
    return (
        torch.arange(weight.shape[1]).repeat(stages, 1),
        {
            "candidate_edge_fraction": 1.0,
            "minimum_stage_candidate_edge_fraction": 1.0,
            "prepared_seconds": 0.0,
            "native_seconds": 0.0,
            "total_seconds": 0.0,
            "native_output_validated": True,
            "native_library_sha256": "test-library",
            "source_sha256": "test-source",
        },
    )


def test_cached_native_matching_uses_registered_refresh_seed_stride(
    monkeypatch,
) -> None:
    calls: list[int] = []

    def recording_matching(*args, **kwargs):
        calls.append(int(kwargs["seed"]))
        return _fake_matching(*args, **kwargs)

    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens."
        "fast_muon_matched_permutations",
        recording_matching,
    )
    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens."
        "zeropower_via_newtonschulz5",
        lambda matrix, steps: matrix,
    )
    module = MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=1,
        residual_stages=0,
        output_stages=1,
        neighbors=2,
        refresh_interval=3,
        fast_fresh_matching=True,
        matching_seed=23,
        matching_seed_step_stride=100,
        output_matching_seed_offset=10,
        weight_std=0.02,
    )
    optimizer = MuonMatchedGivens(
        [module],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    refreshes: list[bool] = []
    reports: list[bool] = []
    for _step in range(4):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
        row = optimizer.consume_diagnostics()[0]
        refreshes.append(bool(row["refresh"]))
        reports.append(bool(row["report_refresh"]))
    assert calls == [23, 33, 323, 333]
    assert refreshes == [True, False, False, True]
    assert reports == [True, False, False, True]
    assert int(module.refresh_count) == 2
    assert int(module.last_refresh_step) == 3


def test_output_only_attention_chart_is_valid_and_compact(
    monkeypatch,
) -> None:
    calls: list[int] = []

    def recording_matching(*args, **kwargs):
        calls.append(int(kwargs["seed"]))
        return _fake_matching(*args, **kwargs)

    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens."
        "fast_muon_matched_permutations",
        recording_matching,
    )
    monkeypatch.setattr(
        "examples.nanogpt.muon_matched_givens."
        "zeropower_via_newtonschulz5",
        lambda matrix, steps: matrix,
    )
    module = MuonMatchedGivensLinear(
        8,
        4,
        bias=False,
        stages=0,
        residual_stages=0,
        output_stages=1,
        neighbors=2,
        refresh_interval=15,
        fast_fresh_matching=True,
        matching_seed=23,
        matching_seed_step_stride=8192,
        output_matching_seed_offset=128,
        weight_std=0.02,
    )
    optimizer = MuonMatchedGivens(
        [module],
        lr=0.001,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=2,
    )
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    row = optimizer.consume_diagnostics()[0]
    assert calls == [151]
    assert module.coordinate_count == 2
    assert module.selected_permutations.shape == (0, 8)
    assert row["matching"]["selector"] == "disabled_input_side"
    assert row["output_matching"]["selector"] == (
        "fast_fresh_output_pass"
    )


def test_attention_targets_build_the_registered_bilateral_geometry() -> None:
    targets = ("attn.c_attn.qk", "attn.c_attn.v", "attn.c_proj")
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=targets,
            block_fht_attn_muon_matched_givens_targets=targets,
            block_fht_attn_muon_matched_givens_stages=1,
            block_fht_attn_muon_matched_givens_neighbors=2,
            block_fht_attn_muon_matched_givens_refresh_interval=15,
            block_fht_attn_muon_matched_givens_fast_matching=True,
        )
    )
    attention = model.transformer.h[0].attn
    assert isinstance(attention.c_attn_qk, MuonMatchedGivensLinear)
    assert isinstance(attention.c_attn_v, MuonMatchedGivensLinear)
    assert isinstance(attention.c_proj, MuonMatchedGivensLinear)
    assert (attention.c_attn_qk.stages, attention.c_attn_qk.output_stages) == (
        1,
        1,
    )
    assert (attention.c_attn_v.stages, attention.c_attn_v.output_stages) == (
        1,
        1,
    )
    assert (attention.c_proj.stages, attention.c_proj.output_stages) == (0, 1)
    assert attention.c_attn_qk.matching_seed == 161803
    assert attention.c_attn_v.matching_seed == 161803 + 256
    assert attention.c_proj.matching_seed + 128 == 161803 + 512
    assert model.block_fht_stats() == {
        "modules": 3,
        "generated": 256,
        "latent": 24,
    }
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)


def test_attention_error_feedback_has_separate_optimizer_ownership() -> None:
    targets = ("attn.c_attn.qk", "attn.c_attn.v", "attn.c_proj")
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=targets,
            block_fht_attn_muon_matched_givens_targets=targets,
            block_fht_attn_muon_matched_givens_stages=1,
            block_fht_attn_muon_matched_givens_neighbors=2,
            block_fht_attn_muon_matched_givens_error_feedback=True,
            block_fht_attn_muon_matched_givens_error_feedback_decay=0.5,
        )
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    matched = [
        item
        for item in optimizer.optimizers
        if isinstance(item, MuonMatchedGivens)
    ]
    assert len(matched) == 1
    group = matched[0].param_groups[0]
    assert group["error_feedback"] is True
    assert group["error_feedback_decay"] == 0.5
    assert group["attention_error_feedback"] is True
    assert "cproj_error_feedback_decay_schedule" not in group


def test_attention_and_mlp_matched_charts_use_distinct_feedback_groups() -> None:
    attention_targets = (
        "attn.c_attn.qk",
        "attn.c_attn.v",
        "attn.c_proj",
    )
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=attention_targets + ("mlp.c_proj",),
            block_fht_attn_muon_matched_givens_targets=attention_targets,
            block_fht_attn_muon_matched_givens_stages=1,
            block_fht_attn_muon_matched_givens_neighbors=2,
            block_fht_attn_muon_matched_givens_error_feedback=True,
            block_fht_attn_muon_matched_givens_error_feedback_decay=0.5,
            block_fht_mlp_cproj_muon_matched_givens=True,
            block_fht_mlp_cproj_muon_matched_givens_stages=1,
            block_fht_mlp_cproj_muon_matched_givens_neighbors=2,
            block_fht_mlp_cproj_muon_matched_givens_error_feedback=True,
            block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay=1.0,
        )
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    matched = [
        item
        for item in optimizer.optimizers
        if isinstance(item, MuonMatchedGivens)
    ]
    assert len(matched) == 2
    groups = [item.param_groups[0] for item in matched]
    attention = next(group for group in groups if group.get("attention_error_feedback"))
    mlp = next(
        group
        for group in groups
        if group.get("cproj_error_feedback_decay_schedule")
    )
    assert attention["error_feedback_decay"] == 0.5
    assert mlp["error_feedback_decay"] == 1.0
