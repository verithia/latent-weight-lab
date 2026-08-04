from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_blockfht_projected_muon import (
    analyze_cell,
    gate_decision,
    weighted_summary,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


def test_full_space_projected_muon_recovers_dense_polar() -> None:
    torch.manual_seed(20260804)
    combined = torch.randn(4, 4)
    dense_polar = zeropower_via_newtonschulz5(combined, steps=5)
    config = {
        "block_fht_latent_ratio": 1.0,
        "block_fht_latent_ratios": None,
        "block_fht_layers": 2,
        "block_fht_seed": 17,
        "n_embd": 4,
        "n_head": 2,
    }
    metrics = analyze_cell(
        combined_momentum=combined,
        dense_polar=dense_polar,
        target="attn.c_proj",
        config=config,
        layer=0,
        ns_steps=5,
    )
    assert metrics["oracle_recovery"] > 0.99999
    assert metrics["proposed_recovery"] > 0.99999
    assert metrics["proposed_over_oracle"] > 0.99999


def test_weighted_summary_uses_dense_polar_energy() -> None:
    rows = [
        {
            "dense_polar_fro": 1.0,
            "oracle_recovery": 0.5,
            "oracle_cosine": 0.5**0.5,
            "proposed_recovery": 0.25,
            "proposed_cosine": 0.5,
            "raw_projected_momentum_recovery": 0.1,
            "raw_projected_momentum_cosine": 0.1**0.5,
        },
        {
            "dense_polar_fro": 3.0,
            "oracle_recovery": 1.0,
            "oracle_cosine": 1.0,
            "proposed_recovery": 0.5,
            "proposed_cosine": 0.5**0.5,
            "raw_projected_momentum_recovery": 0.2,
            "raw_projected_momentum_cosine": 0.2**0.5,
        },
    ]
    summary = weighted_summary(rows)
    assert abs(float(summary["oracle_recovery"]) - 0.95) < 1e-12
    assert abs(float(summary["proposed_recovery"]) - 0.475) < 1e-12
    assert abs(float(summary["proposed_over_oracle"]) - 0.5) < 1e-12


def test_gate_requires_aggregate_and_each_attention_target() -> None:
    passing = {
        "proposed_over_oracle": 0.60,
        "proposed_over_raw": 1.50,
    }
    by_target = {
        "attn.c_attn": {"proposed_over_oracle": 0.50, "proposed_cosine": 0.1},
        "attn.c_proj": {"proposed_over_oracle": 0.40, "proposed_cosine": 0.1},
    }
    decision, failures = gate_decision(
        passing,
        by_target,
        minimum_oracle_fraction=0.50,
        minimum_raw_multiplier=1.25,
        minimum_target_oracle_fraction=0.35,
    )
    assert decision == "AUTHORIZE_IMPLEMENTATION"
    assert failures == []

    by_target["attn.c_proj"]["proposed_over_oracle"] = 0.30
    decision, failures = gate_decision(
        passing,
        by_target,
        minimum_oracle_fraction=0.50,
        minimum_raw_multiplier=1.25,
        minimum_target_oracle_fraction=0.35,
    )
    assert decision == "REJECT_IMPLEMENTATION"
    assert "attn.c_proj_oracle_fraction" in failures
