from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_muon_layer_attribution import (
    orbit_recoveries,
    positive_line_recovery,
    spearman,
    summarize_target,
)


def test_orbit_recoveries_exact_rank_two_right_direction() -> None:
    torch.manual_seed(20260808)
    weight, _ = torch.linalg.qr(torch.randn(8, 8))
    first = torch.randn(8)
    second = torch.randn(8)
    skew = torch.outer(first, second) - torch.outer(second, first)
    direction = weight @ skew
    full, active = orbit_recoveries(weight, direction, side="right", active_rank=2)
    assert full > 0.99999
    assert active > 0.99999


def test_positive_line_and_spearman() -> None:
    target = torch.tensor([1.0, 2.0])
    assert positive_line_recovery(target, target) == 1.0
    assert positive_line_recovery(target, -target) == 0.0
    assert spearman({0: 1.0, 1: 2.0, 2: 3.0}, {0: 2.0, 1: 4.0, 2: 6.0}) > 0.999


def test_temporal_summary_selects_stable_high_layers() -> None:
    rows = []
    for step in (0, 1, 2, 3):
        for layer in range(4):
            recovery = 0.8 - 0.1 * layer
            rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "direction_energy": 1.0,
                    "active_recovery": recovery,
                    "full_orbit_recovery": recovery / 0.9,
                }
            )
    result = summarize_target(
        rows,
        early_steps={0, 1},
        late_steps={2, 3},
        selected_layers=2,
        gates={
            "minimum_jaccard": 0.5,
            "minimum_spearman": 0.4,
            "minimum_separation": 1.1,
            "minimum_selected_recovery": 0.6,
            "minimum_fraction_of_full_orbit": 0.8,
        },
    )
    assert result["early_selected_layers"] == [0, 1]
    assert result["late_selected_layers"] == [0, 1]
    assert result["passed"]
