from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_cproj_fresh_residual_gate import (
    aggregate,
    build_candidate,
    metrics,
)


def test_metrics_exact_prediction() -> None:
    target = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    row = metrics(target, target)
    assert row["fixed_scale_recovery"] == 1.0
    assert abs(row["positive_line_recovery"] - 1.0) < 1e-12
    assert abs(row["cosine"] - 1.0) < 1e-12
    assert abs(row["descent_fraction"] - 1.0) < 1e-12


def test_aggregate_uses_energy_weighting() -> None:
    rows = [
        {"target_energy": 1.0, "prediction_energy": 1.0, "residual_energy": 0.0, "descent_fraction": 1.0},
        {"target_energy": 3.0, "prediction_energy": 0.0, "residual_energy": 3.0, "descent_fraction": 0.0},
    ]
    result = aggregate(rows)
    assert abs(float(result["fixed_scale_recovery"]) - 0.25) < 1e-12
    assert abs(float(result["descent_fraction"]) - 0.25) < 1e-12


def test_build_candidate_is_deterministic_and_finite(tmp_path) -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(16, 16, generator=generator) * 0.02
    target = torch.randn(16, 16, generator=generator) * 1e-3
    passes = [{"side": "input", "stages": 2}, {"side": "output", "stages": 2}]
    first, first_rows = build_candidate(
        source, target, passes, neighbors=4, seed=11, native_cache=tmp_path
    )
    second, second_rows = build_candidate(
        source, target, passes, neighbors=4, seed=11, native_cache=tmp_path
    )
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()
    assert [row["coordinates"] for row in first_rows] == [16, 16]
    assert [row["side"] for row in second_rows] == ["input", "output"]
