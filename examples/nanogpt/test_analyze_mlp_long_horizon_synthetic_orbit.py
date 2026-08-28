from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_long_horizon_synthetic_orbit import (
    orbit_accounting,
    orbit_span_metrics,
    schedule_from_registered_norms,
)


def test_orbit_accounting_is_exact_and_below_one_percent() -> None:
    accounting = orbit_accounting()
    assert accounting["prompt_scalars"] == 565_248
    assert accounting["matrix_base_scale_scalars"] == 24
    assert accounting["shared_schedule_ratio_scalars"] == 64
    assert accounting["total_state_scalars"] == 565_336
    assert accounting["state_fraction"] < 0.01


def test_registered_schedule_uses_node_median_ratios() -> None:
    norms = torch.tensor([[2.0, 4.0], [4.0, 4.0], [1.0, 8.0]])
    ratios, manifest = schedule_from_registered_norms(norms, steps=3)
    torch.testing.assert_close(ratios, torch.tensor([1.0, 1.0, 0.5]))
    assert manifest["steps"] == 3


def test_orbit_span_metrics_recovers_known_target_and_rank() -> None:
    torch.manual_seed(71)
    target = torch.randn(101)
    target /= target.norm()
    noise = torch.randn(7, 101)
    noise -= (noise @ target).unsqueeze(1) * target
    noise /= noise.norm(dim=1, keepdim=True)
    atoms = torch.cat((target.unsqueeze(0), noise), dim=0)
    metrics, gram, products = orbit_span_metrics(atoms, target)
    assert metrics["pc1_span_capture"] > 0.9999
    assert metrics["gram_numerical_rank"] == 8
    assert gram.shape == (8, 8)
    assert products.shape == (8,)
