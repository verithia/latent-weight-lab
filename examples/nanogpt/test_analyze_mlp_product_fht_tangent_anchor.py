from __future__ import annotations

import pytest
import torch

from examples.nanogpt.analyze_mlp_product_fht_tangent_anchor import (
    chronological_split,
    deterministic_mixture,
    natural_pullback_action,
    summarize,
)
from latent_weight_lab import ProductFHTLinear


def test_split_and_mixture_are_deterministic() -> None:
    assert chronological_split(118, 119, 179) == "discovery"
    assert chronological_split(119, 119, 179) == "validation"
    assert chronological_split(179, 119, 179) == "test"
    gradients = torch.randn(7, 4, 8)
    first = deterministic_mixture(gradients, update=3, width=4, seed=9)
    second = deterministic_mixture(gradients, update=3, width=4, seed=9)
    torch.testing.assert_close(first, second)
    assert first.norm().item() == pytest.approx(1.0)


def test_product_tangent_action_is_finite_and_anchor_differentiable() -> None:
    torch.manual_seed(7)
    module = ProductFHTLinear(
        8,
        4,
        factors=2,
        seed=31,
        weight_std=0.04,
        weight_space_muon=False,
        natural_gradient=True,
    )
    target = torch.randn(4, 8)
    action, cosine, score = natural_pullback_action(
        module, target, differentiable_anchor=True
    )
    assert action.shape == target.shape
    assert torch.isfinite(action).all()
    assert 0.0 <= float(score.detach()) <= 1.0 + 1e-6
    (-score).backward()
    assert module.product_log_diagonals.grad is not None
    assert torch.isfinite(module.product_log_diagonals.grad).all()
    assert module.product_output_log_gain.grad is not None
    assert torch.isfinite(module.product_output_log_gain.grad).all()
    assert float(cosine.detach()) >= -1e-6


def test_summary_reports_identity_enrichment() -> None:
    rows = []
    for anchor, value in (("identity", 0.1), ("fitted", 0.4)):
        for split in ("discovery", "validation", "test"):
            rows.append(
                {
                    "anchor": anchor,
                    "split": split,
                    "action_capture": value,
                    "action_cosine": value**0.5,
                }
            )
    result = summarize(rows, parameter="x")
    fitted = [row for row in result if row["anchor"] == "fitted"]
    assert all(row["enrichment_over_identity"] == 4.0 for row in fitted)
