from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_product_fht_rolling_raw import (
    coordinate_statistics,
    normalized_coordinate_step,
)
from latent_weight_lab import ProductFHTLinear


def test_normalized_coordinate_step_obeys_trust_cap() -> None:
    torch.manual_seed(11)
    module = ProductFHTLinear(
        8,
        4,
        factors=2,
        seed=41,
        weight_std=0.04,
        weight_space_muon=False,
        natural_gradient=True,
    )
    before = module.product_log_diagonals.detach().clone()
    diagnostics = normalized_coordinate_step(
        module,
        torch.randn(4, 8),
        learning_rate=0.5,
        coordinate_cap=0.01,
        norm_reference=torch.randn(4, 8) * 100.0,
    )
    assert diagnostics["applied_maximum_coordinate_update"] <= 0.0100001
    assert not torch.equal(before, module.product_log_diagonals)
    statistics = coordinate_statistics(module)
    assert statistics["coordinate_rms"] > 0.0
    assert statistics["coordinate_clamp_fraction"] == 0.0
