from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_global_conditional_tangent_audit import (
    centered_normalized,
    classify,
    per_layer_projection_scores,
    projection_scores,
)


def test_centered_normalized_removes_layer_shared_component() -> None:
    torch.manual_seed(3)
    gradients = torch.randn(2, 8, 16)
    normalized = centered_normalized(gradients)
    torch.testing.assert_close(normalized.norm(dim=-1), torch.ones(2, 8))
    centered = gradients - gradients.mean(dim=1, keepdim=True)
    assert float(centered.mean(dim=1).abs().max()) < 1e-6


def test_global_projection_recovers_known_rank_three_subspace() -> None:
    torch.manual_seed(5)
    basis, _ = torch.linalg.qr(torch.randn(24, 3))
    coefficients = torch.randn(2, 8, 3)
    train = coefficients @ basis.T
    test = (coefficients + 0.01 * torch.randn_like(coefficients)) @ basis.T
    train = train / train.norm(dim=-1, keepdim=True)
    test = test / test.norm(dim=-1, keepdim=True)
    scores = projection_scores(train, test, 3)
    assert scores["explained_energy_minimum_row"] > 0.999


def test_per_layer_projection_localizes_different_layer_subspaces() -> None:
    torch.manual_seed(7)
    values = []
    for _layer in range(3):
        basis, _ = torch.linalg.qr(torch.randn(32, 4))
        rows = torch.randn(8, 4) @ basis.T
        values.append(rows / rows.norm(dim=-1, keepdim=True))
    gradients = torch.stack(values)
    scores = per_layer_projection_scores(gradients, gradients, 4)
    assert scores["explained_energy_minimum_row"] > 0.999


def test_classification_is_conservative() -> None:
    assert classify(True, True, True) == "GLOBAL_RANK7_CONDITIONAL_TANGENT_SUPPORTED"
    assert classify(False, True, True) == "CROSS_LAYER_TEMPLATE_SHARING_LIMIT"
    assert classify(False, False, False) == "CROSS_BANK_TANGENT_INSTABILITY"
