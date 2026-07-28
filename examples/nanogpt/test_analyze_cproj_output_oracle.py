from __future__ import annotations

import torch

from examples.nanogpt.analyze_cproj_output_oracle import (
    fit_oracles,
    functional_orthogonal_procrustes,
)


def random_orthogonal(size: int) -> torch.Tensor:
    matrix = torch.randn(size, size)
    left, _, right_h = torch.linalg.svd(matrix)
    return left @ right_h


def test_functional_procrustes_recovers_rotation() -> None:
    torch.manual_seed(7)
    source = torch.randn(128, 12)
    expected = random_orthogonal(12)
    target = source @ expected.transpose(0, 1)
    actual = functional_orthogonal_procrustes(source, target)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_oracles_generalize_exact_diagonal_rotation() -> None:
    torch.manual_seed(11)
    source = torch.randn(256, 10)
    holdout = torch.randn(128, 10)
    diagonal = torch.linspace(0.5, 1.5, 10)
    rotation = random_orthogonal(10)
    target = (source * diagonal) @ rotation.transpose(0, 1)
    holdout_target = (holdout * diagonal) @ rotation.transpose(0, 1)
    rows = {
        row["family"]: row
        for row in fit_oracles(source, target, holdout, holdout_target)
    }
    assert rows["diagonal_then_orthogonal"][
        "holdout_explained_target_energy"
    ] > 0.99999
    assert rows["full_linear"]["holdout_explained_target_energy"] > 0.99999
