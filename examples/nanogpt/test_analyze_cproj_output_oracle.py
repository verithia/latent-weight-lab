from __future__ import annotations

import torch

from examples.nanogpt.analyze_cproj_output_oracle import (
    LearnedConjugatedGivensOutputMix,
    fit_oracles,
    functional_orthogonal_procrustes,
    write_csv,
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


def test_write_csv_accepts_family_specific_metrics(tmp_path) -> None:
    output = tmp_path / "oracles.csv"
    write_csv(
        output,
        [
            {"family": "identity", "energy": 0.0},
            {
                "family": "givens4",
                "energy": 0.2,
                "operator_explained_energy": 0.1,
            },
        ],
    )
    text = output.read_text()
    assert "operator_explained_energy" in text
    assert "givens4" in text


def test_conjugated_givens_is_identity_initialized_and_norm_preserving() -> None:
    torch.manual_seed(19)
    module = LearnedConjugatedGivensOutputMix(
        features=16, stages=3, seed=23, block_size=8
    )
    values = torch.randn(7, 16)
    assert torch.allclose(module(values), values, atol=2e-6, rtol=2e-6)
    with torch.no_grad():
        module.angles.normal_(std=0.3)
    output = module(values)
    assert torch.allclose(
        output.norm(dim=-1), values.norm(dim=-1), atol=2e-5, rtol=2e-5
    )
