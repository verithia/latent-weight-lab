from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_global_sign_bank import (
    learned_sign_atoms,
    normalized_sign,
    project_rows,
    residual_budget_per_matrix,
    synthetic_self_check,
    union_capture,
)


def test_learned_sign_atom_captures_dominant_direction() -> None:
    torch.manual_seed(7)
    dominant = torch.randn(64)
    rows = torch.stack((dominant, 0.1 * torch.randn(64)))
    weights = torch.tensor((0.99, 0.01))
    atom = learned_sign_atoms(rows, weights, 1)
    capture = project_rows(dominant.unsqueeze(0), atom)
    assert float(capture[0]) > 0.50


def test_union_capture_recovers_own_span() -> None:
    assert synthetic_self_check("cpu") > 0.999999


def test_union_capture_adds_orthogonal_coordinate_axes() -> None:
    atom = normalized_sign(torch.tensor([[1.0, -1.0, 1.0, -1.0]]))
    support = torch.tensor([0])
    rows = torch.stack((atom[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))
    captures = union_capture(rows, atom, support)
    assert torch.all(captures > 0.999999)


def test_exact_one_percent_byte_accounting() -> None:
    ambient = 3_072 * 768
    residual, record = residual_budget_per_matrix(
        dense_scalars_per_matrix=ambient,
        deployment_matrix_count=24,
        maximum_fraction=0.01,
        fixed_bits=3 * ambient,
        coefficients_per_matrix=3,
        residual=True,
    )
    assert residual > 0
    assert 0.0099 < float(record["total_checkpoint_byte_fraction"]) <= 0.01
    assert int(record["fixed_orientation_bits"]) == 3 * ambient
