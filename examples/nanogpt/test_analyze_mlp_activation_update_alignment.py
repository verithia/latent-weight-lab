from __future__ import annotations

import torch
import pytest

import examples.nanogpt.analyze_mlp_activation_update_alignment as alignment
from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    model_from_snapshot,
    randomized_principal_basis,
    subspace_overlap,
    update_energy_capture,
)
from examples.nanogpt.parameter_trajectory import FULL_STATE_SCHEMA_VERSION


def test_aligned_activation_basis_captures_update() -> None:
    generator = torch.Generator().manual_seed(7)
    directions = torch.linalg.qr(
        torch.randn(16, 4, generator=generator)
    ).Q
    coefficients = torch.randn(64, 4, generator=generator)
    activations = coefficients @ directions.T
    update = torch.randn(6, 4, generator=generator) @ directions.T
    basis, singular, total = randomized_principal_basis(
        activations,
        4,
        center=True,
        seed=11,
        oversample=0,
        power_iterations=2,
    )
    assert singular.shape == (4,)
    assert total > 0.0
    assert update_energy_capture(update, basis) > 0.99999
    assert subspace_overlap(basis, directions) > 0.99999


def test_orthogonal_activation_basis_rejects_update() -> None:
    update = torch.zeros(3, 8)
    update[:, :2] = torch.eye(3, 2)
    basis = torch.eye(8)[:, 2:4]
    assert update_energy_capture(update, basis) == 0.0


def test_subspace_overlap_requires_matched_shapes() -> None:
    with pytest.raises(ValueError, match="same"):
        subspace_overlap(torch.eye(4)[:, :2], torch.eye(4)[:, :3])


def test_model_from_snapshot_restores_persistent_buffers(monkeypatch) -> None:
    class FakeGPT(torch.nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("materialized", torch.zeros(2))
            self.register_buffer("optimizer_step", torch.tensor(0, dtype=torch.int64))
            self.register_buffer("cache", torch.ones(1), persistent=False)

    monkeypatch.setattr(alignment, "GPTConfig", lambda **values: values)
    monkeypatch.setattr(alignment, "GPT", FakeGPT)
    payload = {
        "schema_version": FULL_STATE_SCHEMA_VERSION,
        "all_parameters": True,
        "all_buffers": True,
        "model_config": {},
        "parameters": {"weight": torch.tensor([2.0, 3.0])},
        "buffers": {
            "materialized": torch.tensor([5.0, 7.0]),
            "optimizer_step": torch.tensor(19, dtype=torch.int64),
        },
    }
    model = model_from_snapshot(payload, "cpu")
    torch.testing.assert_close(model.weight, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(model.materialized, torch.tensor([5.0, 7.0]))
    assert model.optimizer_step.item() == 19
    assert model.cache.item() == 1.0
