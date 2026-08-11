from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_horizontal_tangent_audit import (
    horizontal_chart,
)


def test_horizontal_chart_removes_square_and_vertical_coordinates() -> None:
    generator = torch.Generator().manual_seed(11)
    first, _ = torch.linalg.qr(
        torch.randn(4, 4, generator=generator), mode="reduced"
    )
    second, _ = torch.linalg.qr(
        torch.randn(12, 2, generator=generator), mode="reduced"
    )
    cores = [
        first.reshape(1, 4, 4),
        second.reshape(4, 3, 2),
        torch.randn(2, 2, 1, generator=generator),
    ]

    def ambient(*values: torch.Tensor) -> torch.Tensor:
        return torch.cat([value.reshape(-1) for value in values])

    feature, coordinates, diagnostics = horizontal_chart(cores, ambient)
    assert [tuple(value.shape) for value in coordinates] == [(10, 2), (2, 2, 1)]
    assert sum(value.numel() for value in coordinates) == 24
    assert diagnostics["per_core"][0]["horizontal_coordinates"] == 0
    assert diagnostics["per_core"][1]["horizontal_coordinates"] == 20
    assert diagnostics["per_core"][1]["complement_cross_error_fro"] < 1e-5
    base = feature(*coordinates)
    expected = ambient(*cores)
    assert torch.allclose(base, expected, atol=1e-5)


def test_horizontal_jvp_is_orthogonal_to_base_core_columns() -> None:
    generator = torch.Generator().manual_seed(13)
    q, _ = torch.linalg.qr(
        torch.randn(12, 2, generator=generator), mode="reduced"
    )
    cores = [q.reshape(3, 4, 2), torch.ones(2, 1, 1)]

    def ambient(*values: torch.Tensor) -> torch.Tensor:
        return values[0].reshape(12, 2)

    feature, coordinates, _diagnostics = horizontal_chart(cores, ambient)
    tangent = (
        torch.randn(coordinates[0].shape, generator=generator),
        torch.zeros_like(coordinates[1]),
    )
    _, image = torch.func.jvp(feature, coordinates, tangent)
    assert torch.allclose(q.T @ image, torch.zeros(2, 2), atol=1e-5)
