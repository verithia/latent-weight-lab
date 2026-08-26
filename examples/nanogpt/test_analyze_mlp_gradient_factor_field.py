from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_gradient_factor_field import (
    centered_kernel_spectrum,
    energy_dimensions,
    grouped_fht_frame,
    projector_kernel,
    support_capture,
    union_spectrum,
)


def test_projector_kernel_is_gauge_invariant() -> None:
    torch.manual_seed(1)
    frame = torch.linalg.qr(torch.randn(12, 3), mode="reduced").Q
    rotation = torch.linalg.qr(torch.randn(3, 3), mode="reduced").Q
    kernel = projector_kernel([frame, frame @ rotation])
    torch.testing.assert_close(kernel, torch.ones_like(kernel), atol=1e-6, rtol=1e-6)
    assert centered_kernel_spectrum(kernel).sum() < 1e-6


def test_union_spectrum_and_energy_dimensions() -> None:
    first = torch.eye(6)[:, :2]
    second = torch.eye(6)[:, 2:4]
    spectrum = union_spectrum([first, second])
    result = energy_dimensions(spectrum)
    assert result["dimension_90"] == 4
    assert result["dimension_95"] == 4
    assert result["participation_dimension"] == 4.0


def test_grouped_fht_and_support_are_norm_preserving() -> None:
    torch.manual_seed(2)
    frame = torch.linalg.qr(torch.randn(12, 3), mode="reduced").Q
    transformed = grouped_fht_frame(frame)
    torch.testing.assert_close(transformed.norm(), frame.norm(), atol=1e-6, rtol=1e-6)
    support = torch.arange(frame.shape[0])
    assert support_capture(frame, support) > 0.999999
