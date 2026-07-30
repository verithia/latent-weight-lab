from __future__ import annotations

import torch

from examples.nanogpt.analyze_attention_blockfht_tangent import (
    latent_geometry,
    project_c_attn,
    projection_metrics,
)


def test_latent_geometry_matches_one_percent_complete_blocks() -> None:
    geometry = latent_geometry(98_304, 0.01)
    assert geometry["latent_dim"] == 983
    assert geometry["block_size"] == 1024
    assert geometry["frame_bound"] == 96


def test_qk_headwise_and_v_projection_preserve_shape() -> None:
    torch.manual_seed(20260730)
    matrix = torch.randn(24, 8)
    projected, geometry = project_c_attn(
        matrix,
        n_embd=8,
        n_head=2,
        ratio=0.25,
        layers=2,
        base_seed=17,
        layer=1,
    )
    assert projected.shape == matrix.shape
    assert geometry["qk_heads"] == 2
    assert geometry["total_size"] == matrix.numel()
    projected_twice, _ = project_c_attn(
        projected,
        n_embd=8,
        n_head=2,
        ratio=0.25,
        layers=2,
        base_seed=17,
        layer=1,
    )
    torch.testing.assert_close(projected_twice, projected, rtol=1e-5, atol=1e-5)


def test_projection_metrics_reports_exact_tangent_capture() -> None:
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metrics = projection_metrics(target, target, target, target)
    assert metrics["tangent_chord_energy_fraction"] == 1.0
    assert metrics["projected_direction_energy_fraction"] == 1.0
    assert metrics["projected_positive_step_line_recovery"] > 0.999999
