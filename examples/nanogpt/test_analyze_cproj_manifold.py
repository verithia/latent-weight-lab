from __future__ import annotations

import torch

from examples.nanogpt.analyze_cproj_manifold import (
    base_c_proj_weight,
    effective_c_proj_weight,
    spectral_residual_metrics,
    spectral_residual_weight,
)
from examples.nanogpt.model import GPT, GPTConfig


def _tiny_spectral_model() -> GPT:
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_fht=True,
        block_fht_targets=("mlp.c_proj",),
        block_fht_latent_ratio=0.25,
        block_fht_layers=1,
        block_fht_cproj_spectral_resid_rank=4,
        block_fht_cproj_spectral_resid_scale_init=1.0,
        block_fht_cproj_spectral_resid_seed=17001,
        block_fht_cproj_spectral_resid_muon_matrix=True,
    )
    model = GPT(config)
    with torch.no_grad():
        model.transformer.h[0].mlp.cproj_spectral_resid_diag.copy_(
            torch.tensor([[0.5, -0.25, 0.125, -0.0625]])
        )
    return model


def test_effective_cproj_weight_includes_fixed_basis_spectral_residual() -> None:
    model = _tiny_spectral_model()
    mlp = model.transformer.h[0].mlp
    residual = spectral_residual_weight(model, 0)
    assert residual is not None

    expected = (
        mlp.cproj_spectral_resid_out_basis
        * mlp.cproj_spectral_resid_diag.reshape(-1).unsqueeze(0)
    ) @ mlp.cproj_spectral_resid_in_basis.transpose(0, 1)
    torch.testing.assert_close(residual, expected)
    torch.testing.assert_close(effective_c_proj_weight(model, 0), base_c_proj_weight(model, 0) + expected)


def test_spectral_residual_metrics_report_learned_diagonal_geometry() -> None:
    model = _tiny_spectral_model()
    metrics = spectral_residual_metrics(model, 0)
    assert metrics is not None
    assert metrics["rank"] == 4
    assert metrics["diag_nonzero_fraction"] == 1.0
    assert metrics["diag_soft_rank"] > 1.0
    assert metrics["diag_hard_rank"] > 1.0
    assert metrics["coefficient_matrix_hard_rank"] == metrics["diag_hard_rank"]
    assert metrics["residual_fro_norm"] > 0.0
    assert metrics["residual_to_base_fro"] > 0.0
    assert metrics["in_basis_orthogonality_max_abs"] < 1e-6
    assert metrics["out_basis_orthogonality_max_abs"] < 1e-6


def test_spectral_residual_metrics_report_full_core_singular_geometry() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht=True,
            block_fht_targets=("mlp.c_proj",),
            block_fht_latent_ratio=0.25,
            block_fht_layers=1,
            block_fht_cproj_spectral_resid_rank=4,
            block_fht_cproj_spectral_resid_scale_init=1.0,
            block_fht_cproj_spectral_resid_seed=17001,
            block_fht_cproj_spectral_resid_full_core=True,
        )
    )
    with torch.no_grad():
        model.transformer.h[0].mlp.cproj_spectral_resid_diag.copy_(
            torch.diag(torch.tensor([1.0, 0.5, 0.0, 0.0]))
        )
    metrics = spectral_residual_metrics(model, 0)
    assert metrics is not None
    assert metrics["parameter_count"] == 16
    assert metrics["coefficient_matrix_hard_rank"] > 1.0
    assert metrics["coefficient_matrix_hard_rank"] < 2.0
    assert metrics["coefficient_matrix_stable_rank"] == 1.25
    assert metrics["coefficient_matrix_spectral_norm"] == 1.0
