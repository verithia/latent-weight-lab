from __future__ import annotations

import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_matched_givens import MuonMatchedGivensLinear


def make_model() -> GPT:
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=("mlp.c_proj",),
            block_fht_mlp_cproj_muon_matched_givens=True,
            block_fht_mlp_cproj_muon_matched_givens_stages=1,
            block_fht_mlp_cproj_muon_matched_givens_residual_stages=0,
            block_fht_mlp_cproj_muon_matched_givens_neighbors=2,
            block_fht_mlp_cproj_muon_matched_givens_refresh_interval=3,
            block_fht_mlp_cproj_muon_matched_givens_fast_fresh=False,
            block_fht_mlp_pregelu_block_rotation_stages=1,
            block_fht_mlp_pregelu_block_rotation_size=4,
            block_fht_mlp_pregelu_block_rotation_basis_size=8,
            block_fht_mlp_pregelu_block_rotation_coordinate_scale=2.0,
            block_fht_mlp_hidden_block_rotation_stages=1,
            block_fht_mlp_hidden_block_rotation_size=4,
            block_fht_mlp_hidden_block_rotation_basis_size=8,
            block_fht_mlp_hidden_block_rotation_coordinate_scale=2.0,
            block_fht_mlp_hidden_gain=False,
            block_fht_mlp_output_block_rotation_stages=1,
            block_fht_mlp_output_block_rotation_size=4,
            block_fht_mlp_output_block_rotation_basis_size=8,
            block_fht_mlp_output_block_rotation_coordinate_scale=2.0,
            block_fht_mlp_residual_output_gain=False,
        )
    )


def chart_parameters(model: GPT) -> list[torch.nn.Parameter]:
    mlp = model.transformer.h[0].mlp
    assert mlp.hidden_block_rotation is not None
    assert mlp.output_block_rotation is not None
    return [
        mlp.hidden_block_rotation.coordinates,
        mlp.output_block_rotation.coordinates,
    ]


def test_materialized_muon_cproj_cached_forward_matches_live_forward() -> None:
    torch.manual_seed(13)
    model = make_model()
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_proj, MuonMatchedGivensLinear)
    with torch.no_grad():
        for parameter in chart_parameters(model):
            parameter.normal_(std=0.03)
    values = torch.randn(2, 3, 32)
    expected = mlp._charted_cproj(values)
    assert expected is not None
    mlp.prepare_charted_cproj_cache()
    assert mlp._cached_charted_cproj_weight is not None
    actual = mlp._charted_cproj(values)
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_materialized_muon_cproj_cache_projects_only_chart_gradients() -> None:
    torch.manual_seed(17)
    model = make_model()
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_proj, MuonMatchedGivensLinear)
    base_before = mlp.c_proj.weight.detach().clone()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in chart_parameters(model):
        parameter.requires_grad_(True)
    values = torch.randn(2, 3, 32)
    mlp.prepare_charted_cproj_cache()
    output = mlp._charted_cproj(values)
    assert output is not None
    output.square().mean().backward()
    mlp.flush_charted_cproj_cache(project_base_gradient=False)
    assert mlp.c_proj.weight.grad is None
    assert torch.equal(mlp.c_proj.weight, base_before)
    assert all(parameter.grad is not None for parameter in chart_parameters(model))
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in chart_parameters(model)
    )


def test_identity_materialized_muon_chart_preserves_weight_bitwise() -> None:
    model = make_model()
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_proj, MuonMatchedGivensLinear)
    assert all(
        torch.count_nonzero(parameter) == 0
        for parameter in chart_parameters(model)
    )
    charted = mlp._materialize_charted_cproj_weight(mlp.c_proj.weight)
    assert torch.equal(charted, mlp.c_proj.weight)


def test_identity_pregelu_frame_is_bounded_to_fht_roundoff() -> None:
    model = make_model()
    mlp = model.transformer.h[0].mlp
    base = mlp._cfc_base_weight()
    delta = (
        mlp._materialize_charted_cfc_weight(base) - base
    ).detach().float()
    assert float(delta.abs().max()) <= 3e-8
    assert float(delta.norm() / base.detach().float().norm()) <= 2e-7
