from __future__ import annotations

import torch
import pytest

from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    align_layer_sequence,
    causal_history_indices,
    fixed_blockfht_basis,
    fit_coordinates,
    flatten_state,
    orthonormal_span,
    recovery_fraction,
    score_basis,
    sparse_moe_output,
    unflatten_state,
)


def test_layer_state_roundtrip() -> None:
    state = LayerState(
        torch.randn(3, 4),
        torch.randn(3, 6, 4),
        torch.randn(3, 4, 6),
    )
    restored = unflatten_state(flatten_state(state), state)
    assert torch.equal(restored.router, state.router)
    assert torch.equal(restored.c_fc, state.c_fc)
    assert torch.equal(restored.c_proj, state.c_proj)


def test_sparse_moe_output_matches_scalar_reference() -> None:
    torch.manual_seed(7)
    state = LayerState(
        torch.randn(3, 4),
        torch.randn(3, 6, 4),
        torch.randn(3, 4, 6),
    )
    x = torch.randn(5, 4)
    actual = sparse_moe_output(state, x, top_k=2)
    logits = x @ state.router.T
    tie = torch.arange(3, dtype=logits.dtype)
    selected = torch.topk(logits - tie * torch.finfo(logits.dtype).eps, 2, dim=-1).indices
    probabilities = torch.softmax(logits.gather(-1, selected), dim=-1)
    expected = torch.zeros_like(actual)
    for token in range(x.shape[0]):
        for slot in range(2):
            expert = int(selected[token, slot])
            hidden = torch.nn.functional.gelu(state.c_fc[expert] @ x[token])
            expected[token] += probabilities[token, slot] * (state.c_proj[expert] @ hidden)
    torch.testing.assert_close(actual, expected)


def test_causal_history_excludes_target_and_future() -> None:
    assert causal_history_indices(8, 4) == [4, 5, 6, 7]
    assert causal_history_indices(2, 4) == [0, 1]


def test_orthonormal_span_has_requested_scaled_gram() -> None:
    torch.manual_seed(11)
    basis = orthonormal_span([torch.randn(20) for _ in range(6)], 4, scale=3.0)
    torch.testing.assert_close(basis @ basis.T, torch.eye(4) * 9.0, atol=1e-4, rtol=1e-4)


def test_functional_coordinate_fit_recovers_in_span_target() -> None:
    torch.manual_seed(13)
    basis = torch.randn(4, 5, 3)
    truth = torch.tensor([0.5, -1.0, 0.25, 2.0])
    target = torch.einsum("k,knd->nd", truth, basis)
    fitted = fit_coordinates(basis, target, ridge_ratio=1e-10)
    predicted = torch.einsum("k,knd->nd", fitted, basis)
    assert recovery_fraction(predicted, target) > 0.999999


def test_fixed_blockfht_basis_is_exact_budget_and_scaled() -> None:
    state = LayerState(
        torch.zeros(2, 4),
        torch.zeros(2, 6, 4),
        torch.zeros(2, 4, 6),
    )
    basis = fixed_blockfht_basis(state, rank=4, scale=2.0, layer=0, device="cpu")
    assert basis.shape == (4, flatten_state(state).numel())
    torch.testing.assert_close(basis @ basis.T, torch.eye(4) * 4.0, atol=1e-5, rtol=1e-5)


def test_materialized_oracle_recovers_small_in_basis_update() -> None:
    torch.manual_seed(17)
    left = LayerState(
        torch.randn(2, 4) * 0.1,
        torch.randn(2, 6, 4) * 0.1,
        torch.randn(2, 4, 6) * 0.1,
    )
    raw_basis = torch.randn(4, flatten_state(left).numel())
    basis = orthonormal_span(raw_basis, 4, scale=1e-3)
    direction = unflatten_state(0.3 * basis[0] - 0.2 * basis[1], left)
    right = LayerState(
        left.router + direction.router,
        left.c_fc + direction.c_fc,
        left.c_proj + direction.c_proj,
    )
    x = torch.randn(64, 4)
    result = score_basis(
        left, right, basis, x[:32], x[32:], top_k=2,
        ridge_ratio=1e-10, device="cpu",
    )
    assert result["heldout_materialized_recovery"] > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression test")
def test_align_layer_sequence_accepts_cpu_assignment_indices_on_cuda() -> None:
    torch.manual_seed(19)
    left = LayerState(
        torch.randn(2, 4),
        torch.randn(2, 6, 4),
        torch.randn(2, 4, 6),
    )
    right = LayerState(
        left.router + torch.randn_like(left.router) * 1e-3,
        left.c_fc + torch.randn_like(left.c_fc) * 1e-3,
        left.c_proj + torch.randn_like(left.c_proj) * 1e-3,
    )
    activations = torch.randn(32, 4, device="cuda")
    aligned, rows = align_layer_sequence(
        [left, right], activations[:16], activations[16:], "cuda"
    )
    assert len(aligned) == 2
    assert len(rows) == 2
    assert aligned[-1].c_fc.device.type == "cpu"
