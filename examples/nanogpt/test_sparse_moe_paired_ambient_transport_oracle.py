from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_paired_ambient_transport_oracle import (
    PairedAmbientTransportChart,
    _apply_householder,
    base_coordinate_count,
    transport_coordinate_count,
)


def _module(kind: str) -> PairedAmbientTransportChart:
    torch.manual_seed(3)
    write, _ = torch.linalg.qr(torch.randn(8, 5))
    return PairedAmbientTransportChart(
        write_basis=write,
        hidden_width=12,
        padded_width=16,
        tensor_layers=2,
        experts=2,
        feature_seed=7,
        pre_matching_seed=11,
        post_matching_seed=13,
        procedural_map_seed=17,
        transport_kind=kind,
        householder_reflectors=4,
        monarch_block_width=3,
        monarch_permutation_seed=19,
        device="cpu",
    )


def test_registered_coordinate_counts() -> None:
    base = base_coordinate_count(
        rank=480, input_width=768, hidden_width=1536,
        tensor_layers=12, experts=8,
    )
    assert base == 1_078_272
    assert transport_coordinate_count(
        kind="householder", tensor_layers=12, hidden_width=1536,
        householder_reflectors=8, monarch_block_width=16,
    ) == 147_456
    assert transport_coordinate_count(
        kind="monarch", tensor_layers=12, hidden_width=1536,
        householder_reflectors=8, monarch_block_width=16,
    ) == 589_824


def test_householder_transpose_is_inverse() -> None:
    torch.manual_seed(23)
    vectors = torch.randn(4, 12)
    values = torch.randn(2, 5, 12)
    moved = _apply_householder(values, vectors, transpose=False)
    restored = _apply_householder(moved, vectors, transpose=True)
    torch.testing.assert_close(restored, values, rtol=2e-5, atol=2e-6)


def test_monarch_identity_initialization_is_exact_identity() -> None:
    module = _module("monarch")
    values = torch.randn(2, 5, 12)
    moved = module._transport(values, layer=0, transpose=False)
    restored = module._transport(values, layer=0, transpose=True)
    torch.testing.assert_close(moved, values)
    torch.testing.assert_close(restored, values)


def test_step_zero_branch_and_jvp_are_zero_for_all_trials() -> None:
    inputs = torch.randn(2, 4, 8)
    directions = torch.randn_like(inputs)
    for kind in ("householder", "monarch", "control"):
        module = _module(kind)
        output, jvp = module.function_and_jvp(inputs, directions, layer=0)
        assert torch.count_nonzero(output) == 0
        assert torch.count_nonzero(jvp) == 0


def test_analytic_jvp_matches_finite_difference() -> None:
    torch.manual_seed(29)
    for kind in ("householder", "monarch"):
        module = _module(kind)
        with torch.no_grad():
            module.output_gain.normal_(std=0.1)
            module.hidden_bias.normal_(std=0.03)
            if module.monarch_blocks is not None:
                module.monarch_blocks.add_(torch.randn_like(module.monarch_blocks) * 0.01)
        inputs = torch.randn(2, 3, 8)
        directions = torch.randn_like(inputs)
        _, jvp = module.function_and_jvp(inputs, directions, layer=1)
        epsilon = 1e-3
        plus, _ = module.function_and_jvp(
            inputs + epsilon * directions, torch.zeros_like(inputs), layer=1
        )
        minus, _ = module.function_and_jvp(
            inputs - epsilon * directions, torch.zeros_like(inputs), layer=1
        )
        finite = (plus - minus) / (2.0 * epsilon)
        torch.testing.assert_close(jvp, finite, rtol=1e-2, atol=1e-4)
