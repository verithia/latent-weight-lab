from __future__ import annotations

import pytest
import torch

from examples.nanogpt.analyze_mlp_conditioned_gate_alignment import (
    FixedBasisBilinearOutputGate,
    MultiHeadPostGeluHiddenSelfBilinearGate,
    PostGeluConditionedBilinearOutputGate,
    PostGeluHiddenSelfBilinearGate,
    ResidualConditionedOutputGate,
    UntiedFixedBasisBilinearOutputGate,
    alignment_rows,
)


def test_conditioned_gate_is_identity_and_has_token_conditioned_gradient() -> None:
    gate = ResidualConditionedOutputGate(3)
    condition = torch.tensor([[1.0, -2.0, 0.5], [0.5, 1.5, -1.0]])
    update = torch.tensor([[2.0, 3.0, 4.0], [-1.0, 2.0, 0.5]])

    output = gate(condition, update)
    torch.testing.assert_close(output, update)
    output.sum().backward()

    assert gate.slope.grad is not None
    assert gate.bias.grad is not None
    torch.testing.assert_close(
        gate.slope.grad, (condition * update).sum(dim=0)
    )
    torch.testing.assert_close(gate.bias.grad, update.sum(dim=0))


def test_fixed_basis_bilinear_gate_is_identity_and_can_rotate_channels() -> None:
    gate = FixedBasisBilinearOutputGate(
        4,
        basis_block_size=4,
        seed=17,
    )
    condition = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    update = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    torch.testing.assert_close(gate(condition, update), update)
    with torch.no_grad():
        gate.slope.fill_(1.0)
    correction = gate(condition, update) - update

    assert correction.abs().sum() > 0
    assert correction[..., [0, 2, 3]].abs().sum() > 0


def test_fixed_basis_bilinear_gate_has_dynamic_and_static_gradients() -> None:
    gate = FixedBasisBilinearOutputGate(
        8,
        basis_block_size=4,
        seed=23,
    )
    condition = torch.randn(2, 3, 8)
    update = torch.randn(2, 3, 8)

    gate(condition, update).square().mean().backward()

    assert gate.slope.grad is not None
    assert gate.bias.grad is not None
    assert gate.slope.grad.abs().sum() > 0
    assert gate.bias.grad.abs().sum() > 0


def test_untied_fixed_basis_bilinear_gate_is_identity_and_routes_channels() -> None:
    gate = UntiedFixedBasisBilinearOutputGate(
        8,
        basis_block_size=4,
        condition_seed=11,
        update_seed=17,
        output_seed=23,
    )
    condition = torch.randn(2, 3, 8)
    update = torch.randn(2, 3, 8)

    torch.testing.assert_close(gate(condition, update), update)
    with torch.no_grad():
        gate.slope.fill_(1.0)
    correction = gate(condition, update) - update

    assert correction.abs().sum() > 0
    assert not torch.equal(
        gate.condition_basis.permutation,
        gate.update_basis.permutation,
    )
    assert not torch.equal(
        gate.update_basis.permutation,
        gate.output_basis.permutation,
    )


def test_untied_fixed_basis_bilinear_gate_has_slope_gradient() -> None:
    gate = UntiedFixedBasisBilinearOutputGate(
        8,
        basis_block_size=4,
        condition_seed=29,
        update_seed=31,
        output_seed=37,
    )
    condition = torch.randn(2, 3, 8)
    update = torch.randn(2, 3, 8)

    gate(condition, update).square().mean().backward()

    assert gate.slope.grad is not None
    assert gate.slope.grad.abs().sum() > 0


def test_postgelu_conditioned_gate_is_identity_and_has_dynamic_gradient() -> None:
    gate = PostGeluConditionedBilinearOutputGate(
        8,
        basis_block_size=4,
        condition_seed=11,
        update_seed=17,
        output_seed=23,
        projection_seed=29,
    )
    activated = torch.randn(2, 3, 32)
    update = torch.randn(2, 3, 8)

    torch.testing.assert_close(gate(activated, update), update)
    gate(activated, update).square().mean().backward()

    assert gate.slope.grad is not None
    assert gate.slope.grad.abs().sum() > 0


def test_postgelu_condition_projection_is_per_token_rms_normalized() -> None:
    gate = PostGeluConditionedBilinearOutputGate(
        8,
        basis_block_size=4,
        projection_seed=31,
    )
    condition = gate.activation_condition(torch.randn(2, 3, 32))
    rms = condition.float().square().mean(dim=-1).sqrt()

    torch.testing.assert_close(rms, torch.ones_like(rms), atol=2e-5, rtol=0)


def test_postgelu_hidden_self_gate_is_identity_and_has_dynamic_gradient() -> None:
    gate = PostGeluHiddenSelfBilinearGate(
        16,
        basis_block_size=4,
        condition_seed=11,
        update_seed=17,
        output_seed=23,
    )
    activated = torch.randn(2, 3, 16)

    torch.testing.assert_close(gate(activated), activated)
    gate(activated).square().mean().backward()

    assert gate.slope.grad is not None
    assert gate.slope.grad.abs().sum() > 0


def test_postgelu_hidden_self_condition_is_per_token_rms_normalized() -> None:
    gate = PostGeluHiddenSelfBilinearGate(
        16,
        basis_block_size=4,
    )
    condition = gate.activation_condition(torch.randn(2, 3, 16))
    rms = condition.float().square().mean(dim=-1).sqrt()

    torch.testing.assert_close(rms, torch.ones_like(rms), atol=2e-5, rtol=0)


def test_multihead_postgelu_hidden_self_gate_is_identity_and_dynamic() -> None:
    gate = MultiHeadPostGeluHiddenSelfBilinearGate(
        16,
        2,
        basis_block_size=4,
        condition_seed=11,
        update_seed=17,
        output_seed=23,
    )
    activated = torch.randn(2, 3, 16)

    torch.testing.assert_close(gate(activated), activated)
    gate(activated).square().mean().backward()

    assert gate.slope.shape == (2, 16)
    assert gate.slope.grad is not None
    assert gate.slope.grad[0].abs().sum() > 0
    assert gate.slope.grad[1].abs().sum() > 0
    assert not torch.equal(
        gate.condition_bases[0].permutation,
        gate.condition_bases[1].permutation,
    )


def test_multihead_postgelu_hidden_self_gate_validates_head_count() -> None:
    with pytest.raises(ValueError, match="at least two heads"):
        MultiHeadPostGeluHiddenSelfBilinearGate(16, 1, basis_block_size=4)


def test_gate_alignment_rows_include_groups_and_layers() -> None:
    left = {
        "layer.0.slope": torch.tensor([1.0, 2.0]),
        "layer.0.bias": torch.tensor([3.0, 4.0]),
        "layer.2.slope": torch.tensor([5.0, 6.0]),
        "layer.2.bias": torch.tensor([7.0, 8.0]),
    }
    rows = alignment_rows(
        left,
        {key: value.clone() for key, value in left.items()},
        comparison="test",
        split="fit",
    )

    assert len(rows) == 1 + 2 + 2 * 3
    assert rows[0]["scope"] == "global"
    assert rows[0]["cosine"] == 1.0


def test_gate_alignment_rows_support_slope_only_probe() -> None:
    gradients = {
        "layer.0.slope": torch.tensor([1.0, 2.0]),
        "layer.2.slope": torch.tensor([3.0, 4.0]),
    }
    rows = alignment_rows(
        gradients,
        {key: value.clone() for key, value in gradients.items()},
        comparison="test",
        split="fit",
    )

    assert len(rows) == 1 + 1 + 2 * 2
    assert {row["group"] for row in rows if row["scope"] == "group"} == {
        "slope"
    }
    assert all(abs(row["cosine"] - 1.0) < 1e-6 for row in rows)
