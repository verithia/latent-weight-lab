from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_conditioned_gate_alignment import (
    FixedBasisBilinearOutputGate,
    ResidualConditionedOutputGate,
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
