from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_conditioned_gate_alignment import (
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
