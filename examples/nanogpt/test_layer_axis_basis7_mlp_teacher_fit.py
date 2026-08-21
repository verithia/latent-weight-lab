from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from examples.nanogpt.analyze_layer_axis_basis7_mlp_teacher_fit import (
    ASSIGNMENT,
    BOUNDARIES,
    ROOTS,
    LayerAxisBasisMLP,
    coefficient_spectrum,
    initial_family,
)


def family() -> LayerAxisBasisMLP:
    generator = torch.Generator().manual_seed(17)
    basis_fc = torch.randn(2, 6, 4, generator=generator)
    basis_proj = torch.randn(2, 4, 6, generator=generator)
    coefficients = torch.tensor([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]])
    return LayerAxisBasisMLP(
        basis_fc=basis_fc,
        basis_proj=basis_proj,
        coefficients_fc=coefficients,
        coefficients_proj=coefficients.flip(-1),
        pre_gain=torch.ones(3, 6),
        output_log_gain=torch.zeros(3, 4),
    )


def test_forward_layer_matches_explicit_mixture() -> None:
    module = family()
    values = torch.randn(5, 4)
    c_fc = (module.coefficients_fc[1, :, None, None] * module.basis_fc).sum(0)
    c_proj = (
        module.coefficients_proj[1, :, None, None] * module.basis_proj
    ).sum(0)
    expected = F.linear(F.gelu(F.linear(values, c_fc)), c_proj)
    torch.testing.assert_close(module.forward_layer(1, values), expected)


def test_selected_forward_matches_layer_calls() -> None:
    module = family()
    values = torch.randn(3, 2, 2, 5, 4)
    layers = torch.tensor([0, 2])
    expected = torch.stack(
        [module.forward_layer(int(layer), values[:, index]) for index, layer in enumerate(layers)],
        dim=1,
    )
    torch.testing.assert_close(module.forward_selected(values, layers), expected)


def test_coefficients_only_freezes_basis_and_gains() -> None:
    module = family()
    selected = module.set_trainable(coefficients_only=True)
    assert selected == [module.coefficients_fc, module.coefficients_proj]
    assert not module.basis_fc.requires_grad
    assert not module.pre_gain.requires_grad


def test_coefficient_spectrum_reports_both_sides() -> None:
    row = coefficient_spectrum(family())
    assert set(row) == {"c_fc", "c_proj"}
    assert len(row["c_fc"]["singular_values"]) == 2
    assert 1.0 <= row["c_fc"]["entropy_effective_rank"] <= 2.0


class FakeMLP(nn.Module):
    def __init__(self, width: int, hidden: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.c_fc = nn.Linear(width, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, width, bias=False)
        with torch.no_grad():
            self.c_fc.weight.copy_(torch.randn(hidden, width, generator=generator))
            self.c_proj.weight.copy_(torch.randn(width, hidden, generator=generator))
        self.pregelu_gain = nn.Parameter(torch.ones(hidden))
        self.residual_output_log_gain = nn.Parameter(torch.zeros(width))
        self.residual_output_gain_scale = 1.0


def test_initial_family_exactly_encodes_seven_trunks() -> None:
    blocks = []
    roots: dict[int, FakeMLP] = {}
    for layer, atom in enumerate(ASSIGNMENT):
        if atom not in roots:
            roots[atom] = FakeMLP(4, 6, seed=atom + 1)
        blocks.append(SimpleNamespace(mlp=roots[atom]))
    checkpoint = SimpleNamespace(
        config=SimpleNamespace(
            mlp_shared_dense_trunk=True,
            mlp_shared_dense_trunk_groups=7,
            mlp_shared_dense_trunk_boundaries=BOUNDARIES,
        ),
        transformer=SimpleNamespace(h=blocks),
    )
    module = initial_family(checkpoint)
    assert module.atoms == len(ROOTS)
    assert module.layers == len(ASSIGNMENT)
    assert torch.equal(module.coefficients_fc.argmax(-1), torch.tensor(ASSIGNMENT))
    values = torch.randn(3, 4)
    for layer in range(len(ASSIGNMENT)):
        source = blocks[layer].mlp
        expected = F.linear(F.gelu(F.linear(values, source.c_fc.weight)), source.c_proj.weight)
        torch.testing.assert_close(module.forward_layer(layer, values), expected)
