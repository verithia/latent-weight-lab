from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from examples.nanogpt.analyze_layer_axis_basis7_mlp_teacher_fit import (
    ASSIGNMENT,
    BOUNDARIES,
)
from examples.nanogpt.analyze_two_bank_atomwise_recombination import (
    TwoBankAtomwiseMLP,
    deterministic_pairing,
    initial_family,
    semantic_signatures,
)


def family() -> TwoBankAtomwiseMLP:
    generator = torch.Generator().manual_seed(19)
    return TwoBankAtomwiseMLP(
        basis_fc=torch.randn(7, 6, 4, generator=generator),
        basis_proj=torch.randn(7, 4, 6, generator=generator),
        alpha_fc=torch.rand(7, 6, generator=generator),
        alpha_proj=torch.rand(7, 6, generator=generator),
        pre_gain=torch.ones(12, 6),
        output_log_gain=torch.zeros(12, 4),
        pairing=torch.arange(6),
    )


def test_semantic_signatures_have_unit_rows() -> None:
    module = family()
    signatures = semantic_signatures(module.basis_fc[5], module.basis_proj[5])
    torch.testing.assert_close(signatures.norm(dim=1), torch.ones(6))


def test_pairing_recovers_permuted_complete_neurons() -> None:
    generator = torch.Generator().manual_seed(23)
    a_fc = torch.randn(8, 5, generator=generator)
    a_proj = torch.randn(5, 8, generator=generator)
    source = torch.tensor([3, 0, 6, 1, 7, 4, 2, 5])
    b_fc = a_fc.index_select(0, source)
    b_proj = a_proj.index_select(1, source)
    permutation, diagnostics = deterministic_pairing(
        a_fc, a_proj, b_fc, b_proj, top_k=4
    )
    inverse = torch.argsort(source)
    assert torch.equal(permutation.cpu(), inverse)
    assert diagnostics["matched_similarity_minimum"] > 0.999


def test_late_weights_match_explicit_atomwise_interpolation() -> None:
    module = family()
    layer = 7
    offset = layer - 5
    c_fc, c_proj = module.weights(layer)
    a_fc = module.alpha_fc[offset][:, None]
    a_proj = module.alpha_proj[offset][None, :]
    torch.testing.assert_close(
        c_fc,
        a_fc * module.basis_fc[5] + (1.0 - a_fc) * module.basis_fc[6],
    )
    torch.testing.assert_close(
        c_proj,
        a_proj * module.basis_proj[5] + (1.0 - a_proj) * module.basis_proj[6],
    )


def test_forward_layer_matches_materialized_weights() -> None:
    module = family()
    values = torch.randn(7, 4)
    c_fc, c_proj = module.weights(10)
    expected = F.linear(
        F.gelu(F.linear(values, c_fc) * module.pre_gain[10]), c_proj
    )
    torch.testing.assert_close(module.forward_layer(10, values), expected)


def test_coefficients_only_freezes_banks_and_gains() -> None:
    module = family()
    selected = module.set_trainable(coefficients_only=True)
    assert selected == [module.alpha_fc, module.alpha_proj]
    assert not module.basis_fc.requires_grad
    assert not module.pre_gain.requires_grad


class FakeMLP(nn.Module):
    def __init__(self, width: int, hidden: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.c_fc = nn.Linear(width, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, width, bias=False)
        with torch.no_grad():
            self.c_fc.weight.copy_(torch.randn(hidden, width, generator=generator))
            self.c_proj.weight.copy_(torch.randn(width, hidden, generator=generator))
        self.pregelu_gain = nn.Parameter(torch.randn(hidden, generator=generator))
        self.residual_output_log_gain = nn.Parameter(torch.randn(width, generator=generator))
        self.residual_output_gain_scale = 1.0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(F.linear(values, self.c_fc.weight) * self.pregelu_gain)
        output = F.linear(hidden, self.c_proj.weight)
        return output * self.residual_output_log_gain.exp()


def test_initial_family_exactly_preserves_all_seven_trunks() -> None:
    blocks = []
    roots: dict[int, FakeMLP] = {}
    for layer, atom in enumerate(ASSIGNMENT):
        if atom not in roots:
            roots[atom] = FakeMLP(4, 6, seed=atom + 101)
        source = roots[atom]
        cloned = source if atom < 5 else FakeMLP(4, 6, seed=atom + 101)
        if atom >= 5:
            with torch.no_grad():
                cloned.c_fc.weight.copy_(source.c_fc.weight)
                cloned.c_proj.weight.copy_(source.c_proj.weight)
        blocks.append(SimpleNamespace(mlp=cloned))
    checkpoint = SimpleNamespace(
        config=SimpleNamespace(
            mlp_shared_dense_trunk=True,
            mlp_shared_dense_trunk_groups=7,
            mlp_shared_dense_trunk_boundaries=BOUNDARIES,
        ),
        transformer=SimpleNamespace(h=blocks),
    )
    module, diagnostics = initial_family(checkpoint, top_k=4)
    assert diagnostics["matched_similarity_mean"] <= 1.0 + 1e-6
    values = torch.randn(9, 4)
    for layer in range(12):
        torch.testing.assert_close(
            module.forward_layer(layer, values), blocks[layer].mlp(values)
        )


def test_parameter_accounting_matches_registered_family() -> None:
    hidden, width = 3072, 768
    module = TwoBankAtomwiseMLP(
        basis_fc=torch.empty(7, hidden, width),
        basis_proj=torch.empty(7, width, hidden),
        alpha_fc=torch.empty(7, hidden),
        alpha_proj=torch.empty(7, hidden),
        pre_gain=torch.empty(12, hidden),
        output_log_gain=torch.empty(12, width),
        pairing=torch.arange(hidden),
    )
    assert sum(parameter.numel() for parameter in module.parameters()) == 33_119_232
