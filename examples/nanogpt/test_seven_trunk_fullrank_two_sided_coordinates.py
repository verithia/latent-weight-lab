from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from examples.nanogpt.analyze_layer_axis_basis7_mlp_teacher_fit import (
    ASSIGNMENT,
    BOUNDARIES,
)
from examples.nanogpt.analyze_seven_trunk_fullrank_two_sided_coordinates import (
    FullRankTwoSidedMLP,
    initial_family,
)


def family() -> FullRankTwoSidedMLP:
    generator = torch.Generator().manual_seed(29)
    return FullRankTwoSidedMLP(
        singleton_fc=torch.randn(5, 6, 4, generator=generator),
        singleton_proj=torch.randn(5, 4, 6, generator=generator),
        late_fc=torch.randn(2, 6, 4, generator=generator),
        late_proj=torch.randn(2, 4, 6, generator=generator),
        input_maps=torch.randn(7, 4, 4, generator=generator),
        output_maps=torch.randn(7, 4, 4, generator=generator),
        singleton_pre_gain=torch.ones(5, 6),
        late_pre_gain=torch.ones(7, 6),
        singleton_output_log_gain=torch.zeros(5, 4),
        late_output_log_gain=torch.zeros(7, 4),
    )


def test_late_weights_match_explicit_two_sided_coordinates() -> None:
    module = family()
    layer = 10
    offset, group = layer - 5, 1
    c_fc, c_proj = module.weights(layer)
    torch.testing.assert_close(
        c_fc, module.late_fc[group] @ module.input_maps[offset]
    )
    torch.testing.assert_close(
        c_proj, module.output_maps[offset] @ module.late_proj[group]
    )


def test_forward_layer_matches_unfolded_coordinate_maps() -> None:
    module = family()
    layer, values = 6, torch.randn(7, 4)
    offset, group = layer - 5, 0
    transformed = F.linear(values, module.input_maps[offset])
    hidden = F.gelu(
        F.linear(transformed, module.late_fc[group])
        * module.late_pre_gain[offset]
    )
    branch = F.linear(hidden, module.late_proj[group])
    expected = F.linear(branch, module.output_maps[offset])
    torch.testing.assert_close(module.forward_layer(layer, values), expected)


def test_maps_only_and_joint_fit_keep_singletons_frozen() -> None:
    module = family()
    selected = module.set_trainable(coefficients_only=True)
    assert selected == [module.input_maps, module.output_maps]
    assert not module.singleton_fc.requires_grad
    assert not module.late_fc.requires_grad
    selected = module.set_trainable(coefficients_only=False)
    assert module.input_maps in selected and module.output_maps in selected
    assert module.late_fc in selected and module.late_proj in selected
    assert module.late_pre_gain in selected
    assert not module.singleton_fc.requires_grad
    assert not module.singleton_pre_gain.requires_grad


def test_maps_only_early_minibatch_has_zero_defined_gradient() -> None:
    module = family()
    module.set_trainable(coefficients_only=True)
    module.forward_layer(2, torch.randn(3, 4)).square().mean().backward()
    assert module.input_maps.grad is not None
    assert module.output_maps.grad is not None
    assert torch.count_nonzero(module.input_maps.grad) == 0
    assert torch.count_nonzero(module.output_maps.grad) == 0


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


def test_initial_family_exactly_preserves_seven_trunk_endpoint() -> None:
    roots = {atom: FakeMLP(4, 6, atom + 131) for atom in range(7)}
    blocks = []
    for layer, atom in enumerate(ASSIGNMENT):
        source = roots[atom]
        clone = FakeMLP(4, 6, atom + 131)
        with torch.no_grad():
            clone.c_fc.weight.copy_(source.c_fc.weight)
            clone.c_proj.weight.copy_(source.c_proj.weight)
        blocks.append(SimpleNamespace(mlp=clone))
    checkpoint = SimpleNamespace(
        config=SimpleNamespace(
            mlp_shared_dense_trunk=True,
            mlp_shared_dense_trunk_groups=7,
            mlp_shared_dense_trunk_boundaries=BOUNDARIES,
        ),
        transformer=SimpleNamespace(h=blocks),
    )
    module = initial_family(checkpoint)
    values = torch.randn(9, 4)
    for layer in range(12):
        torch.testing.assert_close(
            module.forward_layer(layer, values), blocks[layer].mlp(values)
        )


def test_parameter_accounting_matches_registered_ceiling() -> None:
    hidden, width = 3072, 768
    module = FullRankTwoSidedMLP(
        singleton_fc=torch.empty(5, hidden, width),
        singleton_proj=torch.empty(5, width, hidden),
        late_fc=torch.empty(2, hidden, width),
        late_proj=torch.empty(2, width, hidden),
        input_maps=torch.empty(7, width, width),
        output_maps=torch.empty(7, width, width),
        singleton_pre_gain=torch.empty(5, hidden),
        late_pre_gain=torch.empty(7, hidden),
        singleton_output_log_gain=torch.empty(5, width),
        late_output_log_gain=torch.empty(7, width),
    )
    assert sum(parameter.numel() for parameter in module.parameters()) == 41_333_760
