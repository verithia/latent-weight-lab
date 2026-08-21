from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_seven_trunk_selected_atom_unsharing import (
    PairedAtomGroupMLP,
    score_private_atoms,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_seven_trunk_selected_atom_unsharing_plan.json"


def family() -> PairedAtomGroupMLP:
    generator = torch.Generator().manual_seed(17)
    layers, width, hidden, private = 3, 4, 8, 2
    fc = torch.randn(hidden, width, generator=generator)
    proj = torch.randn(width, hidden, generator=generator)
    indices = torch.tensor([[0, 3], [2, 5], [4, 7]])
    return PairedAtomGroupMLP(
        layers=(5, 6, 7),
        c_fc_weight=fc,
        c_proj_weight=proj,
        pre_gain=torch.randn(layers, hidden, generator=generator),
        output_log_gain=torch.randn(layers, width, generator=generator) * 0.1,
        private_indices=indices,
        private_fc=torch.randn(layers, private, width, generator=generator),
        private_proj=torch.randn(layers, width, private, generator=generator),
    )


def test_forward_matches_materialized_paired_replacement() -> None:
    module = family()
    values = torch.randn(7, 4)
    offset = 1
    c_fc = module.c_fc_weight.clone()
    c_proj = module.c_proj_weight.clone()
    indices = module.private_indices[offset]
    c_fc.index_copy_(0, indices, module.private_fc[offset])
    c_proj.index_copy_(1, indices, module.private_proj[offset])
    expected = F.linear(
        F.gelu(F.linear(values, c_fc) * module.pre_gain[offset]), c_proj
    ) * module.output_log_gain[offset].exp()
    torch.testing.assert_close(module.forward_layer(offset, values), expected)


def test_default_private_copies_preserve_shared_function() -> None:
    generator = torch.Generator().manual_seed(19)
    fc = torch.randn(8, 4, generator=generator)
    proj = torch.randn(4, 8, generator=generator)
    module = PairedAtomGroupMLP(
        layers=(5, 6),
        c_fc_weight=fc,
        c_proj_weight=proj,
        pre_gain=torch.ones(2, 8),
        output_log_gain=torch.zeros(2, 4),
        private_indices=torch.tensor([[0, 2, 4], [1, 3, 5]]),
    )
    values = torch.randn(5, 4, generator=generator)
    expected = F.linear(F.gelu(F.linear(values, fc)), proj)
    for offset in range(2):
        torch.testing.assert_close(module.forward_layer(offset, values), expected)


def test_effective_weights_match_forward() -> None:
    module = family()
    values = torch.randn(5, 4)
    for offset in range(3):
        c_fc, c_proj = module.effective_weights(offset)
        expected = F.linear(F.gelu(F.linear(values, c_fc)), c_proj)
        torch.testing.assert_close(module.forward_layer(offset, values), expected)


def test_gradient_residual_selector_is_paired_and_deterministic() -> None:
    fc = torch.zeros(3, 6, 4)
    proj = torch.zeros(3, 4, 6)
    fc[0, 1] = 9.0
    proj[0, :, 4] = 8.0
    fc[1, 2] = 7.0
    proj[2, :, 5] = 6.0
    first, diagnostics = score_private_atoms(fc, proj, private_width=3)
    second, _ = score_private_atoms(fc, proj, private_width=3)
    torch.testing.assert_close(first, second)
    assert first.shape == (3, 3)
    assert all(torch.unique(row).numel() == 3 for row in first)
    assert 1 in first[0].tolist()
    assert 4 in first[0].tolist()
    assert diagnostics["score_max"] > diagnostics["score_mean"]


def test_registered_parameter_accounting() -> None:
    plan = json.loads(PLAN.read_text())
    family_plan = plan["family"]
    extra = 7 * 128 * (768 + 768)
    assert family_plan["additional_private_atom_parameters"] == extra
    total = family_plan["seven_trunk_mlp_parameters"] + extra
    assert family_plan["total_compact_mlp_parameters"] == total
    dense = family_plan["dense_mlp_parameters"]
    assert abs(family_plan["mlp_parameter_compression_ratio"] - dense / total) < 1e-12
