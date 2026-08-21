from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_shared_trunk_private_ridge_teacher_fit import (
    SharedPrivateRidgeMLP,
    expand_private_width,
    passes,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_shared_trunk_private_ridge_teacher_fit_plan.json"


def family(private_width: int = 0) -> SharedPrivateRidgeMLP:
    generator = torch.Generator().manual_seed(7)
    layers, width, hidden = 3, 4, 8
    return SharedPrivateRidgeMLP(
        shared_fc=torch.randn(hidden, width, generator=generator),
        shared_proj=torch.randn(width, hidden, generator=generator),
        pre_gain=torch.randn(layers, hidden, generator=generator),
        post_gain=torch.randn(layers, hidden, generator=generator),
        output_log_gain=torch.randn(layers, width, generator=generator) * 0.1,
        private_u=torch.randn(layers, private_width, width, generator=generator),
        private_bias=torch.randn(layers, private_width, generator=generator),
        private_v=torch.randn(layers, width, private_width, generator=generator),
    )


def test_forward_matches_manual_layer_function() -> None:
    module = family(2)
    values = torch.randn(5, 4)
    layer = 1
    shared_hidden = F.gelu(
        F.linear(values, module.shared_fc) * module.pre_gain[layer]
    )
    expected = F.linear(
        shared_hidden * module.post_gain[layer], module.shared_proj
    )
    expected = expected + F.linear(
        F.gelu(
            F.linear(
                values,
                module.private_u[layer],
                module.private_bias[layer],
            )
        ),
        module.private_v[layer],
    )
    expected = expected * module.output_log_gain[layer].exp()
    torch.testing.assert_close(module.forward_layer(layer, values), expected)


def test_nested_expansion_is_function_preserving() -> None:
    module = family(0)
    values = torch.randn(2, module.layers, 2, 7, 4)
    before = module(values)
    expanded = expand_private_width(module, 5, seed=11)
    torch.testing.assert_close(expanded(values), before)
    assert expanded.private_width == 5
    assert torch.count_nonzero(expanded.private_v) == 0


def test_registered_parameter_accounting() -> None:
    plan = json.loads(PLAN.read_text())
    dense = int(plan["family"]["dense_mlp_parameters"])
    shared = 2 * 768 * 3072 + 12 * (3072 + 3072 + 768)
    for width in plan["family"]["nested_private_widths"]:
        expected = shared + 12 * int(width) * (768 + 1 + 768)
        row = plan["family"]["accounting"][str(width)]
        assert row["compact_parameters"] == expected
        assert abs(row["compression_ratio"] - dense / expected) < 1e-12


def test_pass_requires_every_registered_gate() -> None:
    gates = {
        "minimum_mean_output_recovery": 0.9,
        "minimum_worst_output_recovery": 0.75,
        "minimum_mean_input_jvp_recovery": 0.8,
        "minimum_worst_input_jvp_recovery": 0.5,
        "maximum_fixed_validation_cross_entropy_gap": 0.05,
    }
    row = {
        "optimization_healthy": True,
        "summary": {
            "output": {
                "mean_explained_target_energy": 0.91,
                "minimum_explained_target_energy": 0.76,
            },
            "input_jvp": {
                "mean_explained_target_energy": 0.81,
                "minimum_explained_target_energy": 0.51,
            },
        },
    }
    assert passes(row, 0.049, gates)
    row["summary"]["input_jvp"]["minimum_explained_target_energy"] = 0.49
    assert not passes(row, 0.049, gates)
