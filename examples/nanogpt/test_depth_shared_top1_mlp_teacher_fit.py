from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_depth_shared_top1_mlp_teacher_fit import (
    DepthSharedTop1MLP,
    InstalledTop1MLP,
    objective_parts,
)


PLAN = Path(__file__).parent / "configs" / "selection_artifacts" / (
    "124m_depth_shared_top1_mlp_teacher_fit_plan.json"
)


def family() -> DepthSharedTop1MLP:
    generator = torch.Generator().manual_seed(3)
    return DepthSharedTop1MLP(
        expert_fc=torch.randn(3, 7, 4, generator=generator),
        expert_proj=torch.randn(3, 4, 7, generator=generator),
        pre_gain=torch.randn(2, 7, generator=generator),
        output_log_gain=torch.randn(2, 4, generator=generator) * 0.01,
        router_weight=torch.randn(2, 3, 4, generator=generator),
        router_bias=torch.randn(2, 3, generator=generator),
    )


def test_hard_route_matches_selected_complete_expert() -> None:
    module = family()
    values = torch.randn(5, 4)
    layer = 1
    with torch.no_grad():
        logits = module._logits(values, layer)
        assignment = logits.argmax(dim=-1)
        expected = torch.empty_like(values)
        for expert in range(module.experts):
            selected = assignment == expert
            if bool(selected.any()):
                hidden = F.gelu(
                    F.linear(values[selected], module.expert_fc[expert])
                    * module.pre_gain[layer]
                )
                expected[selected] = F.linear(
                    hidden, module.expert_proj[expert]
                )
        expected *= module.output_log_gain[layer].exp()
        torch.testing.assert_close(module.forward_layer(layer, values), expected)


def test_straight_through_forward_is_exactly_hard() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
    hard = DepthSharedTop1MLP._route_weights(
        logits, mode="hard_top1", temperature=0.25
    )
    straight = DepthSharedTop1MLP._route_weights(
        logits, mode="hard_st", temperature=0.25
    )
    torch.testing.assert_close(straight, hard)
    straight.sum().backward()
    assert logits.grad is not None


def test_direction_objective_is_scale_normalized() -> None:
    target = torch.zeros(3, 1, 1, 2)
    target[1, ..., 0] = 0.01
    target[2, ..., 0] = -0.01
    prediction = target.clone()
    prediction[1, ..., 0] = 0.0
    prediction[2, ..., 0] = 0.0
    parts = objective_parts(prediction, target)
    assert float(parts["value"]) == 0.0
    torch.testing.assert_close(parts["direction"], torch.tensor(1.0))


def test_installed_view_preserves_plain_block_interface() -> None:
    installed = InstalledTop1MLP(family(), 1)
    assert installed.residual_conditioned_output_slope is None
    assert installed.conditioned_output_gate_source == "residual"


def test_registered_parameter_accounting() -> None:
    plan = json.loads(PLAN.read_text())
    family_plan = plan["family"]
    expected = (
        4 * 2 * 768 * 3072
        + 12 * 3072
        + 12 * 768
        + 12 * 4 * (768 + 1)
    )
    assert family_plan["compact_parameters"] == expected
    assert abs(
        family_plan["compression_ratio"]
        - family_plan["dense_mlp_parameters"] / expected
    ) < 1e-12
