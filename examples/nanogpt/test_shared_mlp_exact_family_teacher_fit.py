from __future__ import annotations

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    SharedGroupMLP,
    classify,
    normalized_objective,
)


def make_module() -> SharedGroupMLP:
    return SharedGroupMLP(
        layers=(5, 6),
        c_fc_weight=torch.tensor([[1.0, -2.0], [0.5, 1.0]]),
        c_proj_weight=torch.tensor([[1.0, 0.25], [-0.5, 2.0]]),
        pre_gain=torch.tensor([[1.0, -0.5], [2.0, 0.75]]),
        output_log_gain=torch.log(torch.tensor([[1.5, 0.5], [0.75, 2.0]])),
    )


def test_group_forward_matches_explicit_layer_functions() -> None:
    module = make_module()
    values = torch.randn(3, 2, 2, 4, 2)
    output = module(values)
    for layer in range(2):
        explicit = F.linear(values[:, layer], module.c_fc_weight)
        explicit = F.gelu(explicit * module.pre_gain[layer])
        explicit = F.linear(explicit, module.c_proj_weight)
        explicit = explicit * module.output_log_gain[layer].exp()
        torch.testing.assert_close(output[:, layer], explicit)


def test_effective_weights_are_functionally_exact() -> None:
    module = make_module()
    values = torch.randn(7, 2)
    for layer in range(2):
        c_fc, c_proj = module.effective_weights(layer)
        explicit = F.linear(F.gelu(F.linear(values, c_fc)), c_proj)
        torch.testing.assert_close(module.forward_layer(layer, values), explicit)


def test_normalized_objective_and_fail_closed_classification() -> None:
    target = torch.tensor([[[[[1.0, -2.0]]]]])
    assert normalized_objective(target, target) == 0.0
    good = {
        "mean_explained_target_energy": 0.95,
        "minimum_explained_target_energy": 0.8,
    }
    gates = {
        "minimum_mean_output_recovery": 0.9,
        "minimum_worst_layer_bank_output_recovery": 0.75,
        "minimum_mean_input_jvp_recovery": 0.8,
        "minimum_worst_layer_bank_input_jvp_recovery": 0.5,
        "maximum_fixed_validation_cross_entropy_gap": 0.05,
    }
    assert classify(output=good, jvp=good, ce_gap=0.04, healthy=True, gates=gates) == "REPRESENTATIONAL_PASS"
    assert classify(output=good, jvp=good, ce_gap=0.06, healthy=True, gates=gates) == "REPRESENTATIONAL_FAIL"
    assert classify(output=good, jvp=good, ce_gap=0.0, healthy=False, gates=gates) == "OPTIMIZATION_INCONCLUSIVE"
