from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from examples.nanogpt.dense_pair_vq_optimizer_transition import (
    PairVQOptimizerTransitionOracle,
    direction_metrics,
    project_dense_momentum_to_current_codes,
    three_way_direction_metrics,
    update_compact_momentum,
)
from examples.nanogpt.muon_pair_vq import MuonPairVQ, MuonPairVQLinear


def make_module() -> MuonPairVQLinear:
    return MuonPairVQLinear(
        4,
        4,
        bias=False,
        stages=2,
        base_seed=101,
        weight_std=0.02,
        layer_id=0,
        fast_residual=False,
        error_feedback=True,
        forward_visible_feedback=True,
        feedback_codec="cartesian4x4",
        feedback_output_group_size=0,
        neighbor_candidates=16,
        code_refresh_interval=8,
    )


def test_direction_metrics_are_exact_for_identical_tensors() -> None:
    value = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    metrics = direction_metrics(value, value.clone())
    assert metrics["relative_error"] == 0.0
    assert metrics["cosine"] == 1.0
    assert metrics["positive_line_energy_recovery"] == 1.0


def test_compact_momentum_transition_matches_code_conditioned_means() -> None:
    module = make_module()
    module.codes[0].copy_(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    module.codes[1].copy_(torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]))
    compact = torch.zeros_like(module.codebooks)
    gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    expanded = update_compact_momentum(
        module, compact, gradient, momentum=0.5
    )

    pairs = gradient.reshape(-1, 2)
    expected_expanded = torch.zeros_like(pairs)
    for stage in range(module.stages):
        codes = module.codes[stage].long()
        for code in codes.unique():
            selected = codes == code
            expected = pairs[selected].mean(dim=0)
            torch.testing.assert_close(compact[stage, code], expected)
            expected_expanded[selected] += expected
    expected_expanded.div_(module.stages)
    torch.testing.assert_close(expanded, expected_expanded.reshape_as(gradient))


def test_compact_momentum_transition_matches_production_optimizer_state() -> None:
    module = make_module()
    gradient = torch.linspace(-1.0, 1.0, 16).reshape(4, 4)
    expected = torch.zeros_like(module.codebooks)
    update_compact_momentum(module, expected, gradient, momentum=0.5)
    optimizer = MuonPairVQ(
        [module], lr=1e-3, momentum=0.5, weight_decay=0.0, ns_steps=1
    )
    module.weight.grad = gradient.clone()
    optimizer.step()
    torch.testing.assert_close(
        optimizer.state[module.weight]["compact_momentum"], expected
    )


def test_current_code_projection_is_stateless_code_conditioned_mean() -> None:
    module = make_module()
    module.codes[0].copy_(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    module.codes[1].copy_(torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]))
    dense = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    projected = project_dense_momentum_to_current_codes(module, dense)
    pairs = dense.reshape(-1, 2)
    expected = torch.zeros_like(pairs)
    for stage in range(module.stages):
        codes = module.codes[stage].long()
        for code in codes.unique():
            selected = codes == code
            expected[selected] += pairs[selected].mean(dim=0)
    expected.div_(module.stages)
    torch.testing.assert_close(projected, expected.reshape_as(dense))


def test_three_way_decomposition_has_exact_cross_term_closure() -> None:
    dense = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    chart = dense + torch.tensor([[0.25, 0.0], [-0.5, 0.75]])
    compact = chart + torch.tensor([[-0.1, 0.4], [0.2, -0.3]])
    metrics = three_way_direction_metrics(dense, chart, compact)
    decomposition = metrics["decomposition"]
    assert decomposition["decomposition_closure_relative_error"] < 1e-15
    assert decomposition["instantaneous_subspace_error_energy"] > 0.0
    assert decomposition["historical_transport_error_energy"] > 0.0


def test_isolated_retraction_is_finite_and_forward_visible() -> None:
    module = make_module()
    start = module.weight.detach().float().clone()
    delta = torch.full_like(start, 1e-3)
    metrics, achieved = PairVQOptimizerTransitionOracle._retract(
        module, start, delta, refresh_codes=True
    )
    assert torch.isfinite(achieved).all()
    assert metrics["requested_delta_energy"] > 0.0
    assert metrics["initial_virtual_weight_energy_recovery"] > 0.999


def test_registered_optimizer_transition_plan_is_immutable_and_causal() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_optimizer_transition_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["frozen_protocol"]["optimizer_update_indices"] == [
        0,
        59,
        119,
        179,
        237,
    ]
    assert plan["decision_rule"]["automatic_endpoint"] is False
    assert plan["decision_rule"]["automatic_scale_up"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d38a32c22c93e0dd7354cef356e2d3c7d67e473d0a2d270f4bd5b54f9b8d3931"
    )


def test_registered_momentum_transport_plan_is_immutable_and_causal() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_momentum_transport_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["frozen_protocol"]["optimizer_update_indices"] == [
        0,
        59,
        119,
        179,
        237,
    ]
    assert plan["decision_rule"]["automatic_endpoint"] is False
    assert plan["decision_rule"]["automatic_scale_up"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d2a1e5ed88e20bfa1f0081c60a65d1cc22aa1129161b82f2b4dd988a1d19e7dc"
    )
