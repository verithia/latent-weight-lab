from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_write_subspace_ceiling import (
    aggregate_covariances,
    minimum_rank,
    project_actions,
    spectral_curve,
    subspace_overlap,
    topology_groups,
    validate_plan,
)


def test_topology_groups_cover_every_node_once() -> None:
    layers, experts = [0, 5, 11], 8
    expected = {(layer, expert) for layer in layers for expert in range(experts)}
    for topology, count in (
        ("global_shared_rank619", 1),
        ("layer_shared_rank60", 3),
        ("expert_local_rank7", 24),
    ):
        groups = topology_groups(topology, layers, experts)
        assert len(groups) == count
        assert {node for group in groups for node in group} == expected


def test_covariance_aggregation_matches_topology() -> None:
    node = torch.arange(2 * 3 * 4 * 4, dtype=torch.float64).reshape(2, 3, 4, 4)
    global_covariance = aggregate_covariances(node, "global_shared_rank619")
    layer_covariance = aggregate_covariances(node, "layer_shared_rank60")
    expert_covariance = aggregate_covariances(node, "expert_local_rank7")
    torch.testing.assert_close(global_covariance[0], node.sum((0, 1)))
    torch.testing.assert_close(layer_covariance, node.sum(1))
    torch.testing.assert_close(expert_covariance, node.flatten(0, 1))


def test_projection_is_exact_on_registered_span() -> None:
    torch.manual_seed(7)
    basis, _ = torch.linalg.qr(torch.randn(6, 3))
    coefficients = torch.randn(2, 5, 3)
    actions = coefficients @ basis.T
    projected = project_actions(actions, basis[None])
    torch.testing.assert_close(projected, actions, rtol=1e-5, atol=1e-6)


def test_expert_local_projection_uses_private_bases() -> None:
    basis = torch.zeros(2, 4, 1)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 0] = 1.0
    actions = torch.tensor([[[2.0, 3.0, 4.0, 5.0]], [[7.0, 8.0, 9.0, 10.0]]])
    projected = project_actions(actions, basis)
    torch.testing.assert_close(
        projected,
        torch.tensor([[[2.0, 0.0, 0.0, 0.0]], [[0.0, 8.0, 0.0, 0.0]]]),
    )


def test_spectral_curve_reports_known_rank() -> None:
    covariance = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))[None]
    basis = torch.eye(4)[:, [0, 1, 2, 3]][None]
    curve = spectral_curve(basis, covariance, covariance, registered_rank=2)
    assert curve["registered_output_recovery_energy_weighted"] == 0.7
    assert curve["minimum_rank_output_mean_0p8"] == 3
    assert curve["minimum_rank_jvp_mean_0p6"] == 2


def test_subspace_overlap_and_minimum_rank() -> None:
    basis = torch.eye(5)[:, :2]
    assert subspace_overlap(basis, basis) == 1.0
    assert minimum_rank(torch.tensor([0.2, 0.7, 0.9]), 0.8) == 3
    assert minimum_rank(torch.tensor([0.2, 0.7]), 0.8) is None


def test_preregistered_plan_is_hash_sealed() -> None:
    plan_path = (
        Path(__file__).parent / "configs" / "selection_artifacts"
        / "124m_sparse_moe_write_subspace_ceiling_plan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, plan_path)
    assert plan["identity"]["theory_preregistration_git_commit"] == (
        "2b6504fd7423aee27ff9b86b9e0574fbb6dd5c04"
    )
