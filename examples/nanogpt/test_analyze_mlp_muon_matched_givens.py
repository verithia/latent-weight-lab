from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    aggregate_rows,
    diagonal_metric_causal_givens_update,
    fit_causal_givens_update,
    muon_matched_permutations,
)


def test_muon_matching_selects_strong_identity_angle_edges() -> None:
    weight = torch.eye(8)
    direction = torch.zeros_like(weight)
    direction[0, 1] = 10.0
    direction[2, 3] = 9.0
    permutations, diagnostics = muon_matched_permutations(
        weight,
        direction,
        stages=1,
        neighbors=3,
        seed=7,
    )
    pairs = {
        tuple(sorted(pair))
        for pair in permutations[0].view(-1, 2).tolist()
    }
    assert (0, 1) in pairs
    assert (2, 3) in pairs
    assert len(pairs) == 4
    assert diagnostics[0]["candidate_edge_fraction"] >= 0.5


def test_causal_fit_recovers_an_in_chart_small_update() -> None:
    torch.manual_seed(11)
    source = torch.randn(5, 8)
    permutation = torch.arange(8).view(1, 8)
    angle = 0.01
    requested = torch.zeros_like(source)
    requested[:, 0] = -angle * source[:, 1]
    requested[:, 1] = angle * source[:, 0]
    predicted, metrics = fit_causal_givens_update(
        source,
        requested,
        stages=1,
        seed=3,
        steps=400,
        learning_rate=0.001,
        permutations=permutation,
    )
    cosine = torch.nn.functional.cosine_similarity(
        requested.flatten(), predicted.flatten(), dim=0
    )
    assert float(cosine) > 0.999
    assert float(metrics["requested_update_recovery"]) > 0.998


def test_diagonal_metric_recovers_an_in_chart_small_update() -> None:
    torch.manual_seed(13)
    source = torch.randn(5, 8)
    permutation = torch.arange(8).view(1, 8)
    angle = 0.001
    requested = torch.zeros_like(source)
    requested[:, 0] = -angle * source[:, 1]
    requested[:, 1] = angle * source[:, 0]
    predicted, metrics = diagonal_metric_causal_givens_update(
        source,
        requested,
        stages=1,
        seed=3,
        permutations=permutation,
    )
    cosine = torch.nn.functional.cosine_similarity(
        requested.flatten(), predicted.flatten(), dim=0
    )
    assert float(cosine) > 0.999999
    assert float(metrics["requested_update_recovery"]) > 0.99999


def test_promotion_rule_requires_task_enrichment_and_all_positive() -> None:
    rows = []
    for connectivity, update, future, cosine in (
        ("task_matched", 0.05, 0.03, 0.2),
        ("random", 0.006, 0.01, 0.1),
    ):
        for cell in range(4):
            rows.append(
                {
                    "stages": 8,
                    "connectivity": connectivity,
                    "future_chord_fro": 1.0,
                    "coordinates": 8,
                    "coordinate_fraction": 0.01,
                    "requested_update_recovery": update,
                    "future_recovery": future,
                    "future_cosine": cosine,
                }
            )
    aggregate, decision = aggregate_rows(rows, [8])
    assert decision == "PROMOTE_MUON_MATCHED_GIVENS_TO_ALL_CELL_ORACLE"
    assert aggregate[0]["task_over_random_future_recovery"] == 3.0
