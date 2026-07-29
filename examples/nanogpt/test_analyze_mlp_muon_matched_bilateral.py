from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_muon_matched_bilateral import (
    aggregate_rows,
    causal_bilateral_update,
)
from examples.nanogpt.muon_matched_givens import apply_givens_flow


def test_causal_bilateral_recovers_small_two_sided_update() -> None:
    torch.manual_seed(37)
    source = torch.randn(4, 8)
    right_permutations = torch.arange(8).view(1, 8)
    left_permutations = torch.arange(4).view(1, 4)
    right_angles = torch.zeros(1, 4)
    left_angles = torch.zeros(1, 2)
    right_angles[0, 0] = 1e-4
    left_angles[0, 0] = -1e-4
    after_right = apply_givens_flow(
        source, right_angles, right_permutations
    )
    target = apply_givens_flow(
        after_right.T, left_angles, left_permutations
    ).T
    requested = target - source
    predicted, record, final = causal_bilateral_update(
        source,
        requested,
        right_permutations=right_permutations,
        left_permutations=left_permutations,
        left_stages=1,
        left_neighbors=2,
        left_seed=0,
    )
    assert torch.allclose(final - source, predicted)
    # The hidden/output tangent columns overlap, so one causal diagonal
    # Gauss-Seidel sweep is intentionally approximate rather than a joint
    # normal-equation solve.
    assert float(record["requested_update_recovery"]) > 0.9
    assert float(record["left_incremental_recovery"]) > 0.0
    assert int(record["coordinates"]) == 6


def test_aggregate_requires_bilateral_enrichment() -> None:
    rows = []
    specifications = {
        "task_hidden_32": (8, 0.1, 0.01, 0.1),
        "task_hidden_40": (10, 0.1, 0.01, 0.1),
        "task_output_32": (2, 0.02, 0.002, 0.05),
        "task_bilateral_32x32": (10, 0.12, 0.013, 0.13),
        "random_bilateral_32x32": (10, 0.02, 0.003, 0.06),
    }
    for layer in range(2):
        for chart, (
            coordinates,
            current,
            future,
            cosine,
        ) in specifications.items():
            row = {
                "chart": chart,
                "coordinates": coordinates,
                "coordinate_fraction": coordinates / 100,
                "requested_update_recovery": current,
                "future_recovery": future,
                "future_cosine": cosine,
                "future_chord_fro": 1.0 + layer,
            }
            if "bilateral" in chart:
                row["left_incremental_recovery"] = 0.02
            rows.append(row)
    aggregate, decision = aggregate_rows(
        rows,
        minimum_current_ratio=1.1,
        minimum_future_ratio=1.2,
        minimum_task_over_random=3.0,
        minimum_cell_future_cosine=0.05,
    )
    assert (
        decision
        == "PROMOTE_TASK_SELECTED_BILATERAL_TO_PRODUCTION_DESIGN"
    )
    assert (
        aggregate[
            "bilateral_over_same_coordinate_hidden_current"
        ]
        > 1.1
    )
