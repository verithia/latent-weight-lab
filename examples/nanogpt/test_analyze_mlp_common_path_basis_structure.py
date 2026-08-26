from __future__ import annotations

from examples.nanogpt.analyze_mlp_common_path_basis_structure import budget_frontier


def test_budget_frontier_selects_best_eligible_per_parameter() -> None:
    rows = [
        {
            "parameter": "a",
            "family": "small",
            "weighted_basis_energy_capture": 0.4,
            "total_stored_scalar_fraction_for_all_basis_vectors": 0.005,
        },
        {
            "parameter": "a",
            "family": "large",
            "weighted_basis_energy_capture": 0.9,
            "total_stored_scalar_fraction_for_all_basis_vectors": 0.02,
        },
        {
            "parameter": "b",
            "family": "best",
            "weighted_basis_energy_capture": 0.6,
            "total_stored_scalar_fraction_for_all_basis_vectors": 0.01,
        },
    ]
    eligible, best = budget_frontier(rows, 0.01)
    assert len(eligible) == 2
    assert best["a"]["family"] == "small"
    assert best["b"]["family"] == "best"
