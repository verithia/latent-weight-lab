import torch

from examples.nanogpt.analyze_qk_cfc_procedural_state_codability import (
    classify,
    recovery_metrics,
    summarize,
)
from examples.nanogpt.muon_matched_givens import (
    batched_multistage_directed_sparse_update,
)


def test_recovery_metrics_exact_and_orthogonal() -> None:
    target = torch.tensor([[1.0, 0.0]])
    exact = recovery_metrics(target, target)
    assert exact["energy_recovery"] == 1.0
    assert exact["cosine"] == 1.0
    orthogonal = recovery_metrics(target, torch.tensor([[0.0, 1.0]]))
    assert orthogonal["energy_recovery"] == -1.0
    assert orthogonal["cosine"] == 0.0


def test_production_solver_can_encode_identity_target() -> None:
    source = torch.eye(4).unsqueeze(0)
    prediction, stages = batched_multistage_directed_sparse_update(
        source,
        source,
        incoming_schedule=[2, 2],
        ridge_ratio=1e-6,
        chunk_size=2,
    )
    assert len(stages) == 2
    assert recovery_metrics(source, prediction)["energy_recovery"] > 0.999


def test_summary_is_energy_weighted_and_tracks_minimum() -> None:
    rows = [
        {"layer": 0, "target_energy": 1.0, "energy_recovery": 0.2, "positive_line_recovery": 0.3},
        {"layer": 1, "target_energy": 3.0, "energy_recovery": 1.0, "positive_line_recovery": 0.9},
    ]
    result = summarize(rows, [0, 1])
    assert result["energy_weighted_recovery"] == 0.8
    assert result["minimum_layer_recovery"] == 0.2
    assert abs(result["energy_weighted_positive_line_recovery"] - 0.75) < 1e-12


def test_classification_requires_both_state_families() -> None:
    passing = {
        "all": {"energy_weighted_recovery": 0.9},
        "late": {"minimum_layer_recovery": 0.8},
    }
    failing = {
        "all": {"energy_weighted_recovery": 0.9},
        "late": {"minimum_layer_recovery": 0.6},
    }
    rule = {
        "minimum_aggregate_recovery": 0.8,
        "minimum_late_layer_recovery": 0.7,
        "threshold_changes_after_measurement": False,
    }
    rejected = classify(
        {"momentum_buffer": passing, "compression_residual": failing}, rule
    )
    assert rejected["classification"] == "CFC_PROCEDURAL_STATE_CODE_REJECTED"
    accepted = classify(
        {"momentum_buffer": passing, "compression_residual": passing}, rule
    )
    assert accepted["all_state_families_passed"] is True
