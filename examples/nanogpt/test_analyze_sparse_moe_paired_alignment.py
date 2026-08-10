import torch

from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    _router_counts,
    aggregate,
    paired_alignment_metrics,
)


def test_paired_alignment_is_stable_under_exact_hidden_permutation() -> None:
    generator = torch.Generator().manual_seed(31)
    left_fc = torch.randn(7, 5, generator=generator)
    left_proj = torch.randn(5, 7, generator=generator)
    fit = torch.randn(128, 5, generator=generator)
    evaluate = torch.randn(128, 5, generator=generator)
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    right_fc = left_fc.index_select(0, permutation)
    right_proj = left_proj.index_select(1, permutation)
    metrics = paired_alignment_metrics(
        left_fc,
        left_proj,
        right_fc,
        right_proj,
        fit,
        evaluate,
    )
    assert metrics["assignment_overlap"] == 1.0
    assert metrics["eval_assignment_regret"] < 1e-6
    assert metrics["quotient_to_raw_chord_ratio"] < 1e-12
    assert metrics["expert_output_chord_rms"] < 1e-6


def test_router_counts_match_topk_assignment_total() -> None:
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    router = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    counts = _router_counts(router, values, top_k=2)
    assert int(counts.sum()) == 6
    assert counts.shape == (3,)


def test_aggregate_freezes_occupancy_and_minimum_overlap() -> None:
    base = {
        "fit_identity_fraction": 1.0,
        "eval_identity_fraction": 1.0,
        "fit_mean_similarity": 0.9,
        "eval_mean_similarity_under_fit_assignment": 0.8,
        "eval_oracle_mean_similarity": 0.9,
        "eval_assignment_regret": 0.1,
        "quotient_to_raw_chord_ratio": 0.5,
        "paired_log_norm_correlation": -0.5,
        "expert_output_chord_rms": 0.2,
        "fit_left_count": 300,
        "fit_right_count": 300,
        "eval_left_count": 300,
        "eval_right_count": 300,
    }
    rows = [
        {**base, "assignment_overlap": 0.9},
        {**base, "assignment_overlap": 0.7, "eval_right_count": 200},
    ]
    summary = aggregate(rows, occupancy_minimum=256)
    assert summary["assignment_overlap"]["minimum"] == 0.7
    assert summary["underoccupied_rows"] == 1
