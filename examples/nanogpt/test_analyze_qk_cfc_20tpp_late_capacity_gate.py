import torch

from examples.nanogpt.analyze_qk_cfc_20tpp_late_capacity_gate import (
    classify,
    cosine_lr,
    deterministic_random_sources,
)


def test_random_sources_are_deterministic_unique_and_no_self() -> None:
    first = deterministic_random_sources(width=12, total=5, members=2, seed=17)
    second = deterministic_random_sources(width=12, total=5, members=2, seed=17)
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 5, 12)
    for member in range(2):
        for target in range(12):
            values = first[member, :, target].tolist()
            assert target not in values
            assert len(set(values)) == 5


def test_cosine_lr_matches_boundaries() -> None:
    assert cosine_lr(
        0,
        learning_rate=1.0,
        min_lr=0.1,
        warmup_iters=4,
        lr_decay_iters=10,
    ) == 0.2
    assert cosine_lr(
        10,
        learning_rate=1.0,
        min_lr=0.1,
        warmup_iters=4,
        lr_decay_iters=10,
    ) == 0.1


def _phase(recovery_current: float, recovery_wide: float, recovery_random: float):
    return {
        window: {
            "current_132": {"late_positive_line_recovery": recovery_current},
            "late_wide_176": {"late_positive_line_recovery": recovery_wide},
            "late_random_176": {"late_positive_line_recovery": recovery_random},
        }
        for window in ("fit", "holdout")
    }


def _functional(mean: float, upper: float):
    return {
        "late_wide_176": {
            "vs_current": {
                "candidate_minus_current_mean_ce": mean,
                "upper_confidence_bound": upper,
            }
        }
    }


def test_classify_promotes_only_complete_causal_pass() -> None:
    rows = {
        "4746": _phase(0.60, 0.59, 0.40),
        "7119": _phase(0.55, 0.62, 0.56),
        "9489": _phase(0.50, 0.58, 0.52),
    }
    functional = {
        "7119": _functional(-0.0007, 0.0002),
        "9489": _functional(-0.0003, 0.0004),
    }
    rule = {
        "failure_steps": [7119, 9489],
        "preservation_step": 4746,
        "minimum_failure_recovery_improvement": 0.05,
        "minimum_holdout_margin_over_random": 0.03,
        "maximum_preservation_recovery_regression": 0.02,
        "maximum_functional_upper_bound_regression_ce": 0.001,
        "minimum_one_phase_mean_ce_improvement": 0.0005,
    }
    assert classify(rows, functional, rule)["passes"] is True
    rows["9489"]["holdout"]["late_wide_176"][
        "late_positive_line_recovery"
    ] = 0.53
    assert classify(rows, functional, rule)["passes"] is False
