from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_mlp_muon_matched_functional_metric import (
    aggregate_results,
    fixed_scale_recovery,
    output_space_metrics,
    task_descent_metrics,
)


class FunctionalMetricTest(unittest.TestCase):
    def test_fixed_scale_recovery_identity(self) -> None:
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(fixed_scale_recovery(target, target), 1.0)
        self.assertAlmostEqual(
            fixed_scale_recovery(target, torch.zeros_like(target)),
            0.0,
        )

    def test_identity_activations_match_weight_metric(self) -> None:
        target = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        prediction = 0.75 * target
        metrics = output_space_metrics(
            torch.eye(2), target, prediction
        )
        self.assertAlmostEqual(
            metrics["fixed_scale_recovery"],
            fixed_scale_recovery(target, prediction),
        )
        self.assertAlmostEqual(metrics["cosine"], 1.0)

    def test_activation_metric_suppresses_inactive_channel(self) -> None:
        hidden = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        target = torch.eye(2)
        prediction = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        metrics = output_space_metrics(hidden, target, prediction)
        self.assertAlmostEqual(metrics["fixed_scale_recovery"], 1.0)

    def test_task_descent_sign(self) -> None:
        gradient = torch.tensor([[1.0, -2.0]])
        descent = task_descent_metrics(gradient, -gradient)
        ascent = task_descent_metrics(gradient, gradient)
        self.assertGreater(descent["predicted_ce_decrease"], 0.0)
        self.assertLess(ascent["predicted_ce_decrease"], 0.0)

    def test_aggregate_functional_gain_decision(self) -> None:
        rows = []
        for window in ("fit", "holdout"):
            for candidate, recovery, decrease in (
                ("dense_exact", 1.0, 10.0),
                ("stage32", 0.20, 4.0),
                ("stage64", 0.30, 5.0),
                ("incremental64", 0.05, 1.0),
            ):
                rows.append(
                    {
                        "candidate": candidate,
                        "window": window,
                        "current_output_fixed_scale_recovery": recovery,
                        "current_output_positive_line_recovery": recovery,
                        "future_output_positive_line_recovery": recovery,
                        "current_target_output_energy": 1.0,
                        "future_target_output_energy": 1.0,
                        "train_gradient_predicted_ce_decrease": decrease,
                        "validation_gradient_predicted_ce_decrease": decrease,
                        "update_fro": 1.0,
                    }
                )
        finite = []
        for phase in (0, 60, 120, 180):
            for window in ("fit", "holdout"):
                finite.extend(
                    [
                        {
                            "phase_start": phase,
                            "window": window,
                            "candidate": "stage32",
                            "loss": 2.0,
                        },
                        {
                            "phase_start": phase,
                            "window": window,
                            "candidate": "stage64",
                            "loss": 1.9,
                        },
                    ]
                )
        result = aggregate_results(
            rows,
            finite,
            minimum_output_recovery_ratio=1.25,
            minimum_task_descent_ratio=1.15,
            minimum_finite_step_wins=6,
        )
        self.assertEqual(
            result["decision"],
            "FUNCTIONAL_GAIN_PRESENT_TEST_TEMPORAL_REFRESH",
        )


if __name__ == "__main__":
    unittest.main()
