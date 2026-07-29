import unittest

from examples.nanogpt.analyze_mlp_muon_chart_staleness import (
    aggregate_retention,
)


def synthetic_inputs(retention_by_age):
    rows = []
    finite_rows = []
    for age, retention in retention_by_age.items():
        for window in ("fit", "holdout"):
            for candidate, scale in (
                ("fresh", 1.0),
                ("aged", retention),
            ):
                rows.append(
                    {
                        "phase_anchor": 0,
                        "endpoint": age,
                        "age_updates": age,
                        "layer": 0,
                        "window": window,
                        "candidate": candidate,
                        "output_fixed_scale_recovery": 0.4 * scale,
                        "output_positive_line_recovery": 0.5 * scale,
                        "target_output_energy": 2.0,
                        "train_gradient_predicted_ce_decrease": (
                            0.01 * scale
                        ),
                        "validation_gradient_predicted_ce_decrease": (
                            0.02 * scale
                        ),
                    }
                )
            for candidate, loss in (
                ("baseline", 5.0),
                ("fresh", 4.99),
                ("aged", 4.99 + 0.001 * (1.0 - retention)),
            ):
                finite_rows.append(
                    {
                        "phase_anchor": 0,
                        "endpoint": age,
                        "age_updates": age,
                        "window": window,
                        "candidate": candidate,
                        "loss": loss,
                    }
                )
    return rows, finite_rows


class AggregateRetentionTest(unittest.TestCase):
    def decide(self, retentions):
        rows, finite_rows = synthetic_inputs(retentions)
        return aggregate_retention(
            rows,
            finite_rows,
            stable_retention=0.90,
            stale_retention=0.80,
        )

    def test_keeps_r60_when_age60_is_stable(self):
        result = self.decide(
            {0: 1.0, 15: 0.98, 30: 0.96, 45: 0.94, 60: 0.91}
        )
        self.assertEqual(result["decision"], "R60_CONNECTIVITY_NOT_STALE")

    def test_qualifies_r30_when_decay_occurs_after_30(self):
        result = self.decide(
            {0: 1.0, 15: 0.97, 30: 0.92, 45: 0.76, 60: 0.70}
        )
        self.assertEqual(
            result["decision"], "QUALIFY_R30_PERFORMANCE_PREFLIGHT"
        )

    def test_qualifies_r15_when_age30_is_stale(self):
        result = self.decide(
            {0: 1.0, 15: 0.94, 30: 0.72, 45: 0.65, 60: 0.60}
        )
        self.assertEqual(
            result["decision"], "QUALIFY_R15_PERFORMANCE_PREFLIGHT"
        )

    def test_mixed_signal_does_not_promote(self):
        result = self.decide(
            {0: 1.0, 15: 0.86, 30: 0.84, 45: 0.82, 60: 0.81}
        )
        self.assertEqual(
            result["decision"], "MIXED_CADENCE_SIGNAL_NO_TRAINING_PROMOTION"
        )

    def test_finite_step_summary_is_descriptive(self):
        result = self.decide(
            {0: 1.0, 15: 0.97, 30: 0.92, 45: 0.76, 60: 0.70}
        )
        summary = result["by_age_updates"]["30"]["finite_step"]
        self.assertEqual(summary["comparisons"], 2)
        self.assertEqual(summary["aged_better_than_fresh"], 0)
        self.assertGreater(summary["mean_aged_minus_fresh_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
