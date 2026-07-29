from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_mlp_fast_fresh_residual_decomposition import (
    CANDIDATES,
    aggregate_results,
    fit_left_diagonal,
    fit_right_diagonal,
    fit_two_sided_diagonal,
)


class FastFreshResidualDecompositionTest(unittest.TestCase):
    def test_right_diagonal_recovers_column_scaling(self) -> None:
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scale = torch.tensor([0.25, -0.5])
        target = source * scale.unsqueeze(0)
        update, fitted = fit_right_diagonal(source, target)
        torch.testing.assert_close(update, target)
        torch.testing.assert_close(fitted, scale)

    def test_left_diagonal_recovers_row_scaling(self) -> None:
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scale = torch.tensor([0.25, -0.5])
        target = scale.unsqueeze(1) * source
        update, fitted = fit_left_diagonal(source, target)
        torch.testing.assert_close(update, target)
        torch.testing.assert_close(fitted, scale)

    def test_two_sided_reduces_residual(self) -> None:
        source = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        target = (
            source * torch.tensor([0.2, -0.1]).unsqueeze(0)
            + torch.tensor([0.3, -0.2]).unsqueeze(1) * source
        )
        update, diagnostics = fit_two_sided_diagonal(source, target)
        self.assertLess(
            float((target - update).square().sum()),
            float(target.square().sum()),
        )
        self.assertGreater(diagnostics["right_scale_rms"], 0.0)
        self.assertGreater(diagnostics["left_scale_rms"], 0.0)

    @staticmethod
    def synthetic_rows(
        *,
        output_ratio: float,
        future_ratio: float,
        validation_ratio: float,
    ) -> tuple[list[dict], list[dict]]:
        rows = []
        finite = []
        controls = {
            "fresh_hidden64": 1.0,
            "fresh_hidden72": 1.0,
            "fresh_hidden80": 1.0,
        }
        values = {
            candidate: controls.get(candidate, 1.0)
            for candidate in CANDIDATES
        }
        values[
            "fresh_hidden64_plus_residual_output32"
        ] = output_ratio
        future_values = dict(values)
        future_values[
            "fresh_hidden64_plus_residual_output32"
        ] = future_ratio
        validation_values = dict(values)
        validation_values[
            "fresh_hidden64_plus_residual_output32"
        ] = validation_ratio
        for window in ("fit", "holdout"):
            for candidate in CANDIDATES:
                rows.append(
                    {
                        "candidate": candidate,
                        "window": window,
                        "current_weight_recovery": values[candidate],
                        "current_weight_energy": 1.0,
                        "current_residual_fixed_scale_recovery": (
                            values[candidate]
                        ),
                        "current_residual_energy": 1.0,
                        "future_residual_positive_line_recovery": (
                            future_values[candidate]
                        ),
                        "future_residual_energy": 1.0,
                        "current_output_positive_line_recovery": (
                            values[candidate]
                        ),
                        "current_output_fixed_scale_recovery": (
                            values[candidate]
                        ),
                        "current_output_energy": 1.0,
                        "train_gradient_predicted_ce_decrease": (
                            values[candidate]
                        ),
                        "validation_gradient_predicted_ce_decrease": (
                            validation_values[candidate]
                        ),
                    }
                )
        for phase in (0, 60, 120, 180):
            for window in ("fit", "holdout"):
                for candidate in CANDIDATES:
                    loss = 2.0
                    if (
                        candidate
                        == "fresh_hidden64_plus_residual_output32"
                    ):
                        loss = 1.9
                    finite.append(
                        {
                            "base_update": phase,
                            "window": window,
                            "candidate": candidate,
                            "loss": loss,
                        }
                    )
        return rows, finite

    def test_output_branch_passes_registered_rule(self) -> None:
        rows, finite = self.synthetic_rows(
            output_ratio=1.06,
            future_ratio=1.11,
            validation_ratio=1.06,
        )
        result = aggregate_results(rows, finite)
        self.assertEqual(
            result["decision"],
            "SELECT_RESIDUAL_STRUCTURE_FOR_IMPLEMENTATION_PREFLIGHT",
        )
        self.assertIn(
            "output32_vs_hidden72",
            result["branch_passes"]["output"],
        )

    def test_output_branch_rejects_tie(self) -> None:
        rows, finite = self.synthetic_rows(
            output_ratio=1.0,
            future_ratio=1.0,
            validation_ratio=1.0,
        )
        result = aggregate_results(rows, finite)
        self.assertEqual(
            result["decision"],
            "REJECT_SPARSE_ORTHOGONAL_PLUS_DIAGONAL_EXPANSION",
        )


if __name__ == "__main__":
    unittest.main()
