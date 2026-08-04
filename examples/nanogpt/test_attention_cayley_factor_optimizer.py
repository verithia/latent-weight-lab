from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_attention_cayley_factor_optimizer import (
    apply_coordinate_direction,
    chart_gradients,
    effective_weight,
    make_charts,
)


class AttentionCayleyFactorOptimizerTest(unittest.TestCase):
    def test_identity_and_optimizer_directions_are_finite(self) -> None:
        torch.manual_seed(7)
        weight = torch.randn(8, 6, dtype=torch.float64)
        gradient = torch.randn_like(weight)
        input_chart, output_chart = make_charts(
            weight=weight,
            rank=2,
            base_seed=19,
            layer=0,
            target="qk_shared",
        )
        torch.testing.assert_close(
            effective_weight(weight, input_chart, output_chart),
            weight,
            rtol=0.0,
            atol=0.0,
        )
        gradients = chart_gradients(
            weight=weight,
            task_gradient=gradient,
            rank=2,
            base_seed=19,
            layer=0,
            target="qk_shared",
        )
        self.assertTrue(any(value.norm() > 0 for _, value in gradients))
        for kind in ("adamw", "muon"):
            direction = apply_coordinate_direction(
                weight=weight,
                rank=2,
                base_seed=19,
                layer=0,
                target="qk_shared",
                gradients=gradients,
                optimizer_kind=kind,
                epsilon=1e-5,
                ns_steps=5,
            )
            self.assertEqual(direction.shape, weight.shape)
            self.assertTrue(torch.isfinite(direction).all())
            self.assertGreater(float(direction.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
