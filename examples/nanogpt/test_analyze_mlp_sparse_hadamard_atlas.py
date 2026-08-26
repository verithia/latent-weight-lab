from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_mlp_sparse_hadamard_atlas import (
    balanced_joint_support,
    grouped_hadamard_2d,
    support_capture,
    support_jaccard,
    top_support,
)


class SparseHadamardAtlasTest(unittest.TestCase):
    def test_grouped_hadamard_is_involutory_and_preserves_energy(self) -> None:
        torch.manual_seed(7)
        matrix = torch.randn(12, 20, dtype=torch.float64)
        transformed = grouped_hadamard_2d(matrix, row_group=4, column_group=4)
        recovered = grouped_hadamard_2d(transformed, row_group=4, column_group=4)
        self.assertTrue(torch.allclose(recovered, matrix, atol=1e-10, rtol=1e-10))
        self.assertAlmostEqual(float(transformed.square().sum()), float(matrix.square().sum()))

    def test_top_support_and_jaccard(self) -> None:
        square = torch.tensor([[1.0, 9.0], [4.0, 16.0]])
        support = top_support(square, 2)
        self.assertAlmostEqual(support_capture(square, support), 25.0 / 30.0)
        other = torch.tensor([0, 3])
        self.assertAlmostEqual(support_jaccard(support, other, 4), 1.0 / 3.0)

    def test_balanced_joint_support_keeps_one_coordinate_for_each_field(self) -> None:
        state = torch.tensor([9.0, 0.0, 0.0, 0.0])
        gradient = torch.tensor([0.0, 0.0, 0.0, 16.0])
        support = balanced_joint_support(state, gradient, 2)
        self.assertEqual(set(support.tolist()), {0, 3})


if __name__ == "__main__":
    unittest.main()
