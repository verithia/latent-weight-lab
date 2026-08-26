import unittest

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import (
    chronological_splits,
    energy_capture,
    gram_spectrum,
    phase_mean_rows,
    spectrum_record,
)


class HighCadenceBasisTest(unittest.TestCase):
    def test_centering_removes_constant_bias(self) -> None:
        rows = torch.tensor(
            [[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]], dtype=torch.float32
        )
        centered = gram_spectrum(rows, centered=True)
        raw = gram_spectrum(rows, centered=False)
        self.assertGreater(float(raw[0]), float(centered[0]))
        self.assertLess(float(centered[1]), 1e-8)

    def test_spectrum_dimensions_and_capture(self) -> None:
        rows = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        record = spectrum_record(rows, centered=False)
        self.assertEqual(record["dimension_99pct"], 1)
        basis = torch.tensor([[1.0], [0.0], [0.0]])
        self.assertAlmostEqual(energy_capture(rows, basis), 1.0, places=6)

    def test_chronological_split_and_mean_shift(self) -> None:
        splits = chronological_splits(
            list(range(9)), discovery_stop=3, validation_stop=6
        )
        self.assertEqual([len(splits[key]) for key in splits], [3, 3, 3])
        rows = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 6)
        means = phase_mean_rows(rows, splits)
        self.assertAlmostEqual(means[0]["mean_cosine"], 0.0, places=6)
        self.assertAlmostEqual(means[1]["mean_cosine"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
