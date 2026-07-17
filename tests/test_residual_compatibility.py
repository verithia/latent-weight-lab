from __future__ import annotations

import torch

from examples.nanogpt.analyze_residual_compatibility import compatibility_metrics, fixed_validation_batches


def test_compatibility_metrics_reconstructs_residual_sum() -> None:
    residual = torch.tensor([[3.0, 4.0], [4.0, 3.0]])
    update = torch.tensor([[1.0, -2.0], [-1.0, 2.0]])
    metrics = compatibility_metrics(residual, update, residual + update)

    assert metrics["residual_add_reconstruction_max_abs"] == 0.0
    assert metrics["update_to_residual_rms"] > 0.0
    assert -1.0 <= metrics["residual_update_cos_mean"] <= 1.0


def test_fixed_validation_batches_are_reproducible(tmp_path) -> None:
    np_values = torch.arange(128, dtype=torch.int64).numpy().astype("uint16")
    np_values.tofile(tmp_path / "val.bin")

    first = fixed_validation_batches(tmp_path, batch_size=2, block_size=8, batches=3, seed=7)
    second = fixed_validation_batches(tmp_path, batch_size=2, block_size=8, batches=3, seed=7)

    assert len(first) == len(second) == 3
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
