from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_boolean_partition import (
    learn_partition,
    partition_state_accounting,
    synthetic_self_check,
)


def test_exact_h4a_accounting() -> None:
    record = partition_state_accounting(
        rows=3072,
        columns=768,
        category_bits=3,
        deployment_matrix_count=24,
        maximum_fraction=0.01,
    )
    assert record["partition_bits"] == 7_077_888
    assert record["partition_bytes"] == 884_736
    assert record["gain_scalars_per_matrix"] == 3_840
    assert record["value_scalars_per_matrix"] == 8
    assert record["residual_coordinates_per_matrix"] == 1_312
    assert record["total_checkpoint_bytes"] == 1_132_416
    assert 0.00999 < float(record["total_checkpoint_byte_fraction"]) <= 0.01


def test_partition_family_reconstructs_own_member() -> None:
    assert synthetic_self_check("cpu") > 0.999


def test_partition_learning_returns_all_categories() -> None:
    generator = torch.Generator(device="cpu").manual_seed(3)
    features = torch.randn(12, 4096, generator=generator)
    assignment, history = learn_partition(
        features,
        category_count=8,
        iterations=2,
        coordinate_batch_size=1024,
    )
    assert assignment.shape == (4096,)
    assert torch.unique(assignment).numel() == 8
    assert len(history) == 2
