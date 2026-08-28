from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_phase_sign_carrier import (
    phase_state_accounting,
    routed_aggregate_capture,
    synthetic_self_check,
)


def test_exact_h12a_accounting() -> None:
    record = phase_state_accounting(
        rows=3072,
        columns=768,
        rank=2,
        carrier_count=2,
        deployment_matrix_count=24,
        maximum_fraction=0.01,
    )
    assert record["carrier_bits"] == 4_718_592
    assert record["carrier_bytes"] == 589_824
    assert record["factor_scalars_total"] == 184_320
    assert record["residual_coordinates_per_matrix"] == 3_624
    assert record["total_checkpoint_bytes"] == 1_132_416
    assert 0.00999 < float(record["total_checkpoint_byte_fraction"]) <= 0.01


def test_phase_family_reconstructs_own_members() -> None:
    result = synthetic_self_check("cpu")
    assert result["correct"] > 0.999
    assert result["correct"] - result["swapped"] > 0.30


def test_routed_capture_rejects_length_mismatch() -> None:
    rows = torch.randn(3, 24)
    routes = torch.zeros(2, dtype=torch.long)
    carriers = (torch.ones(24), torch.ones(24))
    try:
        routed_aggregate_capture(
            rows,
            routes,
            carriers=carriers,
            support=torch.empty(0, dtype=torch.long),
            matrix_rows=6,
            matrix_columns=4,
            rank=1,
            seed=1,
            batch_size=2,
        )
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("expected a length mismatch error")
