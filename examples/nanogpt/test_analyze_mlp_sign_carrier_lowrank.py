from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_sign_carrier_lowrank import (
    carrier_capture,
    state_accounting,
    synthetic_self_check,
)


def test_exact_h5a_accounting() -> None:
    record = state_accounting(
        rows=3072,
        columns=768,
        rank=4,
        deployment_matrix_count=24,
        maximum_fraction=0.01,
    )
    assert record["carrier_bits"] == 2_359_296
    assert record["factor_scalars_total"] == 368_640
    assert record["residual_coordinates_per_matrix"] == 2_088
    assert 0.00999 < float(record["total_checkpoint_byte_fraction"]) <= 0.01


def test_sign_masked_lowrank_family_reconstructs_own_member() -> None:
    assert synthetic_self_check("cpu") > 0.999


def test_randomized_capture_rejects_wrong_sign_lowrank() -> None:
    torch.manual_seed(11)
    rows, columns, rank = 48, 32, 3
    left = torch.randn(4, rows, rank)
    right = torch.randn(4, columns, rank)
    true_sign = torch.where(torch.randn(rows * columns) >= 0, 1.0, -1.0)
    targets = true_sign.reshape(rows, columns) * torch.bmm(
        left, right.transpose(1, 2)
    )
    wrong_sign = torch.ones_like(true_sign)
    support = torch.empty(0, dtype=torch.long)
    good = carrier_capture(
        targets.flatten(1),
        carrier=true_sign,
        support=support,
        matrix_rows=rows,
        matrix_columns=columns,
        rank=rank,
        seed=3,
        batch_size=4,
    )
    bad = carrier_capture(
        targets.flatten(1),
        carrier=wrong_sign,
        support=support,
        matrix_rows=rows,
        matrix_columns=columns,
        rank=rank,
        seed=3,
        batch_size=4,
    )
    assert float(good.min()) > 0.999
    assert float(good.mean() - bad.mean()) > 0.40
