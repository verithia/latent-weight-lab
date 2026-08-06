from __future__ import annotations

import pytest
import torch

from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION
from examples.nanogpt.verify_late_cproj_paired_state_trajectory import (
    assert_equal_parameters,
    validate_snapshot,
)


def snapshot_payload() -> dict[str, object]:
    names = {
        f"transformer.h.{layer}.mlp.c_proj.weight": torch.ones(2, 3)
        for layer in (8, 9, 10, 11)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "step": 99,
        "targets": ["mlp.c_proj"],
        "layers": [8, 9, 10, 11],
        "storage_dtype": "float32",
        "all_parameters": False,
        "all_buffers": False,
        "parameters": names,
        "buffers": {},
        "run_identity_sha256": "a" * 64,
    }


def test_validate_targeted_structured_snapshot() -> None:
    contract = {
        "target": "mlp.c_proj",
        "layers": [8, 9, 10, 11],
        "storage_dtype": "float32",
    }
    identity, tensors = validate_snapshot(
        snapshot_payload(), expected_step=99, contract=contract
    )
    assert identity == "a" * 64
    assert len(tensors) == 4


def test_bitwise_pairing_fails_closed() -> None:
    left = {"weight": torch.tensor([1.0, 2.0])}
    right = {"weight": left["weight"].clone()}
    assert_equal_parameters(left, right)
    right["weight"][0] += 1e-3
    with pytest.raises(ValueError, match="same-run paired state mismatch"):
        assert_equal_parameters(left, right)
