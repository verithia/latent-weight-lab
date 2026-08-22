from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from examples.nanogpt.dense_pair_vq_lowbit_momentum import (
    advance_lowbit_momentum,
    decode_blocks,
    encode_blocks,
)


def test_lowbit_codecs_are_deterministic_and_bounded() -> None:
    value = torch.linspace(-3.0, 3.0, 128).reshape(8, 16)
    for codec, lower, upper in (
        ("ternary2", -1, 1),
        ("symmetric_int4", -7, 7),
    ):
        first, decoded_first = encode_blocks(
            value,
            codec=codec,
            block_size=64,
            ternary_threshold_rms=0.5,
        )
        second, decoded_second = encode_blocks(
            value,
            codec=codec,
            block_size=64,
            ternary_threshold_rms=0.5,
        )
        assert first.codes.dtype == torch.int8
        assert first.scales.dtype == torch.float16
        assert int(first.codes.min()) >= lower
        assert int(first.codes.max()) <= upper
        torch.testing.assert_close(first.codes, second.codes)
        torch.testing.assert_close(first.scales, second.scales)
        torch.testing.assert_close(decoded_first, decoded_second)
        torch.testing.assert_close(decoded_first, decode_blocks(first))


def test_lowbit_residual_tier_cannot_increase_state_squared_error() -> None:
    generator = torch.Generator().manual_seed(7)
    gradient = torch.randn(16, 16, generator=generator)
    single, decoded_single = advance_lowbit_momentum(
        None,
        gradient,
        momentum=0.95,
        primary_codec="ternary2",
        residual_codec=None,
        block_size=64,
        ternary_threshold_rms=0.5,
    )
    residual, decoded_residual = advance_lowbit_momentum(
        None,
        gradient,
        momentum=0.95,
        primary_codec="ternary2",
        residual_codec="ternary2",
        block_size=64,
        ternary_threshold_rms=0.5,
    )
    assert single[1] is None
    assert residual[1] is not None
    single_error = (decoded_single - gradient).square().sum()
    residual_error = (decoded_residual - gradient).square().sum()
    assert float(residual_error) <= float(single_error)


def test_compact_state_is_carried_into_next_causal_recurrence() -> None:
    first_gradient = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    state, decoded = advance_lowbit_momentum(
        None,
        first_gradient,
        momentum=0.5,
        primary_codec="symmetric_int4",
        residual_codec="ternary2",
        block_size=64,
        ternary_threshold_rms=0.5,
    )
    second_gradient = torch.full_like(first_gradient, 0.25)
    next_state, next_decoded = advance_lowbit_momentum(
        state,
        second_gradient,
        momentum=0.5,
        primary_codec="symmetric_int4",
        residual_codec="ternary2",
        block_size=64,
        ternary_threshold_rms=0.5,
    )
    expected_target = decoded * 0.5 + second_gradient
    assert next_state[1] is not None
    assert float((next_decoded - expected_target).square().sum()) < float(
        expected_target.square().sum()
    )


def test_registered_lowbit_momentum_plan_is_immutable_and_nonintervening() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_lowbit_momentum_tangent_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["frozen_protocol"]["parameter_updates_by_candidates"] == 0
    assert plan["frozen_protocol"]["block_size"] == 64
    assert plan["decision_gate"]["automatic_endpoint"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "514b02946955ca3d1f5db0e2df38253fb237830e5f42811150d47658a51fcca7"
    )
