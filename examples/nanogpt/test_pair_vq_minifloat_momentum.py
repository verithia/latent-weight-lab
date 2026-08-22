from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from examples.nanogpt.pair_vq_minifloat_momentum import (
    PairVQMinifloatMomentumOracle,
    advance_minifloat_momentum,
    round_fp16_mantissa,
)


def _bits(values: torch.Tensor) -> list[int]:
    raw = (
        values.contiguous().view(torch.int16).to(torch.int32).reshape(-1)
        & 0xFFFF
    )
    return [int(value) for value in raw]


def test_fp16_control_is_bit_exact() -> None:
    generator = torch.Generator().manual_seed(20260822)
    values = torch.randn(4096, generator=generator)
    expected = values.to(torch.float16)
    actual = round_fp16_mantissa(values, mantissa_bits=10)
    assert actual.dtype == torch.float16
    assert _bits(actual) == _bits(expected)


def test_minifloat_rounds_ties_to_even_and_preserves_signed_zero() -> None:
    # Around 1.0, E5M2 has an ulp of 0.25.  1.125 is the exact tie between
    # 1.0 (even retained mantissa) and 1.25; 1.375 ties upward to 1.5.
    values = torch.tensor([1.125, 1.375, 0.0, -0.0], dtype=torch.float32)
    rounded = round_fp16_mantissa(values, mantissa_bits=2)
    assert rounded[:2].tolist() == [1.0, 1.5]
    assert _bits(rounded[2:]) == [0x0000, 0x8000]


def test_minifloat_saturates_instead_of_creating_infinity() -> None:
    values = torch.tensor([65504.0, -65504.0], dtype=torch.float32)
    rounded = round_fp16_mantissa(values, mantissa_bits=2)
    assert torch.isfinite(rounded).all()
    assert _bits(rounded) == [0x7B00, 0xFB00]


def test_retained_fraction_bits_are_zero_and_formats_are_nested_sets() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(2048, generator=generator)
    for mantissa_bits in (2, 4, 6, 8, 10):
        rounded = round_fp16_mantissa(values, mantissa_bits=mantissa_bits)
        drop = 10 - mantissa_bits
        mask = (1 << drop) - 1 if drop else 0
        assert all((value & mask) == 0 for value in _bits(rounded))
        # Every coarser codeword is exactly representable by every finer set.
        for finer_bits in range(mantissa_bits, 11, 2):
            rerounded = round_fp16_mantissa(
                rounded.float(), mantissa_bits=finer_bits
            )
            assert _bits(rerounded) == _bits(rounded)


def test_minifloat_recurrence_has_no_hidden_dense_state() -> None:
    generator = torch.Generator().manual_seed(11)
    first_gradient = torch.randn(32, 32, generator=generator)
    first, first_decoded = advance_minifloat_momentum(
        None, first_gradient, momentum=0.95, mantissa_bits=4
    )
    second_gradient = torch.randn(32, 32, generator=generator)
    second, second_decoded = advance_minifloat_momentum(
        first, second_gradient, momentum=0.95, mantissa_bits=4
    )
    expected = round_fp16_mantissa(
        0.95 * first_decoded + second_gradient, mantissa_bits=4
    )
    assert first.dtype == second.dtype == torch.float16
    assert _bits(second) == _bits(expected)
    torch.testing.assert_close(second_decoded, second.float(), rtol=0.0, atol=0.0)


def test_registered_plan_is_immutable_and_nonintervening() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_pair_vq_minifloat_momentum_precision_frontier_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v2")
    assert plan["frozen_protocol"]["parameter_updates_by_candidates"] == 0
    assert plan["decision_gate"]["automatic_endpoint"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "35f49aadaf4adb59981c858e370385f681eaf11800bf97796026054895a4d7eb"
    )


def test_gate_selects_smallest_passing_format() -> None:
    oracle = PairVQMinifloatMomentumOracle.__new__(
        PairVQMinifloatMomentumOracle
    )
    oracle.stage = "stage_ab_deterministic_replay"
    oracle.update_indices = {238}
    oracle.candidate_order = ["e5m2", "e5m4", "e5m10_fp16_control"]
    oracle.candidates = {
        "e5m2": {"total_bits": 8},
        "e5m4": {"total_bits": 10},
        "e5m10_fp16_control": {"total_bits": 16},
    }
    oracle.plan = {
        "decision_gate": {
            "requirements_at_every_registered_probe": {
                "minimum_all_postpolar_cosine": 0.9999,
                "minimum_every_matrix_postpolar_cosine": 0.999,
                "minimum_all_postpolar_positive_line_energy_recovery": 0.999,
                "minimum_all_prepolar_cosine": 0.9999,
                "all_metrics_finite": True,
            },
            "fp16_control_requirements": {
                "minimum_all_postpolar_cosine": 0.999999,
                "minimum_every_matrix_postpolar_cosine": 0.99999,
                "minimum_all_postpolar_positive_line_energy_recovery": 0.99999,
            },
        }
    }

    def metric(cosine: float) -> dict[str, float]:
        return {
            "cosine": cosine,
            "worst_matrix_cosine": cosine,
            "positive_line_energy_recovery": cosine * cosine,
        }

    oracle.records = [
        {
            "optimizer_update_index": 238,
            "aggregate": {
                "e5m2": {"all": {"polar_update": metric(0.99), "combined_prepolar": metric(0.99999)}},
                "e5m4": {"all": {"polar_update": metric(0.99996), "combined_prepolar": metric(0.99999)}},
                "e5m10_fp16_control": {"all": {"polar_update": metric(1.0), "combined_prepolar": metric(1.0)}},
            },
        }
    ]
    gate = oracle._gate()
    assert gate["classification"] == "PASS"
    assert gate["selected"] == "e5m4"
