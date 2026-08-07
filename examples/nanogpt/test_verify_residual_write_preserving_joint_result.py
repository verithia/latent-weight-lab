from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from examples.nanogpt.verify_residual_write_preserving_joint_result import (
    architecture_checks,
    fixed_curve_decision,
    parse_dense_muon_matrix_count,
    parse_logged_losses,
)


def plan() -> dict:
    return {
        "candidate": {"fixed_evaluation_steps": [10, 20, 30, 40]},
        "decision_rule": {
            "qkv_parent_validation_ce": [4.0, 3.8, 3.7, 3.6],
            "fair_blockwise_dense_validation_ce": [4.1, 3.81, 3.69, 3.59],
            "terminal_validation_ce_maximum": 3.61,
            "terminal_gap_to_qkv_parent_maximum": 0.01,
            "maximum_fixed_curve_gap_to_qkv_parent": 0.015,
        },
    }


def test_parse_and_pass_fixed_curve() -> None:
    text = "\n".join(
        [
            "step 10: train loss 4.0, val loss 3.999",
            "step 20: train loss 3.8, val loss 3.799",
            "step 30: train loss 3.7, val loss 3.699",
            "step 40: train loss 3.6, val loss 3.598",
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text(text)
        decision = fixed_curve_decision(plan(), parse_logged_losses(path))
    assert decision["scientific_gate_passed"] is True
    assert decision["terminal_gap_to_qkv_parent_ce"] == pytest.approx(-0.002)


def test_curve_and_terminal_fail_independently() -> None:
    logged = {
        10: {"train": 4.0, "validation": 4.016},
        20: {"train": 3.8, "validation": 3.8},
        30: {"train": 3.7, "validation": 3.7},
        40: {"train": 3.6, "validation": 3.611},
    }
    decision = fixed_curve_decision(plan(), logged)
    assert decision["terminal_passed"] is False
    assert decision["terminal_gap_passed"] is False
    assert decision["curve_passed"] is False


def test_muon_ownership_and_nonfinite_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text("optimizer=muon matrix_tensors=24 adamw_other_tensors=435\n")
        assert parse_dense_muon_matrix_count(path) == 24
        path.write_text("step 10: train loss nan, val loss 4.0\n")
        with pytest.raises(ValueError, match="non-finite"):
            parse_logged_losses(path)


def test_architecture_requires_dense_residual_writes() -> None:
    config = {
        "block_fht_targets": ["attn.c_attn.qk_headwise", "attn.c_attn.v"],
        "block_fht_mlp_cfc_directed_product": True,
        "block_fht_mlp_cfc_directed_product_schedule": [22] * 6,
        "block_fht_mlp_cproj_muon_matched_givens": False,
        "block_fht_mlp_cproj_muon_matched_givens_layers": [],
        "block_fht_cproj_product_fht_factors": 0,
        "block_fht_cproj_lowrank_rank": 0,
    }
    checks = architecture_checks(config, dict(config))
    assert all(checks.values())
    config["block_fht_targets"].append("attn.c_proj")
    assert architecture_checks(config, config)["attention_cproj_dense"] is False
