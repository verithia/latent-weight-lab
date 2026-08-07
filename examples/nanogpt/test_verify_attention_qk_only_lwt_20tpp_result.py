from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from examples.nanogpt.verify_attention_qk_only_lwt_20tpp_result import (
    architecture_checks,
    fixed_curve_decision,
    parse_dense_muon_matrix_count,
    parse_logged_losses,
)


def plan() -> dict:
    return {
        "candidate": {
            "horizon": {"fixed_evaluation_steps": [10, 20, 30, 40]}
        },
        "decision_rule": {
            "dense_terminal_validation_ce": 3.15,
            "qkv_parent_validation_ce": [4.0, 3.8, 3.7, 3.6],
            "terminal_validation_ce_maximum": 3.59,
            "minimum_terminal_improvement_over_qkv": 0.015,
            "maximum_fixed_curve_gap_to_qkv": 0.0,
        },
    }


def test_parse_and_pass_all_registered_gates() -> None:
    text = "\n".join(
        [
            "step 10: train loss 3.99, val loss 3.99",
            "step 20: train loss 3.79, val loss 3.79",
            "step 30: train loss 3.69, val loss 3.69",
            "step 40: train loss 3.57, val loss 3.57",
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text(text)
        decision = fixed_curve_decision(plan(), parse_logged_losses(path))
    assert decision["scientific_gate_passed"] is True
    assert decision["terminal_improvement_over_qkv_ce"] == pytest.approx(0.03)
    assert decision["terminal_gap_to_dense_ce"] == pytest.approx(0.42)


def test_curve_terminal_and_improvement_gates_fail_independently() -> None:
    curve_failure = {
        10: {"train": 4.0, "validation": 4.001},
        20: {"train": 3.8, "validation": 3.79},
        30: {"train": 3.7, "validation": 3.69},
        40: {"train": 3.57, "validation": 3.57},
    }
    decision = fixed_curve_decision(plan(), curve_failure)
    assert decision["curve_passed"] is False
    assert decision["terminal_passed"] is True
    assert decision["improvement_passed"] is True

    terminal_failure = dict(curve_failure)
    terminal_failure[10] = {"train": 4.0, "validation": 3.99}
    terminal_failure[40] = {"train": 3.6, "validation": 3.591}
    decision = fixed_curve_decision(plan(), terminal_failure)
    assert decision["terminal_passed"] is False

    improvement_failure = dict(terminal_failure)
    improvement_failure[40] = {"train": 3.58, "validation": 3.588}
    modified = plan()
    modified["decision_rule"]["terminal_validation_ce_maximum"] = 3.59
    decision = fixed_curve_decision(modified, improvement_failure)
    assert decision["terminal_passed"] is True
    assert decision["improvement_passed"] is False


def test_missing_nonfinite_and_muon_ownership_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text("optimizer=muon matrix_tensors=48 adamw_other_tensors=363\n")
        assert parse_dense_muon_matrix_count(path) == 48
        path.write_text("step 10: train loss nan, val loss 4.0\n")
        with pytest.raises(ValueError, match="non-finite"):
            parse_logged_losses(path)

    with pytest.raises(ValueError, match="lacks fixed evaluations"):
        fixed_curve_decision(plan(), {})


def test_architecture_requires_qk_only_and_dense_v_cproj_mlp() -> None:
    target = "attn.c_attn.qk_headwise"
    config = {
        "block_fht_targets": [target],
        "block_fht_output_gain_targets": [target],
        "block_fht_attn_cayley_targets": [target],
        "block_fht_attn_cayley_output_targets": [target],
        "block_fht_attn_cayley_bilateral_targets": [target],
        "block_fht_attn_cayley_ranks": {target: 64},
        "block_fht_mlp_cfc_directed_product": False,
        "block_fht_mlp_cproj_muon_matched_givens": False,
        "block_fht_cproj_product_fht_factors": 0,
        "block_fht_cproj_lowrank_rank": 0,
    }
    assert all(architecture_checks(config, dict(config)).values())
    bad = dict(config)
    bad["block_fht_targets"] = [target, "attn.c_attn.v"]
    assert architecture_checks(bad, bad)["config_qk_only_generated"] is False
    bad = dict(config)
    bad["block_fht_mlp_cfc_directed_product"] = True
    assert architecture_checks(bad, bad)["config_mlp_cfc_dense"] is False
