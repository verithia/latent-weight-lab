from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from examples.nanogpt.verify_qk_only_plus_cfc_directed_20tpp_result import (
    architecture_checks,
    fixed_curve_decision,
    parse_runtime_accounting,
)


def plan() -> dict:
    return {
        "candidate": {"horizon": {"fixed_evaluation_steps": [10, 20, 30, 40]}},
        "decision_rule": {
            "dense_terminal_validation_ce": 3.15,
            "qk_only_parent_validation_ce": [4.0, 3.8, 3.7, 3.6],
            "terminal_validation_ce_maximum": 3.605,
            "maximum_fixed_curve_gap_to_qk_only_parent": 0.005,
        },
    }


def test_fixed_curve_pass_and_independent_failures() -> None:
    rows = {
        10: {"train": 3.99, "validation": 3.99},
        20: {"train": 3.79, "validation": 3.79},
        30: {"train": 3.69, "validation": 3.69},
        40: {"train": 3.59, "validation": 3.59},
    }
    decision = fixed_curve_decision(plan(), rows)
    assert decision["scientific_gate_passed"] is True
    assert decision["terminal_gap_to_qk_only_parent_ce"] == pytest.approx(-0.01)
    bad_curve = {key: dict(value) for key, value in rows.items()}
    bad_curve[10]["validation"] = 4.006
    assert fixed_curve_decision(plan(), bad_curve)["curve_passed"] is False
    bad_terminal = {key: dict(value) for key, value in rows.items()}
    bad_terminal[40]["validation"] = 3.606
    assert fixed_curve_decision(plan(), bad_terminal)["terminal_passed"] is False
    with pytest.raises(ValueError, match="lacks fixed evaluations"):
        fixed_curve_decision(plan(), {})


def test_architecture_requires_exact_feature_maps_and_dense_residual_writes() -> None:
    qk = "attn.c_attn.qk_headwise"
    payload = {
        "block_fht_targets": [qk],
        "block_fht_attn_cayley_ranks": {qk: 64},
        "block_fht_output_gain_targets": [qk],
        "block_fht_mlp_cfc_directed_product": True,
        "block_fht_mlp_cfc_directed_product_schedule": [22] * 6,
        "block_fht_mlp_cfc_directed_product_error_feedback": True,
        "block_fht_mlp_cfc_directed_product_error_feedback_decay": 1.0,
        "block_fht_mlp_cproj_muon_matched_givens": False,
        "block_fht_cproj_product_fht_factors": 0,
        "block_fht_cproj_lowrank_rank": 0,
    }
    assert all(architecture_checks(payload, dict(payload)).values())
    bad = dict(payload)
    bad["block_fht_targets"] = [qk, "attn.c_attn.v"]
    assert architecture_checks(bad, bad)["config_qk_only_generated"] is False
    bad = dict(payload)
    bad["block_fht_mlp_cfc_directed_product_error_feedback_decay"] = 0.5
    assert architecture_checks(bad, bad)["config_cfc_error_feedback_decay1"] is False


def test_runtime_accounting_is_exact_and_fail_closed() -> None:
    text = "\n".join(
        [
            "parameters: total=85,605,360 trainable=85,605,360",
            "block_fht: modules=156 generated=42,467,328 latent=5,007,600",
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text(text)
        accounting = parse_runtime_accounting(path)
        assert accounting == {
            "total": 85605360,
            "trainable": 85605360,
            "modules": 156,
            "generated": 42467328,
            "latent": 5007600,
        }
        path.write_text(text + "\n" + text)
        with pytest.raises(ValueError, match="expected one"):
            parse_runtime_accounting(path)
