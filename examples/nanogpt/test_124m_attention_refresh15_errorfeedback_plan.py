from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.nanogpt.make_pro6_attention_refresh15_errorfeedback_config import (
    FP32_FEEDBACK_BYTES,
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    json_bytes,
    make_config,
    make_plan,
)
from examples.nanogpt.train import parse_args


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def test_generator_is_deterministic_and_binds_rejected_parent() -> None:
    source = load(
        ROOT
        / "examples/nanogpt/configs/"
        "pro6_mai_v3_124m_fullattn_refresh15_muon_matched_givens_"
        "0p5tpp_lr24e4.json"
    )
    raw = json_bytes(make_config(source))
    assert raw == OUTPUT_CONFIG.read_bytes()
    assert make_plan(hashlib.sha256(raw).hexdigest()) == load(OUTPUT_PLAN)


def test_candidate_changes_only_temporal_feedback_and_accounts_state() -> None:
    config = load(OUTPUT_CONFIG)
    assert config["block_fht_attn_muon_matched_givens_error_feedback"] is True
    assert (
        config["block_fht_attn_muon_matched_givens_error_feedback_decay"]
        == 0.5
    )
    assert config[
        "block_fht_attn_muon_matched_givens_error_feedback_max_nominal_steps"
    ] is None
    assert config["block_fht_attn_muon_matched_givens_refresh_interval"] == 15
    accounting = config["candidate_error_feedback_accounting"]
    assert accounting["fp32_feedback_state_bytes"] == FP32_FEEDBACK_BYTES
    assert accounting["persistent_dense_feedback_state"] is True
    assert "not optimizer-memory compression" in accounting["claim"]


def test_parser_and_plan_resolve_direct_polling_and_fresh_mfu_gate() -> None:
    with patch.object(sys, "argv", ["train.py", "--config", str(OUTPUT_CONFIG)]):
        args = parse_args()
    assert args.block_fht_attn_muon_matched_givens_error_feedback is True
    assert args.block_fht_attn_muon_matched_givens_error_feedback_decay == 0.5
    plan = load(OUTPUT_PLAN)
    assert plan["performance_gate"]["timed_updates"] == 16
    assert plan["performance_gate"]["watchdog"] is False
    assert plan["scientific_run"]["watchdog"] is False
    assert plan["decision_rule"]["terminal_validation_ce_maximum"] == 5.3924
    assert plan["decision_rule"]["automatic_larger_rung_authorized"] is False
