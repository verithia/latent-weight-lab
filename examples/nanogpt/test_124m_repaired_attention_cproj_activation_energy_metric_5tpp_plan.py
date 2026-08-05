from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cproj_activation_energy_metric_5tpp_plan.json"


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_causal_inputs_are_hash_pinned() -> None:
    plan = load(PLAN)
    for item in plan["causal_evidence"].values():
        assert sha256(REPO / item["path"]) == item["sha256"]


def test_candidate_is_one_bounded_hidden_metric_change() -> None:
    plan = load(PLAN)
    change = plan["single_factor_change"]
    metric = plan["metric_definition"]
    safety = plan["safety_against_prior_failure"]
    assert change["enabled_flag"] == "block_fht_mlp_cproj_activation_energy_metric"
    assert metric["ema_decay"] == 0.95
    assert metric["minimum_weight"] == 0.25
    assert metric["maximum_weight"] == 4.0
    assert metric["epsilon"] == 1e-6
    assert safety["not_output_side"] is True
    assert safety["not_full_activation_covariance"] is True
    assert safety["bounded_metric_condition_number"] == 16.0
    assert "zero additional trainable parameters and zero inference-time transforms" in change["unchanged"]


def test_gate_and_monitoring_are_frozen() -> None:
    plan = load(PLAN)
    performance = plan["performance_gate"]
    rule = plan["decision_rule"]
    monitoring = plan["scientific_run"]["monitoring"]
    assert performance["minimum_mfu_fraction"] >= 0.20
    assert performance["protocol"].startswith("Foreground")
    assert monitoring["milestones"] == [20, 50, 100]
    assert monitoring["heartbeat_minutes"] == 90
    assert monitoring["progress_resets_heartbeat"] is True
    assert monitoring["callback_endpoint"].endswith("/send-opencode-test")
    assert monitoring["callback_mention"] == "@Codex"
    assert monitoring["terminal_delivery_once"] is True
    assert rule["promote_terminal_validation_ce_maximum"] == 3.6478
    assert rule["threshold_changed_after_measurement"] is False


def test_only_one_run_is_authorized() -> None:
    authorization = load(PLAN)["authorization"]
    assert authorization["implementation_and_tests"] is True
    assert authorization["one_exact_config_mfu_gate"] is True
    assert authorization["one_scientific_5tpp_run_after_gate"] is True
    assert authorization["automatic_rerun"] is False
    assert authorization["parallel_arm"] is False
    assert authorization["larger_rung"] is False
    assert authorization["post_hoc_metric_sweep"] is False
