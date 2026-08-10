from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_vo_quotient_manifold_design.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_vo_quotient_manifold_plan.json"
RESULT = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_vo_quotient_manifold_result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_quotient_oracle_couples_v_and_o_and_removes_gauge() -> None:
    design = json.loads(DESIGN.read_text())
    functional = design["functional_object"]
    assert functional["per_head_product"] == "M_h = O[:, h_slice] V_h"
    assert "leave M_h unchanged" in functional["gauge"]
    assert design["atlas"]["primary"] == "exact coupled quotient atoms"
    assert design["atlas"]["parameter_updates"] == 0


def test_quotient_oracle_has_strict_temporal_and_metric_gates() -> None:
    design = json.loads(DESIGN.read_text())
    atlas = design["atlas"]
    assert "through step 1140" in atlas["selection_information"]
    assert "step 1200 onward" in atlas["heldout_information"]
    assert atlas["maximum_atoms"] == 40
    assert design["decision_rule"]["thresholds"] == {
        "aggregate_recovery_minimum": 0.9,
        "minimum_every_layer_recovery": 0.75,
        "minimum_late_layer_8_to_11_recovery": 0.75,
    }
    assert all(value is False for value in design["authorization"].values())


def test_quotient_plan_pins_code_design_and_zero_update() -> None:
    plan = json.loads(PLAN.read_text())
    identity = plan["identity"]
    entrypoint = ROOT / "examples/nanogpt/analyze_attention_vo_quotient_manifold.py"
    assert identity["entrypoint_sha256"] == sha256(entrypoint)
    assert identity["design_sha256"] == sha256(DESIGN)
    assert plan["protocol"]["parameter_updates"] == 0
    assert all(value is False for value in plan["authorization"].values())


def test_quotient_result_accepts_states_but_rejects_static_tangent_atlas() -> None:
    result = json.loads(RESULT.read_text())
    assert result["classification"] == "ATTENTION_VO_QUOTIENT_MANIFOLD_REJECT"
    assert result["identity"]["plan_sha256"] == sha256(PLAN)
    primary = result["summaries"]["joint_quotient_discovery"]
    assert primary["state"]["aggregate_eval_recovery"] > 0.96
    assert primary["state"]["minimum_layer_eval_recovery"] > 0.90
    assert primary["chord"]["aggregate_eval_recovery"] < 0.03
    assert primary["muon_direction"]["aggregate_eval_recovery"] < 0.03
    assert result["decision"]["language_model_training_authorized"] is False
