import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_cproj_int8_lattice_5tpp_plan.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_pins_parent_candidate_and_strict_terminal_gate() -> None:
    plan = json.loads(PLAN.read_text())
    candidate = ROOT / plan["candidate"]["config"]
    parent = ROOT / plan["promotion_basis"]["smallest_rung_result"]["path"]
    qkv_control = ROOT / plan["promotion_basis"]["qkv_dense_cproj_control"]["path"]
    dense_control = ROOT / plan["promotion_basis"]["ordinary_dense_control"]["path"]
    mfu_result = ROOT / plan["mfu_result"]["path"]
    terminal_result = ROOT / plan["terminal_result"]["path"]
    assert sha256(candidate) == plan["candidate"]["config_sha256"]
    assert sha256(parent) == plan["promotion_basis"]["smallest_rung_result"]["sha256"]
    assert sha256(qkv_control) == plan["promotion_basis"]["qkv_dense_cproj_control"]["sha256"]
    assert sha256(dense_control) == plan["promotion_basis"]["ordinary_dense_control"]["sha256"]
    assert sha256(mfu_result) == plan["mfu_result"]["sha256"]
    assert sha256(terminal_result) == plan["terminal_result"]["sha256"]
    config = json.loads(candidate.read_text())
    assert config["max_iters"] == 2373
    assert config["planned_tpp"] == 5.0
    assert config["block_fht_attn_cproj_int8_lattice"] is True
    assert plan["terminal_gate"]["maximum_validation_ce_delta_to_dense"] == 0.02
    assert plan["terminal_gate"]["maximum_terminal_validation_ce"] == 3.5602
    assert plan["authorization"]["automatic_20tpp"] is False
    assert plan["authorization"]["larger_model"] is False
    assert plan["authorization"]["exact_config_mfu_passed"] is True
    assert plan["mfu_result"]["mfu_fraction"] >= 0.2
    result = json.loads(terminal_result.read_text())
    assert result["classification"] == (
        "FULL_ATTENTION_PERSISTENT_STATE_NEAR_DENSE_AT_124M_5TPP"
    )
    assert result["comparison"]["delta_to_ordinary_dense_ce"] <= 0.01
    assert (
        result["fixed_model_compute_equivalence"]["terminal_dense_token_penalty"]
        <= 1.10
    )
    assert result["decision"]["automatic_20tpp_authorized"] is False
