import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_cfcdecay1_cprojdecay0p5_5tpp_lr24e4.json"
V2 = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_cfcdecay1_cprojdecay0p5_5tpp_lr24e4_v2.json"
PLAN = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_full_mlp_5tpp_plan_v2.json"
FAILURE = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_full_mlp_5tpp_pretraining_failure.json"
MFU_V2 = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_full_mlp_5tpp_mfu_result_v2.json"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_v2_identity_and_pretraining_failure_are_sealed() -> None:
    plan = json.loads(PLAN.read_text())
    assert sha256(V2) == plan["identity"]["config_sha256"]
    assert sha256(V1) == plan["registered_config_transform"]["source_sha256"]
    assert sha256(FAILURE) == plan["pretraining_failure"]["sha256"]
    failure = json.loads(FAILURE.read_text())
    assert failure["run"]["training_updates_executed"] == 0
    assert failure["diagnosis"]["scientific_result_valid"] is False
    assert failure["diagnosis"]["loss_or_threshold_observed"] is False

def test_only_registered_v2_fields_changed() -> None:
    v1 = json.loads(V1.read_text())
    v2 = json.loads(V2.read_text())
    changed = {key for key in set(v1) | set(v2) if v1.get(key) != v2.get(key)}
    assert changed == {
        "checkpoint_wall_clock_seconds", "out_dir", "mfu_preflight_certificate",
        "hpo_stage", "ladder_slot", "supersedes_config",
        "supersedes_config_sha256", "pretraining_validation_repair",
        "implementation_commit",
    }
    assert v2["checkpoint_wall_clock_seconds"] == 7200
    assert v2["max_iters"] == 2373
    assert v2["block_fht_native_extension_required"] is True

def test_thresholds_and_performance_policy_are_unchanged() -> None:
    plan = json.loads(PLAN.read_text())
    v2 = json.loads(V2.read_text())
    assert plan["decision_rule"]["pass_validation_ce_maximum"] == 3.6478
    assert plan["decision_rule"]["nonbinding_parity_diagnostic_validation_ce_maximum"] == 3.6378
    assert plan["decision_rule"]["threshold_changed_after_failure"] is False
    assert v2["mfu_preflight_required"] is True
    assert v2["mfu_min_fraction"] >= 0.20
    assert plan["authorization"]["automatic_rerun"] is False

def test_fresh_v2_mfu_gate_passes_for_exact_config() -> None:
    result = json.loads(MFU_V2.read_text())
    assert result["identity"]["config_sha256"] == sha256(V2)
    assert result["measurement"]["mfu_fraction"] >= 0.20
    assert result["measurement"]["native_block_fht_extension"]["loaded"] is True
    assert result["decision"]["scientific_attempt2_authorized"] is True
    assert result["execution"]["watchdog_used"] is False
