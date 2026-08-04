from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PLAN = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_error_feedback_nominal_cap192_plan.json"
)
RESOLUTION = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_error_feedback_nominal_cap192_resolution.json"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(commit: str, path: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=REPO
    )
    return hashlib.sha256(payload).hexdigest()


def test_resolution_binds_plan_config_and_implementation() -> None:
    resolution = load(RESOLUTION)
    assert sha256(PLAN) == resolution["plan"]["sha256"]
    config_path = REPO / resolution["config"]["path"]
    assert sha256(config_path) == resolution["config"]["sha256"]
    config = load(config_path)
    assert config["implementation_commit"] == resolution["implementation"]["commit"]
    for relative, expected in resolution["implementation"]["source_hashes"].items():
        assert git_blob_sha256(config["implementation_commit"], relative) == expected


def test_cap_is_the_only_scientific_change_from_full_carry_parent() -> None:
    resolution = load(RESOLUTION)
    candidate = load(REPO / resolution["config"]["path"])
    parent = load(
        REPO
        / "examples/nanogpt/configs/"
        "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_"
        "twopassfresh88_errorfeedback_0p5tpp.json"
    )
    frozen_fields = (
        "block_fht_targets",
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay",
        "block_fht_mlp_cproj_muon_matched_givens_stages",
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages",
        "block_fht_mlp_cproj_muon_matched_givens_neighbors",
        "block_fht_mlp_cproj_muon_matched_givens_seed",
        "block_fht_mlp_cproj_muon_matched_givens_refresh_interval",
        "block_fht_mlp_cproj_muon_matched_givens_fast_fresh",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "min_lr",
        "max_iters",
        "lr_decay_iters",
        "model_seed",
        "train_data_seed",
        "optimizer",
        "muon_momentum",
        "muon_ns_steps",
    )
    for field in frozen_fields:
        assert candidate[field] == parent[field], field
    assert parent.get(
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps"
    ) is None
    assert candidate[
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps"
    ] == 192.0
    assert "mlp.c_fc" not in candidate["block_fht_targets"]


def test_gates_and_direct_polling_are_frozen() -> None:
    plan = load(PLAN)
    resolution = load(RESOLUTION)
    config = load(REPO / resolution["config"]["path"])
    command = resolution["execution"]["mfu_command"]
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    assert command[command.index("--min-fraction") + 1] == "0.2"
    assert resolution["gates"]["required_scratch_cap_events"] == 1
    assert resolution["gates"]["required_scientific_cap_events"] == 1
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == 5.522365207672119
    assert plan["decision_policy"]["automatic_rerun_authorized"] is False
    assert resolution["authorization"]["automatic_rerun_authorized"] is False
    assert resolution["authorization"]["cap_sweep_authorized"] is False
    assert resolution["execution"]["direct_foreground_polling"] is True
    assert resolution["execution"]["watchdog"] is False
