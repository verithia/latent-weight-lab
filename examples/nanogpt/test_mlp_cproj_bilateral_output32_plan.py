from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.nanogpt.train import parse_args


ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_bilateral_output32_mfu_plan.json"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_binds_endpoint_selection_config_and_sources() -> None:
    plan = load(PLAN)
    config_path = ROOT / plan["candidate"]["config"]
    assert sha256(config_path) == plan["candidate"]["config_sha256"]
    for field, hash_field in (
        ("base_config", "base_config_sha256"),
        ("endpoint_selection_result", "endpoint_selection_result_sha256"),
        ("right_only_control_result", "right_only_control_result_sha256"),
    ):
        assert sha256(ROOT / plan["identity"][field]) == (
            plan["identity"][hash_field]
        )
    for relative, expected in plan["identity"]["source_hashes"].items():
        assert sha256(ROOT / relative) == expected


def test_candidate_is_the_smallest_selected_bilateral_full_carry_path() -> None:
    plan = load(PLAN)
    config = load(ROOT / plan["candidate"]["config"])
    assert config["block_fht_mlp_cproj_muon_matched_givens_stages"] == 64
    assert config[
        "block_fht_mlp_cproj_muon_matched_givens_residual_stages"
    ] == 24
    assert config[
        "block_fht_mlp_cproj_muon_matched_givens_output_stages"
    ] == 32
    assert config["block_fht_mlp_cproj_muon_matched_givens_neighbors"] == 64
    assert config[
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback"
    ] is True
    assert config[
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay"
    ] == 1.0
    assert config["muon_matched_givens_representation"][
        "coordinates_per_layer"
    ] == 147456
    assert config["muon_matched_givens_representation"][
        "coordinate_fraction_per_cproj"
    ] == 0.0625
    assert "mlp.c_fc" not in config["block_fht_targets"]


def test_config_is_accepted_by_production_loader() -> None:
    plan = load(PLAN)
    config_path = ROOT / plan["candidate"]["config"]
    with patch.object(sys, "argv", ["train.py", "--config", str(config_path)]):
        parsed = parse_args()
    assert parsed.block_fht_mlp_cproj_muon_matched_givens_output_stages == 32
    assert parsed.block_fht_mlp_cproj_muon_matched_givens_fast_fresh is True
    assert parsed.block_fht_native_extension_required is True


def test_mfu_gate_and_post_pass_authority_are_frozen() -> None:
    plan = load(PLAN)
    command = plan["execution"]["mfu_command"]
    assert command[command.index("--min-fraction") + 1] == "0.2"
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"
    assert plan["execution"]["mfu_polling"].startswith("foreground")
    assert plan["execution"]["watchdog"] is False
    assert plan["authorization"]["scientific_training_authorized"] is False
    assert plan["authorization"]["automatic_retry_authorized"] is False
    assert plan["authorization"]["larger_rung_authorized"] is False
    assert plan["decision_rule"]["threshold_changes_after_measurement"] is False
