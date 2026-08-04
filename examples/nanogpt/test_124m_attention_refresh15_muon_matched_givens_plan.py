from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.nanogpt.make_pro6_attention_refresh15_muon_matched_givens_config import (
    MATERIALIZED_ATTENTION_WEIGHTS,
    OUTPUT_CONFIG,
    OUTPUT_PLAN,
    UPDATE_COORDINATES,
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_is_deterministic_and_plan_binds_exact_config() -> None:
    source = load(
        ROOT
        / "examples/nanogpt/configs/"
        "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_"
        "cproj8_targeted_bilateral_fullcayleylr_0p5tpp_lr24e4.json"
    )
    generated = make_config(source)
    raw = json_bytes(generated)
    assert raw == OUTPUT_CONFIG.read_bytes()
    assert make_plan(hashlib.sha256(raw).hexdigest()) == load(OUTPUT_PLAN)
    assert sha256(OUTPUT_CONFIG) == load(OUTPUT_PLAN)["identity"][
        "candidate_config_sha256"
    ]


def test_candidate_preserves_parent_schedule_and_registered_geometry() -> None:
    config = load(OUTPUT_CONFIG)
    assert config["method"] == "block_fht"
    assert config["optimizer"] == "muon"
    assert config["max_iters"] == config["lr_decay_iters"] == 238
    assert config["warmup_iters"] == 23
    assert config["learning_rate"] == 0.0024
    assert config["data_dir"] == (
        "/home/pro6000-9980x/MappingNetworks/data/finewebedu_20b"
    )
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk",
        "attn.c_attn.v",
        "attn.c_proj",
    ]
    assert config["block_fht_attn_muon_matched_givens_targets"] == (
        config["block_fht_targets"]
    )
    assert config["block_fht_attn_muon_matched_givens_stages"] == 64
    assert config["block_fht_attn_muon_matched_givens_neighbors"] == 128
    assert config[
        "block_fht_attn_muon_matched_givens_refresh_interval"
    ] == 15
    assert config["block_fht_output_gain_targets"] == []
    assert config["block_fht_input_gain_targets"] == []
    assert not any(
        key.startswith("block_fht_attn_cayley_") for key in config
    )


def test_coordinate_and_materialized_state_accounting_is_honest() -> None:
    config = load(OUTPUT_CONFIG)
    accounting = config["candidate_parameter_accounting"]
    assert MATERIALIZED_ATTENTION_WEIGHTS == 28_311_552
    assert UPDATE_COORDINATES == 1_769_472
    assert UPDATE_COORDINATES / MATERIALIZED_ATTENTION_WEIGHTS == 0.0625
    assert accounting["persistent_dense_weight_buffer"] is True
    assert accounting["dense_weight_gradient"] is True
    assert accounting["dense_muon_momentum"] is True
    assert accounting["sparse_update_coordinates"] == UPDATE_COORDINATES
    assert "does not claim" in accounting["claim"]


def test_parser_resolves_candidate_and_performance_gate_crosses_refresh() -> None:
    with patch.object(sys, "argv", ["train.py", "--config", str(OUTPUT_CONFIG)]):
        args = parse_args()
    assert args.block_fht_attn_muon_matched_givens_targets == [
        "attn.c_attn.qk",
        "attn.c_attn.v",
        "attn.c_proj",
    ]
    assert args.block_fht_attn_muon_matched_givens_refresh_interval == 15
    plan = load(OUTPUT_PLAN)
    gate = plan["performance_gate"]
    assert gate["warmup_updates"] == 1
    assert gate["timed_updates"] == 16
    assert gate["watchdog"] is False
    assert gate["callbacks"] is False
    assert plan["scientific_run"]["watchdog"] is False
    assert plan["decision_rule"]["terminal_validation_ce_maximum"] == 5.3924
    assert plan["decision_rule"]["automatic_larger_rung_authorized"] is False
