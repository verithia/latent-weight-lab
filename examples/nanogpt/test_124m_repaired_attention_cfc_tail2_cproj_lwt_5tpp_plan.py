from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_matched_givens import MuonMatchedGivensLinear
from examples.nanogpt.train import parse_args


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_repairedfullattn_plus_cfc_tail2cproj_lwt_5tpp_lr24e4.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_repaired_attention_cfc_tail2_cproj_lwt_5tpp_plan.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_preregistered_identity_and_strict_thresholds() -> None:
    config = load_config()
    plan = json.loads(PLAN.read_text())
    assert config["registered_plan_sha256"] == sha256(PLAN)
    assert config["block_fht_mlp_cproj_muon_matched_givens_layers"] == [10, 11]
    assert plan["decision_rule"]["primary_terminal_gap_to_cfc_only_maximum"] == 0.005
    assert plan["decision_rule"]["primary_terminal_validation_ce_maximum"] == 3.630838041305542
    assert plan["decision_rule"]["fixed_curve_gap_to_cfc_only_maximum"] == 0.01
    assert plan["decision_rule"]["threshold_changed_after_measurement"] is False
    assert plan["authorization"]["automatic_rerun"] is False
    assert plan["authorization"]["different_mask"] is False
    assert plan["monitoring"]["milestone_callbacks"] is False
    assert plan["monitoring"]["heartbeat_callbacks"] is False


def test_scientific_parent_and_geometry_are_frozen() -> None:
    config = load_config()
    assert config["max_iters"] == 2373
    assert config["planned_tpp"] == 5
    assert config["eval_interval"] == 594
    assert config["eval_iters"] == 400
    assert config["block_fht_mlp_cfc_directed_product_schedule"] == [22] * 6
    assert config["block_fht_mlp_cfc_directed_product_error_feedback_decay"] == 1
    assert config["block_fht_mlp_cproj_muon_matched_givens_stages"] == 64
    assert config["block_fht_mlp_cproj_muon_matched_givens_residual_stages"] == 24
    assert config["block_fht_mlp_cproj_muon_matched_givens_neighbors"] == 64
    assert config["block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay"] == 0.5
    assert config["mfu_min_fraction"] >= 0.20
    assert config["block_fht_native_extension_required"] is True
    assert config["registered_resume_determinism_required"] is True
    assert config["checkpoint_wall_clock_seconds"] == 7200


def test_constructed_module_types_match_tail_two_mask() -> None:
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=8,
        bias=False,
        block_fht=True,
        block_fht_targets=("mlp.c_proj",),
        block_fht_mlp_cproj_muon_matched_givens=True,
        block_fht_mlp_cproj_muon_matched_givens_layers=(10, 11),
        block_fht_mlp_cproj_muon_matched_givens_stages=2,
        block_fht_mlp_cproj_muon_matched_givens_residual_stages=1,
        block_fht_mlp_cproj_muon_matched_givens_neighbors=4,
        block_fht_mlp_cproj_muon_matched_givens_fast_fresh=True,
    )
    model = GPT(config)
    assert all(
        not isinstance(model.transformer.h[layer].mlp.c_proj, MuonMatchedGivensLinear)
        for layer in range(10)
    )
    assert all(
        isinstance(model.transformer.h[layer].mlp.c_proj, MuonMatchedGivensLinear)
        for layer in (10, 11)
    )


def test_exact_scientific_json_passes_real_argument_validation(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(CONFIG)])
    parsed = parse_args()
    assert parsed.block_fht_mlp_cproj_muon_matched_givens is True
    assert parsed.block_fht_mlp_cproj_muon_matched_givens_layers == [10, 11]
