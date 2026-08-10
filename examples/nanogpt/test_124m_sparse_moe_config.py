from __future__ import annotations

import json
from pathlib import Path

from examples.nanogpt.mfu_preflight import estimate_active_params


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_dense_moe8_top2_0p5tpp_lr24e4.json"
)


def load() -> dict:
    return json.loads(CONFIG.read_text())


def test_config_is_complete_expert_top2_control() -> None:
    config = load()
    assert config["method"] == "baseline"
    assert config["moe_num_experts"] == 8
    assert config["moe_top_k"] == 2
    assert config["moe_expert_hidden_multiplier"] == 2
    assert config["optimizer"] == "muon"
    assert config["estimated_active_params"] == estimate_active_params(config)
    assert config["estimated_active_params"] == 124447488
    assert config["estimated_stored_params"] == 294316800


def test_config_is_performance_gated_and_deterministic() -> None:
    config = load()
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] >= 0.2
    assert "foreground" in config["monitoring_policy"]
    assert config["registered_resume_determinism_required"] is True
    assert config["save_checkpoint"] is True
    assert config["fixed_eval_indices"] is True
    assert config["selection_endpoint"].startswith(
        "terminal fixed-evaluation validation CE"
    )


def test_config_matches_0p5tpp_horizon() -> None:
    config = load()
    assert config["tokens_per_iter"] == (
        config["batch_size"]
        * config["gradient_accumulation_steps"]
        * config["block_size"]
    )
    assert config["scheduled_tokens"] == (
        config["tokens_per_iter"] * config["max_iters"]
    )
    assert abs(
        config["scheduled_tpp_active"]
        - config["scheduled_tokens"] / config["estimated_active_params"]
    ) < 1e-15
