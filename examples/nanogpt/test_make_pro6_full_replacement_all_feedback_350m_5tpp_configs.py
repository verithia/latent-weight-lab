from __future__ import annotations

import hashlib
import json

import pytest

from examples.nanogpt.make_pro6_full_replacement_all_feedback_350m_5tpp_configs import (
    OUTPUTS,
    PARENTS,
    PLAN,
    QK,
    RANKING,
    SCREENS,
    SELECTIONS,
    build,
)


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("slot,slug", SELECTIONS.items())
def test_ranked_full_replacement_is_deterministic_and_preserves_5tpp_schedule(slot, slug):
    parent = load(PARENTS[slot])
    screen = load(SCREENS[slug])
    candidate = load(OUTPUTS[slot])
    assert candidate == build(parent, screen, slot)
    for field in (
        "n_layer", "n_embd", "n_head", "batch_size",
        "gradient_accumulation_steps", "block_size", "max_iters",
        "lr_decay_iters", "planned_tokens", "scheduled_tokens",
        "eval_interval", "eval_batch_size", "eval_iters", "warmup_iters",
        "learning_rate", "min_lr", "optimizer",
    ):
        assert candidate[field] == parent[field]
    assert candidate["planned_tpp"] == 5.0
    assert candidate["max_iters"] == 6764
    assert candidate["scheduled_tokens"] == 1773142016
    assert candidate["ladder_role"] == "confirmation_registered"
    assert candidate["ladder_slot"] == slot


@pytest.mark.parametrize("slot", SELECTIONS)
def test_every_ambient_family_has_lattice_and_temporal_feedback(slot):
    candidate = load(OUTPUTS[slot])
    assert candidate["block_fht_targets"] == [QK]
    assert candidate["block_fht_attn_v_int8_lattice"] is True
    assert candidate["block_fht_attn_v_int8_lattice_error_feedback"] is True
    assert candidate["block_fht_attn_cproj_int8_lattice"] is True
    assert candidate["block_fht_attn_cproj_int8_lattice_error_feedback"] is True
    assert candidate["block_fht_mlp_int8_lattice_targets"] == ["mlp.c_fc", "mlp.c_proj"]
    assert candidate["block_fht_mlp_int8_lattice_error_feedback"] is True
    assert candidate["selected_lwt_allocation"] == {
        "generated": [QK],
        "ambient_int8_lattice_with_fp16_feedback": [
            "attn.c_attn.v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"
        ],
    }
    assert candidate["zero_point_five_tpp_ranking_artifact_sha256"] == sha256(RANKING)


def test_plan_materializes_only_ranked_top_two_and_blocks_launch_pending_mfu():
    plan = load(PLAN)
    assert set(OUTPUTS) == {"top1", "top2"}
    assert plan["execution_order"] == ["top1", "top2"]
    assert plan["excluded_candidates"] == ["mult0p50"]
    assert plan["performance_gate"]["foreground_polling"] is True
    assert plan["performance_gate"]["watchdog"] is False
    assert plan["performance_gate"]["both_configs_must_pass_before_first_launch"] is True
    assert plan["performance_gate"]["launch_authorized"] is False
    assert plan["immutable_ranking"]["sha256"] == sha256(RANKING)
    for slot in SELECTIONS:
        assert plan["confirmations"][slot]["config_sha256"] == sha256(OUTPUTS[slot])
    assert plan["monitoring"]["agent_mention"] == "@Codex"
    assert plan["monitoring"]["heartbeat_minutes"] == 90
