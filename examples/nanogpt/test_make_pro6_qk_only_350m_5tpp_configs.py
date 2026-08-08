from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from examples.nanogpt.make_pro6_qk_only_350m_5tpp_configs import (
    HEAD_DIM,
    OUTPUTS,
    PARENTS,
    PLAN,
    QK,
    RANKING,
    SCREENS,
    SELECTIONS,
    build,
)
from examples.nanogpt.train import parse_args


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("slot,slug", SELECTIONS.items())
def test_exact_top2_config_is_deterministic_and_preserves_5tpp_schedule(
    slot: str, slug: str
) -> None:
    parent = load(PARENTS[slot])
    screen = load(SCREENS[slug])
    ranking = load(RANKING)
    candidate = load(OUTPUTS[slot])
    assert candidate == build(parent, screen, ranking, slot)
    for field in (
        "n_layer",
        "n_embd",
        "n_head",
        "batch_size",
        "gradient_accumulation_steps",
        "block_size",
        "max_iters",
        "lr_decay_iters",
        "planned_tokens",
        "scheduled_tokens",
        "eval_interval",
        "eval_batch_size",
        "eval_iters",
        "warmup_iters",
        "learning_rate",
        "min_lr",
        "optimizer",
    ):
        assert candidate[field] == parent[field]
    assert candidate["planned_tpp"] == 5.0
    assert candidate["max_iters"] == 6764
    assert candidate["scheduled_tokens"] == 1773142016
    assert candidate["ladder_role"] == "confirmation_registered"
    assert candidate["ladder_slot"] == slot


@pytest.mark.parametrize("slot,slug", SELECTIONS.items())
def test_qk_only_boundary_and_ranking_are_exact(slot: str, slug: str) -> None:
    candidate = load(OUTPUTS[slot])
    ranking = load(RANKING)
    assert candidate["n_embd"] // candidate["n_head"] == HEAD_DIM
    assert candidate["block_fht_targets"] == [QK]
    assert candidate["block_fht_output_gain_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_output_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_bilateral_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_ranks"] == {QK: HEAD_DIM}
    assert candidate["selected_lwt_allocation"]["generated"] == [QK]
    assert candidate["selected_lwt_allocation"]["dense_muon"] == [
        "attn.c_attn.v",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    ]
    assert candidate["zero_point_five_tpp_ranking_artifact_sha256"] == sha256(
        RANKING
    )
    assert ranking["decision"][slot] == slug
    with patch.object(sys, "argv", ["train.py", "--config", str(OUTPUTS[slot])]):
        args = parse_args()
    assert args.max_iters == 6764
    assert args.block_fht_targets == [QK]


def test_only_ranked_top_two_are_materialized_and_plan_is_launch_blocked() -> None:
    assert set(OUTPUTS) == {"top1", "top2"}
    assert not (
        ROOT
        / "examples/nanogpt/configs/"
        "pro6_mai_v3_350m_qk_only_qk64_outputgain_5tpp_mult0p50.json"
    ).exists()
    ranking = load(RANKING)
    assert ranking["decision"]["authorized_5tpp_config_count"] == 2
    assert ranking["decision"]["rejected_from_5tpp"] == ["mult0p50"]
    plan = load(PLAN)
    assert plan["execution_order"] == ["top1", "top2"]
    assert plan["excluded_candidates"] == ["mult0p50"]
    assert plan["performance_gate"]["foreground_polling"] is True
    assert plan["performance_gate"]["watchdog"] is False
    assert plan["performance_gate"]["both_configs_must_pass_before_first_launch"] is True
    assert plan["performance_gate"]["launch_authorized"] is False
    for slot, record in plan["confirmations"].items():
        assert record["config_sha256"] == sha256(OUTPUTS[slot])
    assert plan["monitoring"]["heartbeat_minutes"] == 90
    assert plan["monitoring"]["heartbeat_resets_on_progress"] is True
