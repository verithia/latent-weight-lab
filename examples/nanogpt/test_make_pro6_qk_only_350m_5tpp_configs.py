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
MFU_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "350m_qk_only_functional_lwt_5tpp_mfu_result.json"
)
TOP1_RUN_METADATA = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "350m_qk_only_functional_lwt_5tpp_top1_run_metadata.json"
)


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


def test_both_exact_5tpp_mfu_gates_are_sealed() -> None:
    result = load(MFU_RESULT)
    assert result["classification"] == "PASS_BOTH_EXACT_CONFIG_MFU_GATES"
    assert result["foreground_polled"] is True
    assert result["watchdog"] is False
    assert result["decision"]["top1_launch_authorized"] is True
    assert result["decision"]["top2_launch_authorized_after_top1_terminal_seal"] is True
    for slot, record in result["confirmations"].items():
        assert record["config_sha256"] == sha256(OUTPUTS[slot])
        assert record["passed"] is True
        assert record["native_extension_loaded"] is True
        assert record["all_logged_losses_finite"] is True
        assert record["mfu_fraction"] >= 0.20
        assert record["peak_mib"] < 97887


def test_top1_running_record_pins_command_commit_and_monitoring() -> None:
    record = load(TOP1_RUN_METADATA)
    assert record["classification"] == "RUNNING_AFTER_BOTH_EXACT_CONFIG_MFU_GATES"
    assert record["scientific_identity"]["slot"] == "top1"
    assert record["scientific_identity"]["execution_git_commit"] == (
        "c2d105be678248b38ac77ea5ff9cffa490d09b1c"
    )
    assert record["immutable_inputs"]["config"]["sha256"] == sha256(
        OUTPUTS["top1"]
    )
    assert record["immutable_inputs"]["mfu_result"]["sha256"] == sha256(
        MFU_RESULT
    )
    assert record["code_and_command"]["command"][2:4] == [
        "-m",
        "examples.nanogpt.train",
    ]
    assert record["immutable_inputs"]["dataset_manifest"]["sha256"] == (
        "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
    )
    assert record["monitoring"]["milestones"] == [20, 50, 80, 100]
    assert record["monitoring"]["callback_mention"] == "@Codex"
