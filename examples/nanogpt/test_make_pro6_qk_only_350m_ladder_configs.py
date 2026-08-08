from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from examples.nanogpt.make_pro6_qk_only_350m_ladder_configs import (
    BASE_LR,
    HEAD_DIM,
    MULTIPLIERS,
    OUTPUTS,
    PARENTS,
    PLAN,
    QK,
    QK_RESULT,
    QK_SOURCE,
    build,
)
from examples.nanogpt.train import parse_args


ROOT = Path(__file__).resolve().parents[2]
MFU_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "350m_qk_only_functional_lwt_0p5tpp_mfu_result.json"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("slug,multiplier", MULTIPLIERS.items())
def test_generated_config_is_deterministic_and_preserves_mai_rung(slug: str, multiplier: float) -> None:
    parent = load(PARENTS[slug])
    source = load(QK_SOURCE)
    candidate = load(OUTPUTS[slug])
    assert candidate == build(parent, source, slug)
    for field in (
        "n_layer",
        "n_embd",
        "n_head",
        "batch_size",
        "gradient_accumulation_steps",
        "block_size",
        "max_iters",
        "planned_tokens",
        "scheduled_tokens",
        "eval_batch_size",
        "eval_iters",
        "eval_seed",
        "fixed_eval_index_spec_sha256",
        "data_manifest_sha256",
        "optimizer",
        "muon_momentum",
        "muon_ns_steps",
        "muon_adamw_lr_scale",
    ):
        assert candidate[field] == parent[field]
    assert candidate["planned_tpp"] == 0.5
    assert candidate["max_iters"] == 677
    assert candidate["learning_rate"] == pytest.approx(BASE_LR * multiplier)
    assert candidate["min_lr"] == pytest.approx(BASE_LR * multiplier * 0.1)


@pytest.mark.parametrize("slug", MULTIPLIERS)
def test_qk_only_structure_is_head_dimension_invariant(slug: str) -> None:
    candidate = load(OUTPUTS[slug])
    assert candidate["n_embd"] // candidate["n_head"] == HEAD_DIM
    assert candidate["block_fht_targets"] == [QK]
    assert candidate["block_fht_output_gain_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_output_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_bilateral_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_ranks"] == {QK: HEAD_DIM}
    assert candidate["selected_lwt_allocation"]["dense_muon"] == [
        "attn.c_attn.v",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    ]
    with patch.object(sys, "argv", ["train.py", "--config", str(OUTPUTS[slug])]):
        args = parse_args()
    assert args.n_layer == 24
    assert args.n_embd == 1024
    assert args.n_head == 16
    assert args.block_fht_targets == [QK]


def test_plan_pins_configs_evidence_and_nonautomatic_promotion() -> None:
    plan = load(PLAN)
    assert plan["schema_version"] == "mai_350m_qk_only_functional_lwt_0p5tpp_plan_v1"
    assert plan["scientific_status"] == {
        "automatic_promotion_from_124m": False,
        "new_user_directed_screen": True,
        "larger_rung_claimed": False,
        "five_tpp_authorized": False,
    }
    assert plan["theory"]["head_dimension_invariant"] == HEAD_DIM
    assert plan["theory"]["qk_cayley_rank"] == HEAD_DIM
    assert plan["ladder"]["candidate_multipliers"] == list(MULTIPLIERS.values())
    for slug, record in plan["candidates"].items():
        assert record["config_sha256"] == sha256(OUTPUTS[slug])
    for record in plan["immutable_evidence"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]
    result = load(QK_RESULT)
    assert result["decision"]["larger_rung_authorized"] is False
    assert plan["performance_gate"]["foreground_polling"] is True
    assert plan["performance_gate"]["watchdog"] is False
    assert plan["monitoring"]["terminal_only"] is True
    assert plan["monitoring"]["agent_mention"] == "@Codex"


def test_all_exact_configs_pass_real_mfu_gate() -> None:
    result = load(MFU_RESULT)
    assert result["classification"] == "PASS_ALL_EXACT_CONFIG_MFU_GATES"
    assert result["foreground_polled"] is True
    assert result["watchdog"] is False
    assert result["decision"]["scientific_screen_authorized"] is True
    assert result["decision"]["five_tpp_authorized"] is False
    assert set(result["candidates"]) == set(MULTIPLIERS)
    for slug, candidate in result["candidates"].items():
        assert candidate["config_sha256"] == sha256(OUTPUTS[slug])
        assert candidate["passed"] is True
        assert candidate["native_extension_loaded"] is True
        assert candidate["all_logged_losses_finite"] is True
        assert candidate["mfu_fraction"] >= 0.2
        assert candidate["peak_mib"] < 97887
