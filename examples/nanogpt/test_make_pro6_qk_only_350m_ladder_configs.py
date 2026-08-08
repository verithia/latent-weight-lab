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
MULT1P00_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "350m_qk_only_functional_lwt_0p5tpp_mult1p00_result.json"
)
MULT0P75_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "350m_qk_only_functional_lwt_0p5tpp_mult0p75_result.json"
)
MULT0P50_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "350m_qk_only_functional_lwt_0p5tpp_mult0p50_result.json"
)
RANKING = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "350m_qk_only_functional_lwt_0p5tpp_ranking.json"
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


def test_mult1p00_terminal_result_is_sealed_but_does_not_promote_early() -> None:
    result = load(MULT1P00_RESULT)
    assert result["schema_version"] == (
        "mai_350m_qk_only_functional_lwt_0p5tpp_candidate_result_v1"
    )
    assert result["immutable_inputs"]["config"]["sha256"] == sha256(
        OUTPUTS["mult1p00"]
    )
    assert result["run"]["source_commit"] == (
        "6a5b9e08e7aa7d5b8d5a75bd1f90324d7bdbb3a7"
    )
    assert result["run"]["classification"] == "clean"
    assert result["run"]["exit_code"] == 0
    assert [row["step"] for row in result["fixed_evaluations"]] == [
        0,
        170,
        340,
        510,
        677,
    ]
    assert result["fixed_evaluations"][-1]["validation_ce"] == pytest.approx(
        3.9511
    )
    verification = result["checkpoint_verification"]
    assert verification["metadata_consistent"] is True
    assert verification["all_model_tensors_finite"] is True
    assert verification["all_optimizer_tensors_finite"] is True
    assert verification["rng_state_present"] is True
    assert result["comparisons"]["same_horizon_full_attention_replacement"][
        "candidate_minus_reference_ce"
    ] == pytest.approx(-0.4118)
    assert result["decision"]["next_candidate"] == "mult0p75"
    assert result["decision"]["next_candidate_authorized"] is True
    assert result["decision"]["five_tpp_authorized"] is False


def test_mult0p75_terminal_result_is_sealed_and_final_screen_is_next() -> None:
    result = load(MULT0P75_RESULT)
    assert result["schema_version"] == (
        "mai_350m_qk_only_functional_lwt_0p5tpp_candidate_result_v1"
    )
    assert result["immutable_inputs"]["config"]["sha256"] == sha256(
        OUTPUTS["mult0p75"]
    )
    assert result["run"]["source_commit"] == (
        "4e27f6dc664e4078dfb542d6246816657dbbb774"
    )
    assert result["run"]["classification"] == "clean"
    assert result["run"]["exit_code"] == 0
    assert [row["step"] for row in result["fixed_evaluations"]] == [
        0,
        170,
        340,
        510,
        677,
    ]
    assert result["fixed_evaluations"][-1]["validation_ce"] == pytest.approx(
        4.0437
    )
    verification = result["checkpoint_verification"]
    assert verification["metadata_consistent"] is True
    assert verification["all_model_tensors_finite"] is True
    assert verification["all_optimizer_tensors_finite"] is True
    assert verification["rng_state_present"] is True
    assert result["comparisons"]["completed_qk_only_mult1p00"][
        "candidate_minus_reference_ce"
    ] == pytest.approx(0.0926)
    assert result["comparisons"]["same_horizon_full_attention_replacement"][
        "candidate_minus_reference_ce"
    ] == pytest.approx(-0.4637)
    assert result["decision"]["next_candidate"] == "mult0p50"
    assert result["decision"]["next_candidate_authorized"] is True
    assert result["decision"]["five_tpp_authorized"] is False


def test_mult0p50_terminal_result_seals_complete_ranking() -> None:
    result = load(MULT0P50_RESULT)
    assert result["immutable_inputs"]["config"]["sha256"] == sha256(
        OUTPUTS["mult0p50"]
    )
    assert result["run"]["source_commit"] == (
        "80d92c7ad2950e13d4d98fd7a4272546b860967f"
    )
    assert result["run"]["classification"] == "clean"
    assert result["run"]["exit_code"] == 0
    assert result["fixed_evaluations"][-1]["validation_ce"] == pytest.approx(
        4.2324
    )
    verification = result["checkpoint_verification"]
    assert verification["metadata_consistent"] is True
    assert verification["all_model_tensors_finite"] is True
    assert verification["all_optimizer_tensors_finite"] is True
    assert result["decision"]["ranking_authorized"] is True
    assert result["decision"]["top1"] == "mult1p00"
    assert result["decision"]["top2"] == "mult0p75"
    assert result["decision"]["five_tpp_runs_authorized"] is False


def test_complete_ranking_pins_all_results_and_excludes_third_candidate() -> None:
    ranking = load(RANKING)
    assert ranking["decision"]["classification"] == (
        "PASS_COMPLETE_IMMUTABLE_TOP2_RANKING"
    )
    assert [row["slug"] for row in ranking["ranked_candidates"]] == [
        "mult1p00",
        "mult0p75",
        "mult0p50",
    ]
    assert [row["terminal_validation_ce"] for row in ranking["ranked_candidates"]] == [
        3.9511,
        4.0437,
        4.2324,
    ]
    assert ranking["decision"]["rejected_from_5tpp"] == ["mult0p50"]
    for slug, path in {
        "mult1p00": MULT1P00_RESULT,
        "mult0p75": MULT0P75_RESULT,
        "mult0p50": MULT0P50_RESULT,
    }.items():
        assert ranking["immutable_inputs"][slug]["result_sha256"] == sha256(path)
