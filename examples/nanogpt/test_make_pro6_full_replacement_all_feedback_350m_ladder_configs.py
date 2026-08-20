from __future__ import annotations

import hashlib
import json

import pytest

from examples.nanogpt.make_pro6_full_replacement_all_feedback_350m_ladder_configs import (
    ALL_AMBIENT_ELEMENTS,
    BASE_LR,
    MULTIPLIERS,
    OUTPUTS,
    PARENTS,
    PLAN,
    QK,
    SOURCE_RESULT,
    build,
)


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("slug,multiplier", MULTIPLIERS.items())
def test_complete_replacement_preserves_registered_350m_rung(
    slug: str, multiplier: float
) -> None:
    parent = load(PARENTS[slug])
    candidate = load(OUTPUTS[slug])
    assert candidate == build(parent, slug)
    for key in (
        "n_layer",
        "n_head",
        "n_embd",
        "batch_size",
        "gradient_accumulation_steps",
        "max_iters",
        "planned_tokens",
        "scheduled_tokens",
        "eval_batch_size",
        "eval_iters",
        "fixed_eval_index_spec_sha256",
        "data_manifest_sha256",
        "model_seed",
    ):
        assert candidate[key] == parent[key]
    assert candidate["learning_rate"] == pytest.approx(BASE_LR * multiplier)
    assert candidate["candidate_main_lr_multiplier"] == multiplier
    assert candidate["block_fht_targets"] == [QK]


@pytest.mark.parametrize("slug", MULTIPLIERS)
def test_every_ambient_family_uses_lattice_and_temporal_feedback(slug: str) -> None:
    candidate = load(OUTPUTS[slug])
    assert candidate["block_fht_attn_v_int8_lattice"] is True
    assert candidate["block_fht_attn_v_int8_lattice_error_feedback"] is True
    assert candidate["block_fht_attn_cproj_int8_lattice"] is True
    assert candidate["block_fht_attn_cproj_int8_lattice_error_feedback"] is True
    assert candidate["block_fht_mlp_int8_lattice_targets"] == [
        "mlp.c_fc",
        "mlp.c_proj",
    ]
    assert candidate["block_fht_mlp_int8_lattice_error_feedback"] is True
    state = candidate["full_replacement_state_accounting"]
    assert state["ambient_elements"] == ALL_AMBIENT_ELEMENTS
    assert state["fp16_feedback_bytes"] == 2 * ALL_AMBIENT_ELEMENTS
    assert state["additional_inference_flops_vs_dense"] == 0


def test_plan_freezes_scale_transfer_screen_and_exact_mfu_gates() -> None:
    plan = load(PLAN)
    assert plan["status"] == "preregistered_pending_tests_and_exact_config_mfu"
    assert plan["frozen_gate"]["maximum_delta_to_same_slot_qk_only_ce"] == 0.02
    assert plan["performance_gate"]["exact_config_required"] is True
    assert plan["performance_gate"]["foreground_polling"] is True
    assert plan["performance_gate"]["watchdog"] is False
    assert plan["monitoring"]["terminal_only"] is True
    assert plan["monitoring"]["agent_mention"] == "@Codex"
    assert plan["immutable_evidence"]["confirmed_124m_result"]["sha256"] == sha256(
        SOURCE_RESULT
    )
    for slug in MULTIPLIERS:
        assert plan["candidates"][slug]["config_sha256"] == sha256(OUTPUTS[slug])
        assert plan["candidates"][slug]["maximum_terminal_validation_ce"] == pytest.approx(
            plan["candidates"][slug]["qk_only_terminal_validation_ce"] + 0.02
        )
