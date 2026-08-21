from __future__ import annotations

import hashlib
import json

import pytest

from examples.nanogpt.make_pro6_full_replacement_all_feedback_350m_5tpp_configs import (
    OUTPUTS,
    MFU_RESULT,
    RUN_METADATA,
    TOP1_RESULT,
    TOP2_REFRESHED_MFU_RESULT,
    TOP2_RUN_METADATA,
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


def test_plan_materializes_only_ranked_top_two_and_authorizes_top1_after_both_mfu_gates():
    plan = load(PLAN)
    assert set(OUTPUTS) == {"top1", "top2"}
    assert plan["execution_order"] == ["top1", "top2"]
    assert plan["excluded_candidates"] == ["mult0p50"]
    assert plan["performance_gate"]["foreground_polling"] is True
    assert plan["performance_gate"]["watchdog"] is False
    assert plan["performance_gate"]["both_configs_must_pass_before_first_launch"] is True
    assert plan["status"] == "top2_running"
    assert plan["performance_gate"]["launch_authorized"] is True
    assert plan["performance_gate"]["top2_launch_authorized"] is True
    assert plan["performance_gate"]["mfu_result"] == {
        "path": str(MFU_RESULT.relative_to(MFU_RESULT.parents[4])),
        "sha256": sha256(MFU_RESULT),
    }
    assert plan["execution_state"] == {
        "path": str(RUN_METADATA.relative_to(RUN_METADATA.parents[4])),
        "sha256": sha256(RUN_METADATA),
    }
    assert plan["top1_terminal_result"] == {
        "path": str(TOP1_RESULT.relative_to(TOP1_RESULT.parents[4])),
        "sha256": sha256(TOP1_RESULT),
    }
    assert plan["performance_gate"]["refreshed_top2_mfu_result"] == {
        "path": str(TOP2_REFRESHED_MFU_RESULT.relative_to(TOP2_REFRESHED_MFU_RESULT.parents[4])),
        "sha256": sha256(TOP2_REFRESHED_MFU_RESULT),
    }
    assert plan["top2_execution_state"] == {
        "path": str(TOP2_RUN_METADATA.relative_to(TOP2_RUN_METADATA.parents[4])),
        "sha256": sha256(TOP2_RUN_METADATA),
    }
    assert plan["immutable_ranking"]["sha256"] == sha256(RANKING)
    for slot in SELECTIONS:
        assert plan["confirmations"][slot]["config_sha256"] == sha256(OUTPUTS[slot])
    assert plan["monitoring"]["agent_mention"] == "@Codex"
    assert plan["monitoring"]["heartbeat_minutes"] == 90


def test_sealed_mfu_result_passes_both_exact_configs_and_only_authorizes_top1():
    result = load(MFU_RESULT)
    assert result["decision"]["classification"] == "PASS_BOTH_EXACT_CONFIG_MFU_GATES"
    assert result["decision"]["top1_scientific_launch_authorized"] is True
    assert result["decision"]["top2_scientific_launch_authorized"] is False
    for slot in SELECTIONS:
        gate = result["confirmations"][slot]
        assert gate["config_sha256"] == sha256(OUTPUTS[slot])
        assert gate["passed"] is True
        assert gate["mfu_fraction"] >= 0.20
        assert gate["all_logged_losses_finite"] is True
        assert gate["native_block_fht_extension_loaded"] is True


def test_top1_run_metadata_seals_result_and_authorizes_top2():
    metadata = load(RUN_METADATA)
    assert metadata["state"] == "finished"
    assert metadata["config_sha256"] == sha256(OUTPUTS["top1"])
    assert metadata["source_commit"] == "c947db1b170b3d7a9031cc537899731ee73aaebd"
    assert metadata["watchdog"]["progress_milestones"] == [20, 50, 80, 100]
    assert metadata["watchdog"]["heartbeat_minutes"] == 90
    assert metadata["terminal"]["frozen_gate_passed"] is True
    assert metadata["result"]["sha256"] == sha256(TOP1_RESULT)
    assert metadata["top2_refreshed_mfu_result"]["sha256"] == sha256(
        TOP2_REFRESHED_MFU_RESULT
    )
    assert metadata["top2_launch_authorized"] is True


def test_top2_refreshed_mfu_gate_matches_exact_config_and_driver_repair():
    result = load(TOP2_REFRESHED_MFU_RESULT)
    assert result["classification"] == "PASS_TOP2_EXACT_CONFIG_MFU_AFTER_DRIVER_COMPAT_REPAIR"
    assert result["config"]["sha256"] == sha256(OUTPUTS["top2"])
    assert result["preflight"]["mfu_fraction"] >= 0.20
    assert result["preflight"]["all_logged_losses_finite"] is True
    assert result["decision"]["top2_launch_authorized"] is True


def test_top2_run_metadata_pins_clean_launch_and_watchdog():
    metadata = load(TOP2_RUN_METADATA)
    assert metadata["state"] == "running"
    assert metadata["source_commit"] == "cb7c54c3d6b9de4d0acabb86b58c7b501b096241"
    assert metadata["config_sha256"] == sha256(OUTPUTS["top2"])
    assert metadata["refreshed_mfu_certificate_sha256"] == load(
        TOP2_REFRESHED_MFU_RESULT
    )["artifacts"]["certificate"]["sha256"]
    assert metadata["watchdog"]["progress_milestones"] == [20, 50, 80, 100]
    assert metadata["watchdog"]["initial_probe"]["alive"] is True
    assert metadata["driver_compatibility"]["library_path_used_by_watchdog_gpu_probe"] is True
