#!/usr/bin/env python3
"""Materialize the ranked 350M/5TPP all-feedback full replacements."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
ARTIFACT_DIR = CONFIG_DIR / "selection_artifacts"
RANKING = ARTIFACT_DIR / "350m_full_replacement_all_feedback_0p5tpp_ranking.json"
PLAN = ARTIFACT_DIR / "350m_full_replacement_all_feedback_5tpp_plan.json"
SELECTIONS = {"top1": "mult1p00", "top2": "mult0p75"}
PARENTS = {
    "top1": CONFIG_DIR / "pro6_mai_v3_350m_qk_only_qk64_outputgain_5tpp_top1_mult1p00.json",
    "top2": CONFIG_DIR / "pro6_mai_v3_350m_qk_only_qk64_outputgain_5tpp_top2_mult0p75.json",
}
SCREENS = {
    slug: CONFIG_DIR / f"pro6_mai_v3_350m_fullreplacement_all_int8_errorfeedback_0p5tpp_{slug}.json"
    for slug in SELECTIONS.values()
}
OUTPUTS = {
    "top1": CONFIG_DIR / "pro6_mai_v3_350m_fullreplacement_all_int8_errorfeedback_5tpp_top1_mult1p00.json",
    "top2": CONFIG_DIR / "pro6_mai_v3_350m_fullreplacement_all_int8_errorfeedback_5tpp_top2_mult0p75.json",
}
QK = "attn.c_attn.qk_headwise"
PRO6_ROOT = "/mnt/ssd-data/orj/MappingNetworks"

MECHANISM_FIELDS = (
    "block_fht_attn_v_int8_lattice",
    "block_fht_attn_v_int8_lattice_block_size",
    "block_fht_attn_v_int8_lattice_error_feedback",
    "block_fht_attn_v_int8_lattice_seed",
    "block_fht_attn_cproj_int8_lattice",
    "block_fht_attn_cproj_int8_lattice_block_size",
    "block_fht_attn_cproj_int8_lattice_error_feedback",
    "block_fht_attn_cproj_int8_lattice_seed",
    "block_fht_mlp_int8_lattice_targets",
    "block_fht_mlp_int8_lattice_block_size",
    "block_fht_mlp_int8_lattice_error_feedback",
    "block_fht_mlp_int8_lattice_seed",
    "selected_lwt_allocation",
    "int8_lattice_representation",
    "mlp_int8_lattice_representation",
    "full_replacement_state_accounting",
    "registered_resume_protocol",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_ranking(ranking: dict[str, Any]) -> None:
    decision = ranking.get("decision", {})
    if decision.get("classification") != "PASS_COMPLETE_IMMUTABLE_TOP2_RANKING":
        raise ValueError("complete-replacement 0.5TPP ranking is not sealed")
    if decision.get("five_tpp_configs_may_be_materialized") is not True:
        raise ValueError("ranking does not authorize 5TPP materialization")
    if decision.get("five_tpp_runs_authorized") is not False:
        raise ValueError("ranking must leave training blocked pending exact MFU gates")
    if decision.get("top1") != SELECTIONS["top1"]:
        raise ValueError("top1 mismatch")
    if decision.get("top2") != SELECTIONS["top2"]:
        raise ValueError("top2 mismatch")


def build(parent: dict[str, Any], screen: dict[str, Any], slot: str) -> dict[str, Any]:
    slug = SELECTIONS[slot]
    ranking = load(RANKING)
    validate_ranking(ranking)
    if parent.get("model_tier") != "350m" or float(parent.get("planned_tpp", 0)) != 5.0:
        raise ValueError(f"{slot} parent is not 350M/5TPP")
    if parent.get("ladder_slot") != slot:
        raise ValueError(f"{slot} parent slot mismatch")
    if screen.get("ladder_slot") != slug or float(screen.get("planned_tpp", 0)) != 0.5:
        raise ValueError(f"{slot} screen mismatch")
    if float(parent["learning_rate"]) != float(screen["learning_rate"]):
        raise ValueError(f"{slot} learning-rate transfer mismatch")

    candidate = copy.deepcopy(parent)
    for field in MECHANISM_FIELDS:
        candidate[field] = copy.deepcopy(screen[field])
    run_name = OUTPUTS[slot].stem
    candidate.update(
        {
            "candidate_scope": (
                "350M/5TPP confirmation of the immutable all-feedback complete "
                "replacement: QK64 Cayley LWT plus ambient int8 lattices with "
                "causal FP16 sigma-delta feedback for V, attention c_proj, MLP "
                "c_fc, and MLP c_proj."
            ),
            "confirmation_source": "immutable three-arm 350M/0.5TPP full-replacement ranking",
            "hpo_stage": "full_replacement_all_feedback_confirmation_350m_5tpp",
            "ladder_role": "confirmation_registered",
            "ladder_slot": slot,
            "learning_rate_transfer_rule": {
                "source_screen_slug": slug,
                "source_screen_learning_rate": screen["learning_rate"],
                "candidate_main_lr_multiplier": screen["candidate_main_lr_multiplier"],
                "rule": "preserve the immutable 0.5TPP top-two learning rate at 5TPP",
            },
            "monitoring_policy": (
                "Expected seven-to-eight-hour confirmation: one persistent "
                "watchdog with 20%, 50%, 80%, completion, actionable failure/stall, "
                "monitor degradation, and one resettable 90-minute heartbeat; "
                "every callback invokes @Codex with a concrete continuation prompt."
            ),
            "mfu_preflight_certificate": f"{PRO6_ROOT}/outputs/{run_name}/performance_preflight.json",
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.20,
            "out_dir": f"{PRO6_ROOT}/outputs/{run_name}/scientific",
            "practical_equivalence_policy": (
                "At the matching LR slot, require terminal fixed-validation CE no "
                "more than +0.0200 above the sealed 350M QK-only 5TPP control."
            ),
            "prelaunch_provenance_requirements": (
                "record clean commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed-eval "
                "digest, and fresh exact-config foreground-polled MFU certificate"
            ),
            "recipe_resolution_dependency": "immutable completed 350M full-replacement 0.5TPP ranking",
            "recipe_resolution_required": False,
            "recipe_resolution_stage": "full_replacement_all_feedback_confirmation_5tpp",
            "resolved_from_template": str(PARENTS[slot].relative_to(ROOT)),
            "scale_transfer_source_config": str(SCREENS[slug].relative_to(ROOT)),
            "scale_transfer_source_config_sha256": sha256(SCREENS[slug]),
            "screen_only": False,
            "screen_only_resolution": None,
            "zero_point_five_tpp_ranking_artifact": str(RANKING.relative_to(ROOT)),
            "zero_point_five_tpp_ranking_artifact_sha256": sha256(RANKING),
            "zero_point_five_tpp_ranking_slot": slot,
        }
    )
    return candidate


def build_plan(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "mai_350m_full_replacement_all_feedback_5tpp_plan_v1",
        "registered_at": "2026-08-21",
        "status": "materialized_pending_both_exact_config_mfu_gates",
        "scope": "top-two 350M/5TPP all-feedback complete-replacement confirmations",
        "immutable_ranking": {"path": str(RANKING.relative_to(ROOT)), "sha256": sha256(RANKING)},
        "schedule": {"planned_tpp": 5.0, "planned_tokens": 1772999680, "scheduled_tokens": 1773142016, "max_iters": 6764, "tokens_per_iter": 262144, "eval_interval": 1691, "warmup_iters": 67},
        "confirmations": {
            slot: {
                "screen_slug": SELECTIONS[slot],
                "config_path": str(OUTPUTS[slot].relative_to(ROOT)),
                "config_sha256": sha256(OUTPUTS[slot]),
                "learning_rate": configs[slot]["learning_rate"],
                "matched_qk_only_terminal_validation_ce": 3.0304 if slot == "top1" else 3.0645,
                "maximum_terminal_validation_ce": 3.0504 if slot == "top1" else 3.0845,
                "mfu_certificate": configs[slot]["mfu_preflight_certificate"],
            }
            for slot in SELECTIONS
        },
        "excluded_candidates": ["mult0p50"],
        "performance_gate": {"minimum_mfu_fraction": 0.20, "exact_config_required": True, "foreground_polling": True, "watchdog": False, "both_configs_must_pass_before_first_launch": True, "launch_authorized": False},
        "execution_order": ["top1", "top2"],
        "monitoring": {"expected_duration_hours_per_candidate": [7, 8], "progress_callbacks": [0.20, 0.50, 0.80, 1.00], "heartbeat_minutes": 90, "heartbeat_resets_on_progress": True, "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test", "agent_mention": "@Codex"},
        "resource_admission": {"host": "PRO6", "gpu": 0, "project_cap_gib": 256, "minimum_post_admission_headroom_gib": 8, "archive_completed_checkpoint_before_next_slot": True},
        "frozen_gate": {"maximum_delta_to_same_slot_qk_only_ce": 0.02, "require_clean_exit": True, "require_finite_terminal_evaluation": True, "threshold_changes_after_measurement": False},
        "scope_limits": ["No 690M promotion", "No multi-seed closure claim", "No optimizer-state compression claim", "No inference-FLOP reduction claim"],
    }


def main() -> None:
    validate_ranking(load(RANKING))
    configs: dict[str, dict[str, Any]] = {}
    for slot in SELECTIONS:
        configs[slot] = build(load(PARENTS[slot]), load(SCREENS[SELECTIONS[slot]]), slot)
        OUTPUTS[slot].write_text(json.dumps(configs[slot], indent=2, sort_keys=True) + "\n")
        print(OUTPUTS[slot].relative_to(ROOT))
    PLAN.write_text(json.dumps(build_plan(configs), indent=2, sort_keys=True) + "\n")
    print(PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
