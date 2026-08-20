#!/usr/bin/env python3
"""Register the PRO6 350M/0.5TPP all-feedback full-replacement screen."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
ARTIFACT_DIR = CONFIG_DIR / "selection_artifacts"
SOURCE_RESULT = (
    ARTIFACT_DIR
    / "124m_full_replacement_cproj_errorfeedback_20tpp_training_result.json"
)
PLAN = ARTIFACT_DIR / "350m_full_replacement_all_feedback_0p5tpp_plan.json"
MULTIPLIERS = {"mult0p50": 0.50, "mult0p75": 0.75, "mult1p00": 1.00}
PARENTS = {
    slug: CONFIG_DIR
    / f"pro6_mai_v3_350m_qk_only_qk64_outputgain_0p5tpp_{slug}.json"
    for slug in MULTIPLIERS
}
OUTPUTS = {
    slug: CONFIG_DIR
    / f"pro6_mai_v3_350m_fullreplacement_all_int8_errorfeedback_0p5tpp_{slug}.json"
    for slug in MULTIPLIERS
}

QK = "attn.c_attn.qk_headwise"
PRO6_ROOT = "/mnt/ssd-data/orj/MappingNetworks"
BASE_LR = 0.0024
BLOCK_SIZE = 4096
PER_ATTENTION_FAMILY_ELEMENTS = 24 * 1024 * 1024
PER_MLP_FAMILY_ELEMENTS = 24 * 1024 * 4096
ALL_AMBIENT_ELEMENTS = 2 * PER_ATTENTION_FAMILY_ELEMENTS + 2 * PER_MLP_FAMILY_ELEMENTS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def family_codec(elements: int) -> dict[str, Any]:
    blocks = elements // BLOCK_SIZE
    return {
        "base": "independent reproducible frozen Gaussian initialization",
        "blocks": blocks,
        "code_bytes": elements,
        "elements": elements,
        "fp16_optimizer_error_feedback_bytes": 2 * elements,
        "fp16_scale_bytes": 2 * blocks,
        "fp32_weight_bytes": 4 * elements,
        "optimizer_error_feedback": "dense_fp16_not_in_model_codec_count",
        "optimizer_momentum": "dense_fp32_not_in_codec_count",
        "persistent_codec_bytes": elements + 2 * blocks,
        "runtime_base": "transient_dense_fp32",
        "runtime_weight": "transient_dense_fp32",
        "storage_ratio": (elements + 2 * blocks) / (4 * elements),
        "storage_reduction": (4 * elements) / (elements + 2 * blocks),
    }


def build(parent: dict[str, Any], slug: str) -> dict[str, Any]:
    multiplier = MULTIPLIERS[slug]
    if parent.get("model_tier") != "350m" or parent.get("planned_tpp") != 0.5:
        raise ValueError(f"{slug} parent is not the registered 350M/0.5TPP rung")
    if parent.get("block_fht_targets") != [QK]:
        raise ValueError(f"{slug} parent is not the sealed QK-only boundary")
    if float(parent["candidate_main_lr_multiplier"]) != multiplier:
        raise ValueError(f"{slug} multiplier mismatch")
    if float(parent["learning_rate"]) != BASE_LR * multiplier:
        raise ValueError(f"{slug} learning-rate mismatch")

    candidate = copy.deepcopy(parent)
    run_name = (
        "pro6_mai_v3_350m_fullreplacement_all_int8_errorfeedback_0p5tpp_"
        f"{slug}"
    )
    candidate.update(
        {
            "block_fht_attn_v_int8_lattice": True,
            "block_fht_attn_v_int8_lattice_block_size": BLOCK_SIZE,
            "block_fht_attn_v_int8_lattice_error_feedback": True,
            "block_fht_attn_v_int8_lattice_seed": 161804,
            "block_fht_attn_cproj_int8_lattice": True,
            "block_fht_attn_cproj_int8_lattice_block_size": BLOCK_SIZE,
            "block_fht_attn_cproj_int8_lattice_error_feedback": True,
            "block_fht_attn_cproj_int8_lattice_seed": 271828,
            "block_fht_mlp_int8_lattice_targets": ["mlp.c_fc", "mlp.c_proj"],
            "block_fht_mlp_int8_lattice_block_size": BLOCK_SIZE,
            "block_fht_mlp_int8_lattice_error_feedback": True,
            "block_fht_mlp_int8_lattice_seed": 314159,
            "selected_lwt_allocation": {
                "generated": [QK],
                "ambient_int8_lattice_with_fp16_feedback": [
                    "attn.c_attn.v",
                    "attn.c_proj",
                    "mlp.c_fc",
                    "mlp.c_proj",
                ],
            },
            "int8_lattice_representation": family_codec(
                PER_ATTENTION_FAMILY_ELEMENTS
            ),
            "mlp_int8_lattice_representation": family_codec(
                2 * PER_MLP_FAMILY_ELEMENTS
            ),
            "full_replacement_state_accounting": {
                "ambient_elements": ALL_AMBIENT_ELEMENTS,
                "equivalent_fp32_weight_bytes": 4 * ALL_AMBIENT_ELEMENTS,
                "int8_code_bytes": ALL_AMBIENT_ELEMENTS,
                "fp16_scale_bytes": 2 * (ALL_AMBIENT_ELEMENTS // BLOCK_SIZE),
                "fp16_feedback_bytes": 2 * ALL_AMBIENT_ELEMENTS,
                "dense_fp32_muon_momentum_retained": True,
                "transient_dense_materialization_retained": True,
                "additional_inference_flops_vs_dense": 0,
            },
            "candidate_scope": (
                "350M/0.5TPP scale-transfer screen of the 124M/20TPP closed "
                "full replacement: QK64 Cayley LWT plus int8 lattices and "
                "causal FP16 sigma-delta feedback for V, attention c_proj, "
                "MLP c_fc, and MLP c_proj."
            ),
            "hpo_stage": "full_replacement_all_feedback_screen_350m_0p5tpp",
            "ladder_role": "screen_only",
            "ladder_slot": slug,
            "learning_rate_transfer_rule": {
                "accepted_350m_dense_main_lr": BASE_LR,
                "candidate_main_lr_multiplier": multiplier,
                "candidate_learning_rate": BASE_LR * multiplier,
                "screen_values": [0.5, 0.75, 1.0],
                "reason": (
                    "The representation changed from QK-only to complete "
                    "replacement, so preserve the registered MAI 0.5TPP LR "
                    "screen instead of copying a winner across operators."
                ),
            },
            "out_dir": f"{PRO6_ROOT}/outputs/{run_name}/scientific",
            "mfu_preflight_certificate": (
                f"{PRO6_ROOT}/outputs/{run_name}/performance_preflight.json"
            ),
            "monitoring_policy": (
                "Expected one-to-two-hour screen: one persistent terminal-only "
                "watchdog; completion, actionable error/stall, or monitor "
                "degradation invokes @Codex with the exact next action."
            ),
            "practical_equivalence_policy": (
                "At the matching LR slot, require terminal validation CE no "
                "more than +0.0200 above the sealed 350M QK-only control. Rank "
                "all stable candidates only after all three terminal results."
            ),
            "recipe_resolution_dependency": (
                "sealed 124M/20TPP all-feedback zero-gap result and sealed "
                "350M/0.5TPP QK-only LR screen"
            ),
            "recipe_resolution_stage": "full_replacement_scale_transfer_screen",
            "registered_resume_protocol": (
                "atomic exact-resume checkpoint with RNG state, int8 codes, "
                "FP16 scales, dense FP32 Muon momentum, and 96 FP16 compression "
                "residuals across all 24 layers"
            ),
            "resolved_from_template": str(PARENTS[slug].relative_to(ROOT)),
            "scale_transfer_source_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "scale_transfer_source_result_sha256": sha256(SOURCE_RESULT),
            "launch_ready": True,
        }
    )
    return candidate


def build_plan(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qk_terminal = {"mult0p50": 4.2324, "mult0p75": 4.0437, "mult1p00": 3.9511}
    dense_terminal = {"lr16e4": 4.4452, "lr20e4": 4.2984, "lr24e4": 4.1784}
    return {
        "schema_version": "mai_350m_full_replacement_all_feedback_0p5tpp_plan_v1",
        "registered_at": "2026-08-20",
        "status": "preregistered_pending_tests_and_exact_config_mfu",
        "question": (
            "Does the all-feedback complete replacement that closed 124M/20TPP "
            "transfer to 350M without changing its mechanism?"
        ),
        "theory": {
            "confirmed_124m_mechanism": (
                "Each ambient projected family needs causal temporal state so "
                "sub-quantum Muon motion is accumulated rather than discarded."
            ),
            "scale_transfer_change": "model width/depth and LR slot only",
            "representation_change": "none",
            "excluded_claims": [
                "multi-seed closure",
                "optimizer-state compression",
                "inference FLOP reduction",
                "690M transfer",
            ],
        },
        "ladder": {
            "model_tier": "350m",
            "planned_tpp": 0.5,
            "max_iters": 677,
            "scheduled_tokens": 177471488,
            "candidate_multipliers": list(MULTIPLIERS.values()),
            "candidate_learning_rates": [
                BASE_LR * value for value in MULTIPLIERS.values()
            ],
            "ranking_rule": (
                "stable ascending terminal validation CE after all three exact "
                "configs finish; hard reject nonfinite or failed runs"
            ),
            "promotion_rule": (
                "only the immutable top1/top2 ranking may authorize 5TPP; no "
                "larger tier is authorized from this screen"
            ),
        },
        "frozen_comparators": {
            "qk_only_same_slot_terminal_validation_ce": qk_terminal,
            "ordinary_dense_350m_0p5tpp_terminal_validation_ce": dense_terminal,
        },
        "frozen_gate": {
            "maximum_delta_to_same_slot_qk_only_ce": 0.0200,
            "require_clean_exit": True,
            "require_finite_terminal_evaluation": True,
            "threshold_changes_after_measurement": False,
            "pass_classification": "PASS_FULL_REPLACEMENT_SCALE_TRANSFER_350M_0P5TPP",
            "fail_classification": "FAIL_FULL_REPLACEMENT_SCALE_TRANSFER_350M_0P5TPP",
        },
        "state_accounting": configs["mult1p00"][
            "full_replacement_state_accounting"
        ],
        "candidates": {
            slug: {
                "config": str(OUTPUTS[slug].relative_to(ROOT)),
                "config_sha256": sha256(OUTPUTS[slug]),
                "candidate_main_lr_multiplier": multiplier,
                "learning_rate": configs[slug]["learning_rate"],
                "qk_only_terminal_validation_ce": qk_terminal[slug],
                "maximum_terminal_validation_ce": qk_terminal[slug] + 0.0200,
                "out_dir": configs[slug]["out_dir"],
                "mfu_certificate": configs[slug]["mfu_preflight_certificate"],
            }
            for slug, multiplier in MULTIPLIERS.items()
        },
        "immutable_evidence": {
            "confirmed_124m_result": {
                "path": str(SOURCE_RESULT.relative_to(ROOT)),
                "sha256": sha256(SOURCE_RESULT),
            },
            **{
                f"qk_parent_{slug}": {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
                for slug, path in PARENTS.items()
            },
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.20,
            "exact_config_required": True,
            "foreground_polling": True,
            "watchdog": False,
            "all_candidates_must_pass_before_scientific_launch": True,
        },
        "monitoring": {
            "terminal_only": True,
            "callbacks": ["completion", "error", "stall", "monitor_degraded"],
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "agent_mention": "@Codex",
            "terminal_action": (
                "seal the candidate against its same-slot QK control and launch "
                "the next preregistered screen slot; rank only when all finish"
            ),
        },
        "resource_admission": {
            "host": "PRO6",
            "gpu": 0,
            "project_cap_gib": 256,
            "minimum_post_admission_headroom_gib": 8,
            "archive_completed checkpoints before subsequent slots": True,
        },
    }


def main() -> None:
    source = load(SOURCE_RESULT)
    if source.get("classification") != (
        "FULL_REPLACEMENT_CPROJ_TEMPORAL_REPAIR_CONFIRMED_124M_20TPP"
    ):
        raise ValueError("the 124M full-replacement closure is not sealed")
    configs: dict[str, dict[str, Any]] = {}
    for slug, output in OUTPUTS.items():
        configs[slug] = build(load(PARENTS[slug]), slug)
        output.write_text(json.dumps(configs[slug], indent=2, sort_keys=True) + "\n")
        print(output.relative_to(ROOT))
    PLAN.write_text(json.dumps(build_plan(configs), indent=2, sort_keys=True) + "\n")
    print(PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
