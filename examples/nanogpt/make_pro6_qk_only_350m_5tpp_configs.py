#!/usr/bin/env python3
"""Materialize only the immutable top-two 350M QK-only 5TPP confirmations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
ARTIFACT_DIR = CONFIG_DIR / "selection_artifacts"
RANKING = ARTIFACT_DIR / "350m_qk_only_functional_lwt_0p5tpp_ranking.json"
PLAN = ARTIFACT_DIR / "350m_qk_only_functional_lwt_5tpp_plan.json"

QK = "attn.c_attn.qk_headwise"
PRO6_ROOT = "/mnt/ssd-data/orj/MappingNetworks"
PRO6_SYMLINK = "/home/pro6000-9980x/MappingNetworks"
HEAD_DIM = 64

SELECTIONS = {
    "top1": "mult1p00",
    "top2": "mult0p75",
}
PARENTS = {
    "top1": CONFIG_DIR / "y400_mai_v2_350m_fullattn_blockfht_5tpp_top1.json",
    "top2": CONFIG_DIR / "y400_mai_v2_350m_fullattn_blockfht_5tpp_top2.json",
}
SCREENS = {
    slug: CONFIG_DIR / f"pro6_mai_v3_350m_qk_only_qk64_outputgain_0p5tpp_{slug}.json"
    for slug in SELECTIONS.values()
}
OUTPUTS = {
    "top1": CONFIG_DIR / "pro6_mai_v3_350m_qk_only_qk64_outputgain_5tpp_top1_mult1p00.json",
    "top2": CONFIG_DIR / "pro6_mai_v3_350m_qk_only_qk64_outputgain_5tpp_top2_mult0p75.json",
}

CAYLEY_FIELDS = (
    "block_fht_attn_cayley_bilateral_targets",
    "block_fht_attn_cayley_lr_scale",
    "block_fht_attn_cayley_output_targets",
    "block_fht_attn_cayley_rank",
    "block_fht_attn_cayley_ranks",
    "block_fht_attn_cayley_scale",
    "block_fht_attn_cayley_seed",
    "block_fht_attn_cayley_targets",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_ranking(ranking: dict[str, Any]) -> None:
    decision = ranking.get("decision", {})
    if decision.get("classification") != "PASS_COMPLETE_IMMUTABLE_TOP2_RANKING":
        raise ValueError("350M QK-only ranking is not complete")
    if decision.get("five_tpp_configs_may_be_materialized") is not True:
        raise ValueError("ranking does not authorize 5TPP config materialization")
    if decision.get("authorized_5tpp_config_count") != 2:
        raise ValueError("ranking must authorize exactly two 5TPP configs")
    if decision.get("top1") != SELECTIONS["top1"]:
        raise ValueError("top1 selection mismatch")
    if decision.get("top2") != SELECTIONS["top2"]:
        raise ValueError("top2 selection mismatch")
    if set(SELECTIONS.values()) & set(decision.get("rejected_from_5tpp", [])):
        raise ValueError("selected candidate also appears in rejection set")


def build(
    parent: dict[str, Any],
    screen: dict[str, Any],
    ranking: dict[str, Any],
    slot: str,
) -> dict[str, Any]:
    _validate_ranking(ranking)
    slug = SELECTIONS[slot]
    if parent.get("model_tier") != "350m" or float(parent.get("planned_tpp", 0.0)) != 5.0:
        raise ValueError(f"{slot} parent is not a 350M/5TPP confirmation")
    if parent.get("confirmation_slot") != slot:
        raise ValueError(f"{slot} parent confirmation slot mismatch")
    if screen.get("ladder_slot") != slug or float(screen.get("planned_tpp", 0.0)) != 0.5:
        raise ValueError(f"{slot} screen source mismatch")
    if float(parent["learning_rate"]) != float(screen["learning_rate"]):
        raise ValueError(f"{slot} learning-rate transfer mismatch")
    if int(parent["n_embd"]) // int(parent["n_head"]) != HEAD_DIM:
        raise ValueError("350M parent violates the 64-dimensional head invariant")

    candidate = copy.deepcopy(parent)
    for field in CAYLEY_FIELDS:
        candidate[field] = copy.deepcopy(screen[field])
    candidate.update(
        {
            "block_fht_targets": [QK],
            "block_fht_output_gain_targets": [QK],
            "candidate_learning_rate_resolution": (
                f"immutable 350M QK-only 0.5TPP {slot}: {slug}"
            ),
            "candidate_scope": (
                "350M 5TPP confirmation of the immutable QK-only functional-LWT "
                "top two: generate headwise Q/K only; keep V, attention c_proj, "
                "and the full MLP dense and Muon-trained."
            ),
            "confirmation_source": (
                "immutable 350M QK-only 0.5TPP ranking pinned by SHA-256"
            ),
            "data_dir": f"{PRO6_SYMLINK}/data/finewebedu_20b",
            "data_staging_policy": (
                "PRO6 resident immutable FineWeb-Edu 20B dataset; verify manifest "
                "and runtime fixed-index digest before launch"
            ),
            "data_staging_source": f"{PRO6_ROOT}/data/finewebedu_20b",
            "head_dimension_scaling_rule": copy.deepcopy(
                screen["head_dimension_scaling_rule"]
            ),
            "hpo_stage": "attention_qk_only_functional_lwt_confirmation_350m_5tpp",
            "ladder_role": "confirmation_registered",
            "ladder_slot": slot,
            "learning_rate_transfer_rule": {
                "source_screen_slug": slug,
                "source_screen_learning_rate": screen["learning_rate"],
                "candidate_main_lr_multiplier": screen[
                    "candidate_main_lr_multiplier"
                ],
                "rule": "preserve the immutable 0.5TPP top-two learning rate at 5TPP",
            },
            "monitoring_policy": (
                "Expected seven-to-eight-hour confirmation: one persistent watchdog; "
                "callbacks at 20%, 50%, 80%, clean completion, error, stall, or monitor "
                "degradation; one 90-minute heartbeat reset by any progress callback; "
                "every callback invokes @Codex with a concrete continuation prompt."
            ),
            "mfu_preflight_certificate": None,
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.20,
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "automatic_larger_rung_promotion": False,
                "reason": (
                    "This is the preregistered 5TPP confirmation selected by the "
                    "completed three-arm 350M QK-only screen."
                ),
                "recorded_at": "2026-08-09",
                "scope": f"350M/5TPP QK-only functional-LWT {slot}",
            },
            "out_dir": (
                f"{PRO6_ROOT}/outputs/pro6_mai_v3_350m_qk_only_ladder/"
                f"{OUTPUTS[slot].stem}"
            ),
            "practical_equivalence_policy": (
                "Compare the two stable 5TPP terminal fixed-validation CEs. Select "
                "the lower loss only when the gap exceeds 0.02; otherwise retain "
                "the 0.5TPP leader and record PRACTICAL_TIE."
            ),
            "prelaunch_provenance_requirements": (
                "record clean commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed-eval "
                "digest, and fresh exact-config foreground-polled MFU certificate"
            ),
            "provisional_zero_point_five_tpp_ranking_artifact": None,
            "provisional_zero_point_five_tpp_ranking_artifact_sha256": None,
            "qk_only_zero_point_five_tpp_ranking_artifact": str(
                RANKING.relative_to(ROOT)
            ),
            "qk_only_zero_point_five_tpp_ranking_artifact_sha256": sha256(RANKING),
            "recipe_resolution_dependency": (
                "immutable completed 350M QK-only 0.5TPP ranking pinned by SHA-256"
            ),
            "recipe_resolution_required": False,
            "recipe_resolution_stage": "qk_only_functional_lwt_confirmation_5tpp",
            "resolved_from_template": str(PARENTS[slot].relative_to(ROOT)),
            "scale_transfer_source_config": str(SCREENS[slug].relative_to(ROOT)),
            "scale_transfer_source_config_sha256": sha256(SCREENS[slug]),
            "screen_only": False,
            "screen_only_resolution": None,
            "selected_lwt_allocation": copy.deepcopy(
                screen["selected_lwt_allocation"]
            ),
            "zero_point_five_tpp_ranking_artifact": str(RANKING.relative_to(ROOT)),
            "zero_point_five_tpp_ranking_artifact_required": True,
            "zero_point_five_tpp_ranking_artifact_schema": ranking[
                "schema_version"
            ],
            "zero_point_five_tpp_ranking_artifact_sha256": sha256(RANKING),
            "zero_point_five_tpp_ranking_hpo_stage": (
                "attention_qk_only_functional_lwt_screen_350m_0p5tpp"
            ),
            "zero_point_five_tpp_ranking_method": "qk_only_functional_lwt",
            "zero_point_five_tpp_ranking_slot": slot,
            "zero_point_five_tpp_ranking_tier": "350m",
        }
    )
    return candidate


def build_plan(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "mai_350m_qk_only_functional_lwt_5tpp_plan_v1",
        "registered_at": "2026-08-09",
        "scope": "top-two 350M/5TPP QK-only functional-LWT confirmations",
        "immutable_ranking": {
            "path": str(RANKING.relative_to(ROOT)),
            "sha256": sha256(RANKING),
        },
        "schedule": {
            "planned_tpp": 5.0,
            "planned_tokens": 1772999680,
            "scheduled_tokens": 1773142016,
            "max_iters": 6764,
            "tokens_per_iter": 262144,
            "eval_interval": 1691,
            "warmup_iters": 67,
        },
        "confirmations": {
            slot: {
                "screen_slug": SELECTIONS[slot],
                "config_path": str(OUTPUTS[slot].relative_to(ROOT)),
                "config_sha256": sha256(OUTPUTS[slot]),
                "learning_rate": configs[slot]["learning_rate"],
                "mfu_certificate": (
                    f"{PRO6_ROOT}/outputs/pro6_mai_v3_350m_qk_only_ladder/preflight/"
                    f"{OUTPUTS[slot].stem}.json"
                ),
            }
            for slot in SELECTIONS
        },
        "excluded_candidates": ["mult0p50"],
        "performance_gate": {
            "minimum_mfu_fraction": 0.20,
            "exact_config_required": True,
            "foreground_polling": True,
            "watchdog": False,
            "both_configs_must_pass_before_first_launch": True,
            "launch_authorized": False,
        },
        "execution_order": ["top1", "top2"],
        "monitoring": {
            "expected_duration_hours_per_candidate": [7, 8],
            "progress_callbacks": [0.20, 0.50, 0.80, 1.00],
            "heartbeat_minutes": 90,
            "heartbeat_resets_on_progress": True,
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "agent_mention": "@Codex",
        },
        "resource_admission": {
            "host": "PRO6",
            "gpu": 0,
            "project_cap_gib": 256,
            "minimum_post_admission_headroom_gib": 8,
            "no_launch_until_verified_reclaim": True,
        },
        "selection_after_5tpp": {
            "endpoint": "terminal fixed-validation CE",
            "practical_tie_threshold_ce": 0.02,
            "tie_break": "retain the 0.5TPP leader",
        },
    }


def main() -> None:
    ranking = load(RANKING)
    _validate_ranking(ranking)
    configs: dict[str, dict[str, Any]] = {}
    for slot, output in OUTPUTS.items():
        slug = SELECTIONS[slot]
        configs[slot] = build(
            load(PARENTS[slot]), load(SCREENS[slug]), ranking, slot
        )
        output.write_text(json.dumps(configs[slot], indent=2, sort_keys=True) + "\n")
        print(output.relative_to(ROOT))
    PLAN.write_text(json.dumps(build_plan(configs), indent=2, sort_keys=True) + "\n")
    print(PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
