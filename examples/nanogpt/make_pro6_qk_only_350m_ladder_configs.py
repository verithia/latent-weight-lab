#!/usr/bin/env python3
"""Register the PRO6 350M/0.5TPP QK-only functional-LWT screen.

This is a scale-transfer screen, not an automatic promotion of the 124M
result.  It preserves the MAI per-rung LR screen and changes the full-attention
parents only where required to keep V/O dense and install the already selected
headwise Q/K chart.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
ARTIFACT_DIR = CONFIG_DIR / "selection_artifacts"
QK_SOURCE = CONFIG_DIR / "pro6_mai_v3_124m_qk_only_qk64_outputgain_20tpp_lr24e4.json"
QK_RESULT = ARTIFACT_DIR / "124m_attention_qk_only_lwt_20tpp_result.json"
MAI_GENERATOR = ROOT / "examples/nanogpt/make_y400_mai_scaling_ladder_configs.py"
FULLATTN_RANKING = ARTIFACT_DIR / "350m_fullattn_0p5tpp_provisional_ranking.json"
PLAN = ARTIFACT_DIR / "350m_qk_only_functional_lwt_0p5tpp_plan.json"

MULTIPLIERS = {
    "mult0p50": 0.50,
    "mult0p75": 0.75,
    "mult1p00": 1.00,
}
PARENTS = {
    slug: CONFIG_DIR / f"y400_mai_v2_350m_fullattn_blockfht_0p5tpp_{slug}.json"
    for slug in MULTIPLIERS
}
OUTPUTS = {
    slug: CONFIG_DIR / f"pro6_mai_v3_350m_qk_only_qk64_outputgain_0p5tpp_{slug}.json"
    for slug in MULTIPLIERS
}

QK = "attn.c_attn.qk_headwise"
PRO6_ROOT = "/mnt/ssd-data/orj/MappingNetworks"
PRO6_SYMLINK = "/home/pro6000-9980x/MappingNetworks"
BASE_LR = 0.0024
HEAD_DIM = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build(parent: dict[str, Any], source: dict[str, Any], slug: str) -> dict[str, Any]:
    multiplier = MULTIPLIERS[slug]
    if parent.get("model_tier") != "350m" or float(parent.get("planned_tpp", 0.0)) != 0.5:
        raise ValueError(f"{slug} parent is not a 350M/0.5TPP screen")
    if int(parent["n_embd"]) // int(parent["n_head"]) != HEAD_DIM:
        raise ValueError("350M parent violates the invariant 64-dimensional attention head")
    if float(parent["candidate_main_lr_multiplier"]) != multiplier:
        raise ValueError(f"{slug} parent multiplier mismatch")
    if float(parent["learning_rate"]) != BASE_LR * multiplier:
        raise ValueError(f"{slug} parent learning-rate mismatch")

    candidate = copy.deepcopy(parent)
    for field in (
        "block_fht_attn_cayley_bilateral_targets",
        "block_fht_attn_cayley_lr_scale",
        "block_fht_attn_cayley_output_targets",
        "block_fht_attn_cayley_rank",
        "block_fht_attn_cayley_ranks",
        "block_fht_attn_cayley_scale",
        "block_fht_attn_cayley_seed",
        "block_fht_attn_cayley_targets",
    ):
        candidate[field] = copy.deepcopy(source[field])

    run_name = f"pro6_mai_v3_350m_qk_only_qk64_outputgain_0p5tpp_{slug}"
    candidate.update(
        {
            "block_fht_targets": [QK],
            "block_fht_output_gain_targets": [QK],
            "data_dir": f"{PRO6_SYMLINK}/data/finewebedu_20b",
            "data_staging_policy": (
                "PRO6 resident immutable FineWeb-Edu 20B dataset; verify manifest "
                "and runtime fixed-index digest before launch"
            ),
            "data_staging_source": f"{PRO6_ROOT}/data/finewebedu_20b",
            "hpo_stage": "attention_qk_only_functional_lwt_screen_350m_0p5tpp",
            "ladder_role": "screen_only",
            "ladder_slot": slug,
            "candidate_scope": (
                "350M scale-transfer screen of the accepted 124M functional-LWT "
                "boundary: generate only headwise Q/K; keep V, attention c_proj, "
                "and the entire MLP dense and Muon-trained."
            ),
            "selected_lwt_allocation": {
                "generated": [QK],
                "dense_muon": ["attn.c_attn.v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
            },
            "head_dimension_scaling_rule": {
                "head_dim": HEAD_DIM,
                "qk_cayley_rank": HEAD_DIM,
                "reason": (
                    "All registered MAI tiers keep n_embd/n_head=64. Rank 64 is "
                    "already full per-head Q/K chart rank; layers and heads scale "
                    "the total LWT budget automatically."
                ),
            },
            "learning_rate_transfer_rule": {
                "accepted_350m_dense_main_lr": BASE_LR,
                "candidate_main_lr_multiplier": multiplier,
                "candidate_learning_rate": BASE_LR * multiplier,
                "screen_values": [0.5, 0.75, 1.0],
                "reason": (
                    "Do not copy the 124M winner. Re-screen the new operator/size "
                    "at 0.5TPP using the MAI candidate multipliers."
                ),
            },
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "automatic_larger_rung_promotion": False,
                "reason": (
                    "The 124M result accepted QK-only functional LWT but explicitly "
                    "did not authorize automatic scaling. This separately preregistered "
                    "user-directed screen tests transfer without claiming it."
                ),
                "recorded_at": "2026-08-09",
                "scope": "350M/0.5TPP QK-only functional-LWT scale-transfer screen",
                "source_result": str(QK_RESULT.relative_to(ROOT)),
                "source_result_sha256": sha256(QK_RESULT),
            },
            "out_dir": f"{PRO6_ROOT}/outputs/pro6_mai_v3_350m_qk_only_ladder/{run_name}",
            "practical_equivalence_policy": (
                "Rank all three stable candidates only by terminal held-out NLL on "
                "the shared fixed evaluation windows. Advance top1/top2 to 5TPP; "
                "authorize no 5TPP config before the immutable ranking exists."
            ),
            "monitoring_policy": (
                "Expected one-to-two-hour screen: one persistent terminal-only "
                "watchdog; callbacks only on clean completion, error, stall, or "
                "monitor degradation; every callback invokes @Codex with a concrete "
                "continuation prompt."
            ),
            "prelaunch_provenance_requirements": (
                "record clean commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed-eval "
                "digest, and exact-config foreground-polled MFU certificate"
            ),
            "mfu_preflight_certificate": None,
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.20,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "resolved_from_template": str(PARENTS[slug].relative_to(ROOT)),
            "scale_transfer_source_config": str(QK_SOURCE.relative_to(ROOT)),
            "scale_transfer_source_config_sha256": sha256(QK_SOURCE),
        }
    )
    return candidate


def build_plan(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = {
        "accepted_124m_qk_only_result": QK_RESULT,
        "accepted_124m_qk_only_config": QK_SOURCE,
        "mai_ladder_generator": MAI_GENERATOR,
        "prior_350m_full_attention_ranking": FULLATTN_RANKING,
        **{f"parent_{slug}": path for slug, path in PARENTS.items()},
    }
    return {
        "schema_version": "mai_350m_qk_only_functional_lwt_0p5tpp_plan_v1",
        "registered_at": "2026-08-09",
        "scope": "350M/0.5TPP QK-only functional-LWT scale-transfer LR screen",
        "scientific_status": {
            "automatic_promotion_from_124m": False,
            "new_user_directed_screen": True,
            "larger_rung_claimed": False,
            "five_tpp_authorized": False,
        },
        "theory": {
            "accepted_boundary": (
                "Generate Q/K only; V, attention output projection, and MLP remain dense."
            ),
            "head_dimension_invariant": HEAD_DIM,
            "qk_cayley_rank": HEAD_DIM,
            "rank_scaling": "constant because head dimension is constant across MAI tiers",
            "budget_scaling": "layer/head-local charts increase automatically with layer/head count",
            "excluded_claims": [
                "full attention replacement",
                "V replacement",
                "attention c_proj replacement",
                "inference parameter reduction",
            ],
        },
        "ladder": {
            "planned_tpp": 0.5,
            "model_tier": "350m",
            "materialized_parameter_count": configs["mult1p00"]["estimated_active_params"],
            "max_iters": 677,
            "scheduled_tokens": 177471488,
            "accepted_dense_main_lr": BASE_LR,
            "candidate_multipliers": list(MULTIPLIERS.values()),
            "candidate_learning_rates": [BASE_LR * value for value in MULTIPLIERS.values()],
            "selection_endpoint": "terminal held-out NLL on fixed MAI evaluation windows",
            "ranking_rule": "stable ascending terminal validation CE; hard reject nonfinite/failed runs",
            "promotion_rule": "only immutable top1/top2 ranking may authorize two 5TPP confirmations",
        },
        "identity": {
            "dataset_manifest_sha256": configs["mult1p00"]["data_manifest_sha256"],
            "fixed_eval_index_spec_sha256": configs["mult1p00"]["fixed_eval_index_spec_sha256"],
            "eval_protocol_id": configs["mult1p00"]["eval_protocol_id"],
            "model_seed": configs["mult1p00"]["model_seed"],
            "block_fht_seed": configs["mult1p00"]["block_fht_seed"],
            "attn_cayley_seed": configs["mult1p00"]["block_fht_attn_cayley_seed"],
        },
        "candidates": {
            slug: {
                "config": str(OUTPUTS[slug].relative_to(ROOT)),
                "config_sha256": sha256(OUTPUTS[slug]),
                "candidate_main_lr_multiplier": multiplier,
                "learning_rate": configs[slug]["learning_rate"],
                "out_dir": configs[slug]["out_dir"],
                "mfu_certificate": (
                    f"{PRO6_ROOT}/outputs/pro6_mai_v3_350m_qk_only_ladder/preflight/"
                    f"{OUTPUTS[slug].stem}.json"
                ),
            }
            for slug, multiplier in MULTIPLIERS.items()
        },
        "immutable_evidence": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in evidence.items()
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.20,
            "exact_config_required": True,
            "foreground_polling": True,
            "watchdog": False,
            "all_candidates_must_pass_before_queue_launch": True,
        },
        "monitoring": {
            "terminal_only": True,
            "expected_duration_hours_per_candidate": [1, 2],
            "callbacks": ["completion", "error", "stall", "monitor_degraded"],
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "agent_mention": "@Codex",
            "terminal_action": (
                "verify and seal exact artifacts, rank when all screens finish, then "
                "continue only through the preregistered top1/top2 gate"
            ),
        },
        "resource_admission": {
            "host": "PRO6",
            "gpu": 0,
            "project_cap_gib": 256,
            "minimum_post_admission_headroom_gib": 8,
            "no_launch_until_verified_reclaim": True,
        },
    }


def main() -> None:
    source = load(QK_SOURCE)
    result = load(QK_RESULT)
    if result.get("classification") != "PASS_FUNCTIONAL_LWT_QK_ONLY_124M_20TPP":
        raise ValueError("accepted 124M QK-only source result is absent")
    if result.get("decision", {}).get("functional_lwt_qk_only_partial_attention_accepted") is not True:
        raise ValueError("124M QK-only boundary is not accepted")
    configs: dict[str, dict[str, Any]] = {}
    for slug, output in OUTPUTS.items():
        configs[slug] = build(load(PARENTS[slug]), source, slug)
        output.write_text(json.dumps(configs[slug], indent=2, sort_keys=True) + "\n")
        print(output.relative_to(ROOT))
    PLAN.write_text(json.dumps(build_plan(configs), indent=2, sort_keys=True) + "\n")
    print(PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
