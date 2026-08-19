#!/usr/bin/env python3
"""Register the 124M/5TPP full-MLP lattice error-feedback transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_errorfeedback_0p5tpp.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_errorfeedback_5tpp.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_mlp_int8_lattice_errorfeedback_5tpp_plan.json"
)
SMALL_RUNG_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_mlp_int8_lattice_errorfeedback_0p5tpp_training_result.json"
)
ATTENTION_PARENT_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_cproj_int8_lattice_5tpp_training_result.json"
)
REMOTE_OUTPUT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_errorfeedback_5tpp"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "candidate_scope": (
                "Fresh 124M/5TPP same-size horizon transfer of the passed "
                "full-replacement temporal codec: QK64/V16 Cayley mappings, "
                "attention-c_proj signed-int8 lattice without feedback, and "
                "both ambient MLP signed-int8 lattices with optimizer-side "
                "FP16 sigma-delta residuals."
            ),
            "confirmation_slot": "full_mlp_int8_lattice_errorfeedback_5tpp",
            "confirmation_source": (
                "The sealed 124M/0.5TPP full-replacement candidate reached "
                "validation CE 5.291025, 0.000675 better than its matched "
                "full-attention parent and 0.075075 better than direct rounding."
            ),
            "eval_interval": 594,
            "hpo_stage": "full_mlp_int8_lattice_errorfeedback_124m_5tpp_transfer",
            "ladder_role": "full_mlp_int8_lattice_temporal_closure_horizon_transfer",
            "launch_block_reason": None,
            "launch_ready": True,
            "lr_decay_iters": 2373,
            "max_iters": 2373,
            "mfu_preflight_certificate": f"{REMOTE_OUTPUT}/performance_preflight.json",
            "monitoring_policy": (
                "foreground-poll the <=5 minute exact-config MFU gate; after "
                "a pass, use one idempotent terminal/error-only @Codex "
                "watchdog for the expected 1-2 hour scientific run"
            ),
            "out_dir": f"{REMOTE_OUTPUT}/scientific",
            "planned_tokens": 621868800,
            "planned_tpp": 5.0,
            "practical_equivalence_nll": 0.02,
            "practical_equivalence_policy": (
                "Primary gate: finite terminal validation CE <=3.5602, the "
                "already frozen ordinary-dense 3.5402 plus 0.0200. This also "
                "permits at most +0.0103 versus the sealed full-attention "
                "lattice parent 3.5499."
            ),
            "recipe_resolution_dependency": (
                "sealed 124M/0.5TPP full-MLP temporal-feedback PASS at "
                "validation CE 5.291025161743164"
            ),
            "recipe_resolution_stage": "same_size_horizon_transfer_only",
            "scheduled_tokens": 622067712,
            "scheduled_tpp": 5.001599308407175,
            "selection_endpoint": (
                "terminal step-2373 fixed-window validation CE versus sealed "
                "full-attention lattice 3.5499 and ordinary dense 3.5402"
            ),
            "warmup_iters": 23,
        }
    )
    return config


def build_plan(config_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "mai_124m_full_mlp_int8_lattice_errorfeedback_5tpp_plan_v1",
        "created_at": "2026-08-20",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Does MLP-local temporal error feedback preserve full replacement "
            "at 5TPP, rather than only repairing the 0.5TPP endpoint?"
        ),
        "promotion_basis": {
            "smallest_rung_result": {
                "path": str(SMALL_RUNG_RESULT.relative_to(ROOT)),
                "sha256": sha256(SMALL_RUNG_RESULT),
                "terminal_validation_ce": 5.291025161743164,
                "delta_to_matched_attention_parent_ce": -0.0006748382568357851,
                "improvement_over_no_feedback_ce": 0.07507483825683559,
                "parent_equivalent_compute_ratio": 0.9982192749239748,
            },
            "five_tpp_attention_parent": {
                "path": str(ATTENTION_PARENT_RESULT.relative_to(ROOT)),
                "sha256": sha256(ATTENTION_PARENT_RESULT),
                "terminal_validation_ce": 3.5499,
                "ordinary_dense_terminal_validation_ce": 3.5402,
            },
        },
        "candidate": {
            "generator": str(Path(__file__).relative_to(ROOT)),
            "config": str(OUTPUT.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "model_size": "124M",
            "planned_tpp": 5.0,
            "max_iters": 2373,
            "learning_rate": 0.0024,
            "min_lr": 0.00024,
            "fresh_from_scratch": True,
            "persistent_model_state": (
                "QK64/V16 mappings plus signed-int8 attention-c_proj and "
                "signed-int8 c_fc/c_proj lattices"
            ),
            "optimizer_only_feedback": "FP16 residual on the 24 MLP matrices",
        },
        "causal_contract": [
            "Change only horizon, warmup, evaluation cadence, and output identity from the passed 0.5TPP candidate.",
            "Preserve the exact model structure, lattice block size, code precision, residual dtype, optimizer, LR, data, seeds, and fixed-evaluation protocol.",
            "Start from a fresh initialization; do not continue the 0.5TPP checkpoint.",
            "Keep attention c_proj feedback disabled so the repaired mechanism remains MLP-localized.",
        ],
        "execution_gate": {
            "host": "PRO6",
            "gpu": 0,
            "exact_config_mfu_minimum": 0.20,
            "mfu_gate_runtime": "foreground-polled; no watchdog or callback",
            "scientific_run_runtime": (
                "one idempotent terminal/error-only @Codex watchdog; no "
                "20%, 50%, 80%, or heartbeat callbacks"
            ),
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "callback_action_prompt": (
                "Verify terminal artifacts and hashes against the active "
                "project note, seal fixed-checkpoint comparisons, update "
                "durable notes, and continue only with the next causally "
                "authorized action. Do not merely acknowledge this callback."
            ),
        },
        "terminal_gate": {
            "ordinary_dense_terminal_validation_ce": 3.5402,
            "full_attention_parent_terminal_validation_ce": 3.5499,
            "maximum_terminal_validation_ce": 3.5602,
            "maximum_delta_to_ordinary_dense_ce": 0.0200,
            "maximum_delta_to_full_attention_parent_ce": 0.0103,
            "require_clean_exit": True,
            "require_all_fixed_evaluations_finite": True,
            "threshold_changes_after_measurement": False,
            "pass_classification": "FULL_MLP_PERSISTENT_STATE_NEAR_DENSE_AT_124M_5TPP",
            "fail_classification": "MLP_TEMPORAL_CLOSURE_DOES_NOT_TRANSFER_TO_5TPP",
        },
        "authorization": {
            "exact_config_mfu_passed": False,
            "one_124m_5tpp_training_after_mfu_pass": False,
            "automatic_rerun": False,
            "automatic_20tpp": False,
            "larger_model": False,
        },
        "scope_guards": [
            "The int8 model-weight codec is about 4x smaller than FP32, not 200x.",
            "The FP16 residual and dense FP32 Muon momentum are optimizer state and must be counted.",
            "Dense transient materialization and materialized inference FLOPs remain unchanged.",
            "A 5TPP pass authorizes no larger model or 20TPP run automatically.",
        ],
    }


def main() -> None:
    config = build_config()
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    PLAN.write_text(
        json.dumps(build_plan(sha256(OUTPUT)), indent=2, sort_keys=True) + "\n"
    )
    print(OUTPUT)
    print(PLAN)


if __name__ == "__main__":
    main()
