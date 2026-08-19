#!/usr/bin/env python3
"""Preregister the 124M/5TPP ambient-lattice repair of attention V."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_only_qk64_outputgain_5tpp_lr24e4.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_v_int8_lattice_errorfeedback_5tpp_lr24e4.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_v_int8_lattice_errorfeedback_5tpp_plan.json"
)
CERTIFICATE = (
    "/mnt/ssd-data/orj/MappingNetworks/outputs/"
    "pro6_mai_v3_124m_qk_v_int8_lattice_errorfeedback_5tpp/"
    "preflight/performance_preflight.json"
)
OUT_DIR = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_124m_qk_v_int8_lattice_errorfeedback_5tpp/scientific"
)
EVIDENCE = {
    "qk_only_5tpp": (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_attention_qk_only_partial_control_result.json"
    ),
    "qkv_cayley_5tpp": (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_attention_qkv_only_partial_control_result.json"
    ),
    "qkv_cayley_20tpp": (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_attention_qkv_only_20tpp_run_result.json"
    ),
    "qk_only_20tpp": (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_attention_qk_only_lwt_20tpp_result.json"
    ),
    "full_mlp_temporal_closure_5tpp": (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_full_mlp_int8_lattice_errorfeedback_5tpp_training_result.json"
    ),
}
SOURCE_FILES = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/train.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/test_muon_int8_lattice.py",
    "examples/nanogpt/mfu_preflight.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_attn_v_int8_lattice": True,
            "block_fht_attn_v_int8_lattice_block_size": 4096,
            "block_fht_attn_v_int8_lattice_seed": 161804,
            "block_fht_attn_v_int8_lattice_error_feedback": True,
            "candidate_scope": (
                "Causal 124M/5TPP attention-V repair: retain the accepted "
                "QK64 headwise Cayley mapping and dense attention c_proj/MLP; "
                "replace only dense V with a block-4096 signed-int8 ambient "
                "lattice updated by Muon plus FP16 temporal error feedback."
            ),
            "confirmation_slot": "qk_plus_v_int8_lattice_errorfeedback_5tpp",
            "confirmation_source": (
                "Generated Cayley V costs +0.0290 CE at 5TPP and +0.0365 CE "
                "at 20TPP relative to the QK-only anchor. The MLP experiment "
                "showed that an ambient lattice with temporal error feedback "
                "can preserve useful dense update directions at 5TPP."
            ),
            "hpo_stage": "attention_v_temporal_direction_closure_124m_5tpp",
            "ladder_role": "attention_v_mechanism_localization",
            "launch_block_reason": None,
            "launch_ready": True,
            "mfu_preflight_certificate": CERTIFICATE,
            "monitoring_policy": (
                "foreground-poll the <=5 minute exact-config MFU gate; after "
                "a pass, use one idempotent terminal/error-only @Codex "
                "watchdog for the expected 1-2 hour scientific run"
            ),
            "out_dir": OUT_DIR,
            "practical_equivalence_nll": 0.02,
            "practical_equivalence_policy": (
                "At terminal require finite fixed-window validation CE "
                "<=3.5058: within +0.0200 of QK-only 3.4858 and at least "
                "0.0090 better than Cayley-QKV 3.5148. Threshold is frozen "
                "before MFU or training."
            ),
            "recipe_resolution_dependency": (
                "sealed QK-only and QK+V Cayley 5TPP localization plus sealed "
                "full-MLP int8-lattice error-feedback 5TPP closure"
            ),
            "recipe_resolution_stage": "v_specific_direction_repair",
            "selection_endpoint": (
                "terminal step-2373 fixed-window validation CE versus QK-only "
                "3.4858, Cayley-QKV 3.5148, and ordinary dense 3.5402"
            ),
            "implementation_source_hashes": {
                item: sha256(ROOT / item) for item in SOURCE_FILES
            },
        }
    )
    return config


def build_plan(config_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "mai_124m_attention_v_int8_lattice_errorfeedback_5tpp_plan_v1",
        "created_at": "2026-08-20",
        "status": "preregistered_pending_exact_config_mfu",
        "question": (
            "Can an ambient signed-int8 V lattice with FP16 temporal error "
            "feedback preserve the useful Muon V direction better than the "
            "static Cayley/BlockFHT chart?"
        ),
        "candidate": {
            "config": str(OUTPUT.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "generator": str(Path(__file__).resolve().relative_to(ROOT)),
            "model_size": "124M",
            "planned_tpp": 5.0,
            "max_iters": 2373,
            "fresh_from_scratch": True,
            "only_structural_change_from_qk_parent": (
                "dense V -> signed-int8 block-4096 ambient lattice with FP16 "
                "optimizer error feedback"
            ),
        },
        "causal_basis": {
            "qk_only_5tpp_validation_ce": 3.4858,
            "qkv_cayley_5tpp_validation_ce": 3.5148,
            "cayley_v_penalty_5tpp_ce": 0.0290,
            "qk_only_20tpp_validation_ce": 3.1488,
            "qkv_cayley_20tpp_validation_ce": 3.1853,
            "cayley_v_penalty_20tpp_ce": 0.0365,
            "reason_for_5tpp_first": (
                "The V penalty is already stable at 5TPP, while 0.5TPP is too "
                "short for a reliable V-direction decision."
            ),
        },
        "persistent_state_accounting": {
            "v_matrices": 12,
            "v_values": 7077888,
            "int8_code_bytes": 7077888,
            "fp16_scale_bytes": 3456,
            "codec_bytes": 7081344,
            "fp32_dense_weight_bytes": 28311552,
            "codec_compression_ratio": 3.998048780487805,
            "fp16_error_feedback_bytes_during_training": 14155776,
            "dense_fp32_muon_momentum_is_additional_optimizer_state": True,
            "materialized_forward_flops_unchanged": True,
        },
        "terminal_gate": {
            "maximum_validation_ce": 3.5058,
            "maximum_delta_to_qk_only_ce": 0.0200,
            "minimum_improvement_over_cayley_v_ce": 0.0090,
            "require_all_fixed_evaluations_finite": True,
            "require_clean_exit": True,
            "threshold_changes_after_measurement": False,
            "pass_classification": "ATTENTION_V_TEMPORAL_DIRECTION_REPAIR",
            "fail_classification": "ATTENTION_V_REMAINS_DIRECTION_LIMITED",
        },
        "execution_gate": {
            "host": "PRO6",
            "gpu": 0,
            "exact_config_mfu_minimum": 0.20,
            "mfu_gate_runtime": "foreground-polled; no watchdog or callback",
            "scientific_run_runtime": (
                "one idempotent terminal/error-only @Codex watchdog; no "
                "intermediate milestones or heartbeat"
            ),
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "callback_action_prompt": (
                "Verify terminal artifacts and hashes against the active "
                "project note, seal fixed-checkpoint comparisons, update "
                "durable notes, and continue only with the next causally "
                "authorized action. Do not merely acknowledge this callback."
            ),
        },
        "authorization": {
            "exact_config_mfu_preflight": True,
            "training_before_mfu_pass": False,
            "one_124m_5tpp_training_after_mfu_pass": True,
            "automatic_rerun": False,
            "automatic_20tpp": False,
            "combined_full_replacement": False,
            "larger_model": False,
        },
        "evidence": {
            key: {"path": path, "sha256": sha256(ROOT / path)}
            for key, path in EVIDENCE.items()
        },
    }


def main() -> None:
    config_bytes = encoded(build_config())
    OUTPUT.write_bytes(config_bytes)
    PLAN.write_bytes(encoded(build_plan(hashlib.sha256(config_bytes).hexdigest())))
    print(OUTPUT.relative_to(ROOT))
    print(PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
