#!/usr/bin/env python3
"""Register the causal 124M MLP int8-lattice error-feedback test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_0p5tpp.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_errorfeedback_0p5tpp.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_mlp_int8_lattice_errorfeedback_0p5tpp_plan.json"
)
IMPLEMENTATION_COMMIT = "63cbe80c1da5904ce287cfde84b181ea1a4768d0"
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/muon_int8_lattice.py",
    "examples/nanogpt/test_muon_int8_lattice.py",
    "examples/nanogpt/train.py",
    "examples/nanogpt/mfu_preflight.py",
)
REMOTE_OUTPUT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_errorfeedback_0p5tpp"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in SOURCE_PATHS}


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_mlp_int8_lattice_error_feedback": True,
            "candidate_scope": (
                "124M/0.5TPP temporal-quantization test. Preserve the sealed "
                "QK64, V16, attention-c_proj int8 lattice and both ambient "
                "MLP int8 lattices. Add only an optimizer-side FP16 sigma-"
                "delta residual to MLP c_fc and c_proj projections; attention "
                "c_proj remains the unchanged no-feedback parent codec."
            ),
            "confirmation_slot": "full_mlp_int8_lattice_errorfeedback_0p5tpp",
            "hpo_stage": "full_mlp_int8_lattice_errorfeedback_124m_0p5tpp",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": source_hashes(),
            "implementation_source_paths": list(SOURCE_PATHS),
            "ladder_role": "full_mlp_int8_lattice_temporal_closure_smallest_rung",
            "ladder_slot": "full_attention_plus_full_mlp_int8_lattice_errorfeedback",
            "mfu_preflight_certificate": f"{REMOTE_OUTPUT}/performance_preflight.json",
            "mfu_measurement_protocol": (
                "foreground exact-config real-training preflight with one "
                "warmup and eight timed updates; includes FP16 residual "
                "accumulation and all int8 projections/rematerializations"
            ),
            "monitoring_policy": (
                "foreground-poll the <=5 minute MFU gate; after a pass, use "
                "one idempotent terminal/error-only @Codex watchdog"
            ),
            "out_dir": f"{REMOTE_OUTPUT}/scientific",
            "practical_equivalence_policy": (
                "Require finite terminal validation CE <=5.3117, no more "
                "than +0.0200 behind the sealed full-attention parent 5.2917. "
                "No automatic 5TPP or larger-model transfer."
            ),
            "registered_resume_protocol": (
                "atomic exact-resume checkpoint with RNG state, int8 codes, "
                "FP16 running-max scales, dense FP32 Muon momentum, and FP16 "
                "MLP compression residuals; no serialized dense MLP weight"
            ),
            "selection_endpoint": (
                "terminal step-238 fixed-window validation CE versus the "
                "sealed full-attention parent CE 5.2917 and no-feedback full-"
                "MLP lattice CE 5.3661"
            ),
            "mlp_int8_lattice_representation": {
                "base": "independent reproducible frozen Gaussian initialization",
                "blocks": 13824,
                "code_bytes": 56623104,
                "elements": 56623104,
                "fp16_scale_bytes": 27648,
                "fp16_optimizer_error_feedback_bytes": 113246208,
                "fp32_weight_bytes": 226492416,
                "optimizer_momentum": "dense_fp32_not_in_codec_count",
                "optimizer_error_feedback": "dense_fp16_not_in_model_codec_count",
                "persistent_codec_bytes": 56650752,
                "runtime_base": "transient_dense_fp32",
                "runtime_weight": "transient_dense_fp32",
                "storage_ratio": 0.2501220703125,
                "storage_reduction": 3.998047828208882,
            },
        }
    )
    return config


def build_plan(config_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "mai_124m_full_mlp_int8_lattice_errorfeedback_plan_v1",
        "created_at": "2026-08-20",
        "status": "registered_before_exact_config_mfu_and_training",
        "question": (
            "Is the +0.0744 CE full-MLP lattice gap caused primarily by "
            "discarded sub-quantum late Muon requests?"
        ),
        "causal_basis": {
            "no_feedback_terminal_validation_ce": 5.3661,
            "full_attention_parent_validation_ce": 5.2917,
            "delta_to_parent_ce": 0.0744,
            "terminal_update_audit": {
                "attention_cproj_requested_realized_cosine": 0.02819,
                "mlp_cfc_requested_realized_cosine": 0.01525,
                "mlp_cproj_requested_realized_cosine": 0.01337,
                "mlp_cfc_realized_delta_zero_fraction": 0.78469,
                "mlp_cproj_realized_delta_zero_fraction": 0.80002,
                "mlp_cfc_half_quantum_over_update_rms": 8.39496,
                "mlp_cproj_half_quantum_over_update_rms": 8.03214,
                "endpoint_code_saturation_fraction_maximum": 0.000232,
            },
            "inference": (
                "The endpoint has ample spatial code support and is not "
                "saturated, but most late optimizer requests fall inside the "
                "current code bin. The next causal intervention must preserve "
                "discarded temporal increments, not change the spatial basis."
            ),
        },
        "candidate": {
            "generator": str(Path(__file__).relative_to(ROOT)),
            "config": str(OUTPUT.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": source_hashes(),
            "model_size": "124M",
            "planned_tpp": 0.5,
            "max_iters": 238,
            "fresh_from_scratch": True,
        },
        "update_rule": {
            "requested": "U_t = W_t + Delta_t + R_t",
            "projection": "W_(t+1) = Q_t(U_t)",
            "residual": "R_(t+1) = U_t - W_(t+1)",
            "residual_dtype": "float16",
            "scope": ["mlp.c_fc", "mlp.c_proj"],
            "attention_cproj_feedback": False,
            "model_codec_unchanged": True,
            "inference_parameters_unchanged": True,
            "extra_training_state_bytes": 113246208,
        },
        "frozen_invariants": [
            "Change only optimizer-side MLP lattice error feedback.",
            "Preserve the exact attention parent, MLP lattice, data, optimizer, LR, schedule, seeds, evaluation windows, and checkpoint cadence.",
            "Keep the attention c_proj lattice without error feedback so the intervention is MLP-localized.",
            "Do not change lattice block size, scale rule, code precision, or dense Muon momentum.",
        ],
        "verification": {
            "unit_tests": [
                "sub-quantum requests accumulate into a code transition",
                "optimizer residual remains FP16 across exact resume",
                "model routes feedback only to configured MLP lattice modules",
            ],
            "exact_config_mfu_minimum": 0.20,
            "mfu_runtime": "foreground-polled with no watchdog",
            "training_runtime": "terminal/error-only idempotent @Codex watchdog",
        },
        "authorization": {
            "exact_config_mfu_passed": False,
            "one_124m_0p5tpp_training": False,
            "automatic_rerun": False,
            "automatic_5tpp": False,
            "larger_model": False,
        },
        "terminal_gate": {
            "maximum_terminal_validation_ce": 5.3117,
            "maximum_delta_to_full_attention_parent_ce": 0.0200,
            "required_improvement_over_no_feedback_ce": 0.0544,
            "require_clean_exit": True,
            "require_all_fixed_evaluations_finite": True,
            "pass_classification": "TEMPORAL_QUANTIZATION_WAS_THE_DOMINANT_MLP_LATTICE_OBSTRUCTION",
            "fail_classification": "TEMPORAL_ERROR_FEEDBACK_IS_NOT_SUFFICIENT_FOR_FULL_MLP_CLOSURE",
        },
        "scope_guards": [
            "FP16 residual is optimizer state, not an inference parameter or Mapping-Network latent.",
            "The model-weight codec remains about 4x smaller than FP32; training state grows by two bytes per MLP weight.",
            "A smallest-rung pass authorizes only a separate decision, never automatic scale-up.",
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
