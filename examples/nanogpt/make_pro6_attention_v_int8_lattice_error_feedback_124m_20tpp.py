#!/usr/bin/env python3
"""Preregister the 124M/20TPP horizon transfer of repaired attention V."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_v_int8_lattice_errorfeedback_5tpp_lr24e4.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_v_int8_lattice_errorfeedback_20tpp_lr24e4.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_v_int8_lattice_errorfeedback_20tpp_plan.json"
)
RESULT_5TPP = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_v_int8_lattice_errorfeedback_5tpp_training_result.json"
)
QK_20TPP = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_qk_only_lwt_20tpp_result.json"
)
CAYLEY_20TPP = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_qkv_only_20tpp_run_result.json"
)
CERTIFICATE = (
    "/mnt/ssd-data/orj/MappingNetworks/outputs/"
    "pro6_mai_v3_124m_qk_v_int8_lattice_errorfeedback_20tpp/"
    "preflight/performance_preflight.json"
)
OUT_DIR = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_124m_qk_v_int8_lattice_errorfeedback_20tpp/scientific"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "max_iters": 9489,
            "lr_decay_iters": 9489,
            "warmup_iters": 94,
            "eval_interval": 2373,
            "planned_tokens": 2487475200,
            "scheduled_tokens": 2487484416,
            "planned_tpp": 20.0,
            "scheduled_tpp": 20.00007409923122,
            "candidate_scope": (
                "Fresh 124M/20TPP horizon transfer of the accepted QK64 plus "
                "ambient signed-int8 V lattice with FP16 temporal error "
                "feedback; attention c_proj and the complete MLP remain dense."
            ),
            "confirmation_slot": "qk_plus_v_int8_lattice_errorfeedback_20tpp",
            "confirmation_source": (
                "At 124M/5TPP the repaired V endpoint was only +0.000406 CE "
                "behind QK-only and recovered 98.60% of the Cayley-V penalty."
            ),
            "hpo_stage": "attention_v_temporal_direction_closure_124m_20tpp",
            "ladder_role": "attention_v_horizon_transfer",
            "launch_block_reason": None,
            "launch_ready": True,
            "mfu_preflight_certificate": CERTIFICATE,
            "monitoring_policy": (
                "foreground-poll the <=5 minute exact-config MFU gate; after "
                "a pass use one persistent @Codex watchdog with 20%, 50%, and "
                "terminal callbacks plus a resettable 90-minute heartbeat"
            ),
            "out_dir": OUT_DIR,
            "practical_equivalence_nll": 0.02,
            "practical_equivalence_policy": (
                "At each registered positive fixed evaluation require candidate "
                "validation CE no more than +0.0200 above the matched QK-only "
                "curve; terminal maximum is 3.1688, which also improves at "
                "least 0.0165 over Cayley-QKV 3.1853."
            ),
            "recipe_resolution_dependency": (
                "sealed 124M/5TPP attention-V temporal repair plus sealed "
                "124M/20TPP QK-only and Cayley-QKV controls"
            ),
            "recipe_resolution_stage": "v_temporal_horizon_transfer",
            "selection_endpoint": (
                "four registered fixed-window validation evaluations versus "
                "the matched QK-only and Cayley-QKV 20TPP curves"
            ),
            "resolved_from_template": str(BASE.relative_to(ROOT)),
        }
    )
    return config


def build_plan(config_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "mai_124m_attention_v_int8_lattice_errorfeedback_20tpp_plan_v1",
        "created_at": "2026-08-20",
        "status": "preregistered_pending_exact_config_mfu",
        "question": (
            "Does the V temporal-direction repair remain QK-only-equivalent "
            "through the 124M/20TPP horizon?"
        ),
        "candidate": {
            "config": str(OUTPUT.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "generator": str(Path(__file__).resolve().relative_to(ROOT)),
            "model_size": "124M",
            "planned_tpp": 20.0,
            "max_iters": 9489,
            "fresh_from_scratch": True,
            "only_changes_from_passed_5tpp_candidate": [
                "training horizon and LR-decay horizon 2373 -> 9489",
                "warmup 23 -> 94",
                "fixed evaluation interval 594 -> 2373",
                "output identity and monitoring policy"
            ]
        },
        "frozen_comparators": {
            "qk_only": {
                "validation_ce": [3.5038, 3.3179, 3.2046, 3.1488],
                "steps": [2373, 4746, 7119, 9489]
            },
            "cayley_qkv": {
                "validation_ce": [3.5397, 3.3533, 3.2413, 3.1853],
                "steps": [2373, 4746, 7119, 9489]
            }
        },
        "terminal_gate": {
            "maximum_validation_ce": 3.1688,
            "maximum_delta_to_qk_only_ce": 0.0200,
            "minimum_improvement_over_cayley_qkv_ce": 0.0165,
            "maximum_delta_to_qk_only_at_every_fixed_evaluation_ce": 0.0200,
            "require_all_fixed_evaluations_finite": True,
            "require_clean_exit": True,
            "threshold_changes_after_measurement": False,
            "pass_classification": "ATTENTION_V_TEMPORAL_DIRECTION_REPAIR_CONFIRMED_20TPP",
            "fail_classification": "ATTENTION_V_TEMPORAL_DIRECTION_REPAIR_NONTRANSFER_20TPP"
        },
        "execution_gate": {
            "host": "PRO6",
            "gpu": 0,
            "exact_config_mfu_minimum": 0.20,
            "mfu_gate_runtime": "foreground-polled; no watchdog or callback",
            "scientific_run_runtime": (
                "one persistent aggregate @Codex watchdog with 20%, 50%, and "
                "terminal callbacks; 90-minute heartbeat reset by any callback"
            ),
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "callback_action_prompt": (
                "Verify live state, loss, GPU health, and active project note; "
                "terminal success must seal artifacts and continue only with "
                "the next causally authorized full-replacement experiment."
            )
        },
        "authorization": {
            "exact_config_mfu_preflight": True,
            "training_before_mfu_pass": False,
            "one_124m_20tpp_training_after_mfu_pass": True,
            "automatic_rerun": False,
            "combined_full_replacement": False,
            "larger_model": False
        },
        "evidence": {
            "v_repair_5tpp": {"path": str(RESULT_5TPP.relative_to(ROOT)), "sha256": sha256(RESULT_5TPP)},
            "qk_only_20tpp": {"path": str(QK_20TPP.relative_to(ROOT)), "sha256": sha256(QK_20TPP)},
            "cayley_qkv_20tpp": {"path": str(CAYLEY_20TPP.relative_to(ROOT)), "sha256": sha256(CAYLEY_20TPP)}
        },
        "scope_guards": [
            "This is V localization with dense attention c_proj and dense MLP, not full replacement.",
            "The persistent V codec is approximately 4x smaller, but dense materialization and inference FLOPs remain.",
            "No recombination or larger-rung run is authorized before terminal analysis."
        ]
    }


def main() -> None:
    config_bytes = encoded(build_config())
    OUTPUT.write_bytes(config_bytes)
    PLAN.write_bytes(encoded(build_plan(hashlib.sha256(config_bytes).hexdigest())))
    print(OUTPUT.relative_to(ROOT))
    print(PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
