#!/usr/bin/env python3
"""Preregister the 124M/20TPP attention-c_proj feedback attribution run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullreplacement_repaired_v_int8_errorfeedback_20tpp.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullreplacement_all_int8_errorfeedback_20tpp.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_replacement_cproj_errorfeedback_20tpp_plan.json"
)
FAILED_PARENT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_replacement_repaired_v_20tpp_training_result.json"
)
QK_20TPP = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_qk_only_lwt_20tpp_result.json"
)
V_REPAIR_20TPP = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_v_int8_lattice_errorfeedback_20tpp_training_result.json"
)
OUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_124m_fullreplacement_all_int8_errorfeedback_20tpp"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_attn_cproj_int8_lattice_error_feedback": True,
            "candidate_scope": (
                "Fresh 124M/20TPP single-variable attribution of FP16 "
                "temporal error feedback on the attention c_proj int8 "
                "lattice. QK64, repaired V, both MLP lattices, optimizer, "
                "data, seeds, schedule, and evaluation windows are unchanged."
            ),
            "confirmation_slot": (
                "full_replacement_attention_cproj_errorfeedback_124m_20tpp"
            ),
            "confirmation_source": (
                "The no-feedback parent passed at 5TPP but accumulated a "
                "+0.0319 CE QK-only deficit by 20TPP while all codec states "
                "remained healthy; attention c_proj was the only ambient "
                "lattice family without a temporal residual."
            ),
            "hpo_stage": "full_replacement_cproj_temporal_attribution",
            "ladder_role": "smallest_informative_full_replacement_repair",
            "mfu_preflight_certificate": f"{OUT_ROOT}/performance_preflight.json",
            "out_dir": f"{OUT_ROOT}/scientific",
            "practical_equivalence_policy": (
                "Pass requires every fixed checkpoint within +0.0200 CE of "
                "QK-only, terminal CE <=3.1647, and improvements >=0.0080 CE "
                "over the no-feedback parent at steps 7119 and 9489."
            ),
            "recipe_resolution_dependency": (
                "sealed 124M/20TPP no-feedback complete-replacement failure, "
                "plus sealed QK-only and isolated repaired-V controls"
            ),
            "recipe_resolution_stage": "attention_cproj_temporal_attribution",
            "registered_resume_protocol": (
                "atomic exact-resume checkpoint with RNG state, int8 codes, "
                "FP16 running-max scales, dense FP32 Muon momentum, and FP16 "
                "compression residuals for V, attention c_proj, c_fc, and "
                "MLP c_proj; no serialized dense lattice weight"
            ),
            "resolved_from_template": str(BASE.relative_to(ROOT)),
            "selection_endpoint": (
                "four registered fixed-window evaluations against QK-only, "
                "isolated V, and the no-feedback full-replacement parent"
            ),
        }
    )
    return config


def build_plan(config_sha256: str) -> dict[str, object]:
    return {
        "schema_version": (
            "mai_124m_full_replacement_cproj_errorfeedback_20tpp_plan_v1"
        ),
        "created_at": "2026-08-20",
        "status": "preregistered_pending_tests_and_exact_config_mfu",
        "question": (
            "Is discarded sub-quantum attention-c_proj motion the cause of "
            "the complete replacement's late 20TPP drift?"
        ),
        "hypothesis": {
            "observation": (
                "The failed parent has healthy noncollapsed codecs, validated "
                "V feedback, MLP feedback, and no attention-c_proj feedback."
            ),
            "intervention": (
                "Enable the existing FP16 sigma-delta compression residual on "
                "the 12 attention-c_proj lattices and change nothing else."
            ),
            "falsification": (
                "If late fixed-curve drift is not materially reduced, reject "
                "attention c_proj as the primary cause and localize the "
                "remaining defect to full-MLP composition."
            ),
        },
        "candidate": {
            "config": str(OUTPUT.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "generator": str(Path(__file__).resolve().relative_to(ROOT)),
            "model_size": "124M",
            "planned_tpp": 20.0,
            "max_iters": 9489,
            "fresh_from_scratch": True,
            "only_scientific_change_from_failed_parent": (
                "block_fht_attn_cproj_int8_lattice_error_feedback: false -> true"
            ),
            "additional_persistent_optimizer_state_bytes": 14155776,
            "additional_trainable_parameters": 0,
            "additional_materialized_inference_flops": 0,
        },
        "why_no_5tpp_screen": (
            "The parent already passed 5TPP and first violated its temporal "
            "gate at step 7119; a 5TPP rerun cannot test this hypothesis."
        ),
        "frozen_comparators": {
            "ordinary_dense_terminal": {"step": 9489, "validation_ce": 3.1547},
            "qk_only": {
                "steps": [2373, 4746, 7119, 9489],
                "validation_ce": [3.5038, 3.3179, 3.2046, 3.1488],
            },
            "isolated_repaired_v": {
                "steps": [2373, 4746, 7119, 9489],
                "validation_ce": [3.5040, 3.3194, 3.2055, 3.1489],
            },
            "no_feedback_full_replacement_parent": {
                "steps": [2373, 4746, 7119, 9489],
                "validation_ce": [3.5139, 3.3373, 3.2327, 3.1807],
            },
        },
        "frozen_gate": {
            "maximum_delta_to_qk_only_at_every_fixed_evaluation_ce": 0.0200,
            "minimum_improvement_over_parent_at_step_7119_ce": 0.0080,
            "minimum_improvement_over_parent_at_step_9489_ce": 0.0080,
            "terminal_practical_maximum_validation_ce": 3.1647,
            "terminal_zero_gap_validation_ce": 3.1547,
            "require_all_fixed_evaluations_finite": True,
            "require_clean_exit": True,
            "threshold_changes_after_measurement": False,
            "pass_classification": (
                "FULL_REPLACEMENT_CPROJ_TEMPORAL_REPAIR_CONFIRMED_124M_20TPP"
            ),
            "fail_classification": (
                "FULL_REPLACEMENT_CPROJ_TEMPORAL_REPAIR_REJECTED_124M_20TPP"
            ),
        },
        "execution_gate": {
            "host": "PRO6",
            "gpu": 0,
            "unit_tests_before_remote_sync": True,
            "exact_config_mfu_minimum": 0.20,
            "mfu_gate_runtime": "foreground-polled; no watchdog or callback",
            "scientific_run_runtime": (
                "one persistent @Codex watchdog with 20%, 50%, terminal, "
                "failure/stall, and resettable 90-minute heartbeat callbacks"
            ),
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
        },
        "evidence": {
            "failed_parent": {
                "path": str(FAILED_PARENT.relative_to(ROOT)),
                "sha256": sha256(FAILED_PARENT),
            },
            "qk_only_20tpp": {
                "path": str(QK_20TPP.relative_to(ROOT)),
                "sha256": sha256(QK_20TPP),
            },
            "isolated_repaired_v_20tpp": {
                "path": str(V_REPAIR_20TPP.relative_to(ROOT)),
                "sha256": sha256(V_REPAIR_20TPP),
            },
        },
        "authorization": {
            "one_exact_config_mfu_preflight": True,
            "one_fresh_124m_20tpp_training_after_mfu_pass": True,
            "automatic_rerun": False,
            "parallel_arm": False,
            "larger_model": False,
            "multi_seed_closure_claim": False,
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
