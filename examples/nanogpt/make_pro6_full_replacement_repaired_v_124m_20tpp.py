#!/usr/bin/env python3
"""Preregister the 124M/20TPP transfer of the repaired-V full replacement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullreplacement_repaired_v_int8_errorfeedback_5tpp.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullreplacement_repaired_v_int8_errorfeedback_20tpp.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_replacement_repaired_v_20tpp_plan.json"
)
RESULT_5TPP = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_replacement_repaired_v_5tpp_training_result.json"
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
    "pro6_mai_v3_124m_fullreplacement_repaired_v_int8_errorfeedback_20tpp"
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
                "Fresh 124M/20TPP horizon transfer of the complete repaired-V "
                "replacement: QK64 Cayley mapping, ambient V signed-int8 "
                "lattice with FP16 temporal feedback, attention-c_proj "
                "signed-int8 lattice without feedback, and c_fc/c_proj "
                "signed-int8 lattices with FP16 temporal feedback."
            ),
            "confirmation_slot": "full_replacement_repaired_v_124m_20tpp",
            "confirmation_source": (
                "The identical complete replacement beat ordinary dense by "
                "0.030621 CE and its Cayley-V parent by 0.040183 CE at 5TPP; "
                "isolated repaired V remained QK-only-equivalent through 20TPP."
            ),
            "dense_fixed_validation_curve": [
                {"step": 9489, "validation_ce": 3.1547}
            ],
            "hpo_stage": "full_replacement_repaired_v_124m_20tpp_transfer",
            "ladder_role": "full_replacement_horizon_transfer",
            "launch_block_reason": None,
            "launch_ready": True,
            "mfu_preflight_certificate": f"{OUT_ROOT}/performance_preflight.json",
            "monitoring_policy": (
                "foreground-poll the <=5 minute exact-config MFU gate; after "
                "a pass use one persistent @Codex watchdog with 20%, 50%, and "
                "terminal callbacks plus a resettable 90-minute heartbeat"
            ),
            "out_dir": f"{OUT_ROOT}/scientific",
            "practical_equivalence_nll": 0.01,
            "practical_equivalence_policy": (
                "Zero-gap closure is terminal validation CE <=3.1547; practical "
                "acceptance is <=3.1647. At every registered checkpoint the "
                "candidate must be no more than +0.0200 above both QK-only and "
                "the isolated repaired-V 20TPP curves."
            ),
            "recipe_resolution_dependency": (
                "sealed repaired-V complete replacement at 124M/5TPP plus "
                "sealed repaired-V and QK-only controls at 124M/20TPP"
            ),
            "recipe_resolution_stage": "full_replacement_horizon_transfer",
            "selection_endpoint": (
                "four registered fixed-window validation evaluations versus "
                "QK-only and repaired-V 20TPP, plus ordinary-dense terminal"
            ),
            "resolved_from_template": str(BASE.relative_to(ROOT)),
        }
    )
    return config


def build_plan(config_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "mai_124m_full_replacement_repaired_v_20tpp_plan_v1",
        "created_at": "2026-08-20",
        "status": "preregistered_pending_exact_config_mfu",
        "question": (
            "Does the complete repaired-V replacement retain zero-gap or "
            "practical-equivalence performance through 124M/20TPP?"
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
                "training and LR-decay horizon 2373 -> 9489",
                "warmup 23 -> 94",
                "fixed evaluation interval 594 -> 2373",
                "output identity and long-run monitoring policy",
            ],
        },
        "causal_contract": [
            "Do not change QK, V, attention-cproj, c_fc, or MLP-cproj representations.",
            "Preserve optimizer, LR, data, seeds, microbatch, accumulation, and evaluation set.",
            "Run fresh from initialization; do not resume or transplant the 5TPP checkpoint.",
            "No larger model or parallel arm is authorized by this transfer.",
        ],
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
        },
        "terminal_gate": {
            "zero_gap_closure_validation_ce": 3.1547,
            "practical_acceptance_maximum_validation_ce": 3.1647,
            "maximum_delta_to_qk_only_at_every_fixed_evaluation_ce": 0.0200,
            "maximum_delta_to_isolated_repaired_v_at_every_fixed_evaluation_ce": 0.0200,
            "require_all_fixed_evaluations_finite": True,
            "require_clean_exit": True,
            "threshold_changes_after_measurement": False,
            "zero_gap_classification": "FULL_REPLACEMENT_REPAIRED_V_ZERO_GAP_AT_124M_20TPP",
            "practical_pass_classification": "FULL_REPLACEMENT_REPAIRED_V_NEAR_DENSE_AT_124M_20TPP",
            "fail_classification": "FULL_REPLACEMENT_REPAIRED_V_HORIZON_NONTRANSFER_124M_20TPP",
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
                "Verify live state, loss, GPU health, and the active project "
                "note; terminal success must seal exact artifacts and choose "
                "only the next causally authorized full-replacement action."
            ),
        },
        "evidence": {
            "complete_replacement_5tpp": {
                "path": str(RESULT_5TPP.relative_to(ROOT)),
                "sha256": sha256(RESULT_5TPP),
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
            "exact_config_mfu_preflight": True,
            "training_before_mfu_pass": False,
            "one_fresh_124m_20tpp_training_after_mfu_pass": True,
            "automatic_rerun": False,
            "parallel_arm": False,
            "larger_model": False,
        },
        "scope_guards": [
            "The ambient int8 codecs are approximately 4x smaller than FP32 weights, not 200x latent generators.",
            "FP16 feedback residuals and dense FP32 Muon momentum remain optimizer state.",
            "Dense transient materialization and materialized inference FLOPs remain unchanged.",
            "A single 20TPP seed is horizon confirmation, not multi-seed final closure.",
        ],
    }


def main() -> None:
    config_bytes = encoded(build_config())
    OUTPUT.write_bytes(config_bytes)
    PLAN.write_bytes(encoded(build_plan(hashlib.sha256(config_bytes).hexdigest())))
    print(OUTPUT.relative_to(ROOT))
    print(PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
