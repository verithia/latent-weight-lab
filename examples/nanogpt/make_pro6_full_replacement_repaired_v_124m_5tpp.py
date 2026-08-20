#!/usr/bin/env python3
"""Register the smallest full-replacement recombination with repaired V."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_errorfeedback_5tpp.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullreplacement_repaired_v_int8_errorfeedback_5tpp.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_replacement_repaired_v_5tpp_plan.json"
)
FULL_REPLACEMENT_PARENT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_mlp_int8_lattice_errorfeedback_5tpp_training_result.json"
)
V_REPAIR_5TPP = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_v_int8_lattice_errorfeedback_5tpp_training_result.json"
)
V_REPAIR_20TPP = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_v_int8_lattice_errorfeedback_20tpp_training_result.json"
)
DENSE_REPLAY = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_dense_5tpp_replay_result.json"
)
REMOTE_OUTPUT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_124m_fullreplacement_repaired_v_int8_errorfeedback_5tpp"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())

    # Preserve the accepted QK64 chart while removing only Cayley-V.
    qk_only = ["attn.c_attn.qk_headwise"]
    config["block_fht_targets"] = qk_only
    config["block_fht_attn_cayley_targets"] = qk_only
    config["block_fht_attn_cayley_bilateral_targets"] = qk_only
    config["block_fht_attn_cayley_output_targets"] = qk_only
    config["block_fht_attn_cayley_ranks"] = {
        "attn.c_attn.qk_headwise": 64
    }
    config["block_fht_output_gain_targets"] = qk_only

    # The only new scientific component: the twice-confirmed V codec.
    config["block_fht_attn_v_int8_lattice"] = True
    config["block_fht_attn_v_int8_lattice_block_size"] = 4096
    config["block_fht_attn_v_int8_lattice_seed"] = 161804
    config["block_fht_attn_v_int8_lattice_error_feedback"] = True

    config.update(
        {
            "candidate_scope": (
                "Fresh 124M/5TPP recombination of all accepted persistent-state "
                "components: QK64 Cayley mapping, ambient V signed-int8 lattice "
                "with FP16 temporal error feedback, attention-c_proj signed-int8 "
                "lattice without feedback, and c_fc/c_proj signed-int8 lattices "
                "with FP16 temporal error feedback."
            ),
            "confirmation_slot": "full_replacement_repaired_v_124m_5tpp",
            "confirmation_source": (
                "V temporal repair is QK-only-equivalent at both 5TPP and "
                "20TPP; the prior combined attention/full-MLP codec is already "
                "within +0.009562 CE of ordinary dense at 5TPP."
            ),
            "hpo_stage": "full_replacement_repaired_v_124m_5tpp_recombination",
            "ladder_role": "smallest_full_replacement_repaired_v_recombination",
            "launch_block_reason": None,
            "launch_ready": True,
            "mfu_preflight_certificate": f"{REMOTE_OUTPUT}/performance_preflight.json",
            "monitoring_policy": (
                "foreground-poll the <=5 minute exact-config MFU gate; after "
                "a pass, use one idempotent terminal/error-only @Codex "
                "watchdog for the expected 1-2 hour scientific run"
            ),
            "out_dir": f"{REMOTE_OUTPUT}/scientific",
            "practical_equivalence_nll": 0.01,
            "practical_equivalence_policy": (
                "Zero-gap closure is terminal validation CE <=3.5402; practical "
                "acceptance is <=3.5502 (ordinary dense plus 0.0100) and no "
                "registered checkpoint more than +0.0200 above the sealed "
                "Cayley-V combined parent."
            ),
            "recipe_resolution_dependency": (
                "sealed V-repair passes at 124M/5TPP and 20TPP plus sealed "
                "full-attention/full-MLP 5TPP parent"
            ),
            "recipe_resolution_stage": "accepted_component_recombination",
            "selection_endpoint": (
                "four registered fixed-window validation evaluations versus "
                "ordinary dense and the sealed Cayley-V combined parent"
            ),
        }
    )
    return config


def build_plan(config_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "mai_124m_full_replacement_repaired_v_5tpp_plan_v1",
        "created_at": "2026-08-20",
        "status": "preregistered_pending_exact_config_mfu",
        "question": (
            "Does the validated temporal V codec recombine with the accepted "
            "attention-c_proj and full-MLP codecs without reopening a loss gap?"
        ),
        "causal_contract": [
            "Change only the V representation from Cayley rank-16 to block-4096 signed-int8 plus FP16 temporal error feedback.",
            "Preserve QK64, attention-c_proj lattice without feedback, both MLP lattices with feedback, optimizer, LR, data, seeds, horizon, and fixed evaluations.",
            "Run fresh from initialization; do not resume or transplant any prior checkpoint.",
            "This is the smallest 124M/5TPP recombination; no 20TPP or larger-model promotion is implicit.",
        ],
        "candidate": {
            "generator": str(Path(__file__).resolve().relative_to(ROOT)),
            "config": str(OUTPUT.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "model_size": "124M",
            "planned_tpp": 5.0,
            "max_iters": 2373,
            "fresh_from_scratch": True,
            "persistent_attention_state": (
                "QK64 mapping plus signed-int8 V with feedback plus signed-int8 c_proj"
            ),
            "persistent_mlp_state": (
                "signed-int8 c_fc and c_proj, each with FP16 temporal feedback"
            ),
        },
        "frozen_comparators": {
            "ordinary_dense": {
                "steps": [594, 1188, 1782, 2373],
                "validation_ce": [4.1742, 3.7631, 3.6039, 3.5402],
            },
            "cayley_v_combined_parent": {
                "steps": [594, 1188, 1782, 2373],
                "validation_ce": [4.0566, 3.7438, 3.6074, 3.549762010574341],
            },
        },
        "terminal_gate": {
            "zero_gap_closure_validation_ce": 3.5402,
            "practical_acceptance_maximum_validation_ce": 3.5502,
            "maximum_delta_to_ordinary_dense_ce": 0.0100,
            "maximum_delta_to_cayley_v_combined_parent_at_every_fixed_evaluation_ce": 0.0200,
            "secondary_strong_v_transfer_minimum_improvement_ce": 0.0100,
            "secondary_strong_v_transfer_maximum_validation_ce": 3.539762010574341,
            "secondary_strong_v_transfer_required_for_practical_acceptance": False,
            "require_clean_exit": True,
            "require_all_fixed_evaluations_finite": True,
            "threshold_changes_after_measurement": False,
            "zero_gap_classification": "FULL_REPLACEMENT_REPAIRED_V_ZERO_GAP_AT_124M_5TPP",
            "practical_pass_classification": "FULL_REPLACEMENT_REPAIRED_V_NEAR_DENSE_AT_124M_5TPP",
            "fail_classification": "FULL_REPLACEMENT_RECOMBINATION_REOPENS_GAP",
        },
        "execution_gate": {
            "host": "PRO6",
            "gpu": 0,
            "exact_config_mfu_minimum": 0.20,
            "mfu_gate_runtime": "foreground-polled; no watchdog or callback",
            "scientific_run_runtime": (
                "one idempotent terminal/error-only @Codex watchdog; no "
                "milestone or heartbeat callbacks"
            ),
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "callback_action_prompt": (
                "Verify terminal artifacts, fixed-checkpoint gaps, codec state, "
                "and hashes; seal the active project note, then choose only "
                "the next causally justified full-replacement action."
            ),
        },
        "evidence": {
            "full_replacement_cayley_v_parent": {
                "path": str(FULL_REPLACEMENT_PARENT.relative_to(ROOT)),
                "sha256": sha256(FULL_REPLACEMENT_PARENT),
            },
            "v_repair_5tpp": {
                "path": str(V_REPAIR_5TPP.relative_to(ROOT)),
                "sha256": sha256(V_REPAIR_5TPP),
            },
            "v_repair_20tpp": {
                "path": str(V_REPAIR_20TPP.relative_to(ROOT)),
                "sha256": sha256(V_REPAIR_20TPP),
            },
            "ordinary_dense_replay": {
                "path": str(DENSE_REPLAY.relative_to(ROOT)),
                "sha256": sha256(DENSE_REPLAY),
            },
        },
        "authorization": {
            "exact_config_mfu_preflight": True,
            "training_before_mfu_pass": False,
            "one_124m_5tpp_recombination_after_mfu_pass": True,
            "automatic_rerun": False,
            "automatic_20tpp": False,
            "larger_model": False,
        },
        "scope_guards": [
            "The ambient int8 codecs are approximately 4x smaller than FP32 weights, not 200x latent generators.",
            "FP16 feedback residuals and dense FP32 Muon momentum are optimizer state and must be counted.",
            "Dense transient materialization and materialized inference FLOPs remain unchanged.",
        ],
    }


def main() -> None:
    config = build_config()
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    PLAN.write_text(
        json.dumps(build_plan(sha256(OUTPUT)), indent=2, sort_keys=True) + "\n"
    )
    print(OUTPUT.relative_to(ROOT))
    print(PLAN.relative_to(ROOT))


if __name__ == "__main__":
    main()
