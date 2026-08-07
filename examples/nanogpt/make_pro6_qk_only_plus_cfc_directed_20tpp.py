#!/usr/bin/env python3
"""Generate the preregistered QK-only plus procedural-c_fc 20TPP test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
ARTIFACT_DIR = CONFIG_DIR / "selection_artifacts"
PARENT = CONFIG_DIR / "pro6_mai_v3_124m_qk_only_qk64_outputgain_20tpp_lr24e4.json"
CONFIG = CONFIG_DIR / "pro6_mai_v3_124m_qk_only_plus_cfc_directed_20tpp_lr24e4.json"
PLAN = ARTIFACT_DIR / "124m_qk_only_plus_cfc_directed_20tpp_plan.json"
ACCOUNTING_CORRECTION = ARTIFACT_DIR / (
    "124m_qk_only_plus_cfc_directed_20tpp_accounting_correction.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parent = json.loads(PARENT.read_text())
    correction = {
        "schema_version": "mai_124m_qk_only_plus_cfc_directed_20tpp_accounting_correction_v1",
        "artifact_kind": "nondecision_preregistration_accounting_correction",
        "recorded_at": "2026-08-07T20:15:00+08:00",
        "classification": "CORRECT_REGISTERED_COUNT_BEFORE_SCIENTIFIC_LAUNCH",
        "invalid_description": {
            "text": (
                "The first preregistered config expected the QK-only registered count "
                "113,916,912 after enabling procedural c_fc. Exact construction instead "
                "reported 85,605,360 ordinary registered/trainable parameters because "
                "the persistent materialized c_fc and its custom optimizer state are "
                "owned outside the ordinary model.parameters() count."
            ),
            "config_sha256": "0c91de4f402379e41d7438b0f6ff2720d3637f3223c386196a848ae0cfb5aec4",
        },
        "authoritative_runtime_accounting": {
            "active_materialized_parameter_estimate_for_mfu": 124475904,
            "ordinary_registered_total_parameters": 85605360,
            "ordinary_registered_trainable_parameters": 85605360,
            "block_fht_generated_weight_elements_reported": 42467328,
            "block_fht_latent_elements_reported": 5007600,
            "ordinary_registered_reduction_vs_active_estimate": 38870544,
            "ordinary_registered_reduction_fraction_vs_active_estimate": 0.31227364293735116,
            "cfc_component_realized_algorithmic_parameter_reduction": 0,
            "cfc_materialized_matrix_elements": 28311552,
            "cfc_update_coordinate_reduction_fraction": 0.828125,
            "inference_parameter_reduction": 0,
            "inference_flop_reduction": 0,
            "scope": (
                "The ordinary PyTorch registered count falls because directed-product "
                "c_fc keeps its persistent materialized matrices and custom dense "
                "optimizer state outside model.parameters(). This is not an independent "
                "algorithmic trainable-parameter or optimizer-state reduction for c_fc. "
                "Generated weights are still materialized for dense inference GEMMs."
            ),
        },
        "invalidated_preflight": {
            "certificate_path": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_qk_only_plus_cfc_directed_20tpp/"
                "performance_preflight.json"
            ),
            "certificate_sha256": "dfa3d614761b67fe95e43c813ee74343305e096070fcb632ec4c0e39baed0ac0",
            "preflight_log_sha256": "6f30da44edf0d9b0b0fa5454c0dffa9dce56cb43a9cc71f7de929c90e8b13dc0",
            "mfu_fraction": 0.2725802865459523,
            "reason": (
                "Architecture, finite-state, native-kernel, and speed gates passed, but "
                "the exact config pinned an incorrect ordinary registered count. The "
                "certificate cannot authorize scientific training after config repair."
            ),
        },
        "unchanged_science": {
            "generated_attention": ["attn.c_attn.qk_headwise"],
            "procedural_mlp": ["mlp.c_fc"],
            "dense_residual_write_targets": ["attn.c_proj", "mlp.c_proj"],
            "dense_v": True,
            "terminal_validation_ce_maximum": 3.1538,
            "maximum_fixed_curve_gap_to_qk_only_parent": 0.005,
            "optimizer_data_seed_schedule": "unchanged",
            "scientific_parameter_updates_before_correction": 0,
            "threshold_changed_after_measurement": False,
        },
        "required_repair": {
            "pin_correction_in_config": True,
            "rerun_exact_config_mfu": True,
            "scientific_launch_before_repair": False,
        },
    }
    dump(ACCOUNTING_CORRECTION, correction)
    config = dict(parent)
    config.update(
        {
            "accounting_correction": str(ACCOUNTING_CORRECTION.relative_to(ROOT)),
            "accounting_correction_sha256": sha256(ACCOUNTING_CORRECTION),
            "block_fht_mlp_cfc_directed_product": True,
            "block_fht_mlp_cfc_directed_product_chunk_size": 256,
            "block_fht_mlp_cfc_directed_product_error_feedback": True,
            "block_fht_mlp_cfc_directed_product_error_feedback_decay": 1.0,
            "block_fht_mlp_cfc_directed_product_family_radius_ratio": 1.0,
            "block_fht_mlp_cfc_directed_product_ridge_ratio": 1e-6,
            "block_fht_mlp_cfc_directed_product_schedule": [22] * 6,
            "block_fht_mlp_cfc_functional_shear": False,
            "block_fht_native_extension_required": True,
            "candidate_learning_rate_resolution": (
                "freeze the accepted QK-only 20TPP and directed-product c_fc recipes: "
                "main LR 2.4e-3, AdamW fallback 0.3, Cayley subgroup scale 10/3, "
                "and c_fc error-feedback decay 1.0"
            ),
            "candidate_scope": (
                "From-scratch 124M/20TPP composition of the accepted generated-QK "
                "functional LWT endpoint with the accepted procedural c_fc in all "
                "layers. V, attention c_proj, and MLP c_proj stay dense Muon residual-"
                "write maps. This is a partial architecture, not full replacement."
            ),
            "confirmation_slot": "qk_only_plus_cfc_directed_20tpp",
            "confirmation_source": (
                "QK-only passed the full 20TPP curve at val 3.1488; directed-product "
                "c_fc improved its repaired-attention parent by 0.00196 CE and improved "
                "all four checkpoints in the residual-write-preserving joint test."
            ),
            "cfc_parent_config": (
                "examples/nanogpt/configs/"
                "pro6_mai_v3_124m_repairedfullattn_plus_cfconly_decay1_5tpp_lr24e4.json"
            ),
            "cfc_parent_config_sha256": (
                "baee2a5148f8e66bcd955680b39b7b6ccc7b3a7be00440a535a0dabf60a6c857"
            ),
            "cfc_parent_result": (
                "examples/nanogpt/configs/selection_artifacts/"
                "124m_repaired_attention_cfc_only_5tpp_result.json"
            ),
            "cfc_parent_result_sha256": (
                "309556d0290c4a51fc8535c42b41f4e2982dcb7e3c4a048ceed5dfd3bd734b5f"
            ),
            "cfc_component_parameter_reduction": 0,
            "control_only_runtime_guard": (
                "generate QK and use directed-product c_fc while V, attention c_proj, "
                "and MLP c_proj remain absent from generated/procedural target sets"
            ),
            "expected_registered_parameter_reduction_fraction_vs_preflight_active_estimate": 0.31227364293735116,
            "expected_registered_trainable_parameters": 85605360,
            "hpo_stage": "qk_only_plus_cfc_directed_124m_20tpp_confirmation",
            "inference_flop_reduction": 0,
            "inference_parameter_reduction": 0,
            "ladder_interpretation": (
                "one preregistered 124M/20TPP transfer discriminator; not a sweep"
            ),
            "ladder_role": "partial_architecture_qk_plus_cfc_transfer",
            "ladder_slot": "qk_only_plus_cfc_directed_20tpp",
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "Causal composition of two separately accepted feature-forming "
                    "procedures after the QK-only 20TPP closure."
                ),
                "recorded_at": "2026-08-07",
                "scope": "124M/20TPP generated QK plus procedural c_fc with dense residual writes",
            },
            "out_dir": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_qk_only_plus_cfc_directed_20tpp"
            ),
            "parent_fixed_validation_curve": [
                {"step": 2373, "validation_ce": 3.5038},
                {"step": 4746, "validation_ce": 3.3179},
                {"step": 7119, "validation_ce": 3.2046},
                {"step": 9489, "validation_ce": 3.1488},
            ],
            "practical_equivalence_nll": 0.005,
            "practical_equivalence_policy": (
                "Require exact-config MFU >=20%, finite fixed evaluations, terminal "
                "validation CE <=3.1538, and no fixed checkpoint more than +0.0050 CE "
                "behind the sealed QK-only 20TPP parent. No post-hoc threshold change, "
                "retry, parallel arm, or larger rung."
            ),
            "qk_only_20tpp_result": (
                "examples/nanogpt/configs/selection_artifacts/"
                "124m_attention_qk_only_lwt_20tpp_result.json"
            ),
            "qk_only_20tpp_result_sha256": (
                "faf150ce3f4c335947e67adc3415cc254839a7bb83350fdd9b7587511d0ffe2a"
            ),
            "recipe_resolution_stage": "qk_only_plus_cfc_directed_124m_20tpp",
            "residual_write_joint_5tpp_result_sha256": (
                "f264b334b86b1e7ac15fa83318e97ad4239779183152e936c25c5388e2d473c3"
            ),
            "selected_lwt_allocation": {
                "generated": ["attn.c_attn.qk_headwise"],
                "procedural_feature_map": ["mlp.c_fc"],
                "dense_muon": ["attn.c_attn.v", "attn.c_proj", "mlp.c_proj"],
                "reason": (
                    "Generate only the accepted functionally simple QK map and apply "
                    "the accepted c_fc tangent procedure; preserve both residual-write "
                    "projections and V as dense matrices."
                ),
            },
        }
    )
    dump(CONFIG, config)

    evidence = {
        "qk_only_20tpp_result": "124m_attention_qk_only_lwt_20tpp_result.json",
        "cfc_only_5tpp_result": "124m_repaired_attention_cfc_only_5tpp_result.json",
        "residual_write_joint_5tpp_result": "124m_residual_write_preserving_joint_5tpp_result.json",
        "tail2_cproj_rejection": "124m_repaired_attention_cfc_tail2_cproj_lwt_5tpp_result.json",
    }
    plan = {
        "schema_version": "mai_124m_qk_only_plus_cfc_directed_20tpp_plan_v1",
        "artifact_kind": "124m_qk_only_plus_cfc_directed_20tpp_preregistration",
        "registered_at": "2026-08-07T20:00:00+08:00",
        "status": "registered_before_performance_preflight_and_training",
        "scientific_question": (
            "Does the accepted procedural c_fc remain neutral or beneficial across "
            "20TPP when jointly trained from initialization with the accepted QK-only "
            "attention map while every residual-write projection remains dense?"
        ),
        "theory": {
            "manifold_match": (
                "The accepted QK chart matches the simpler causal score-function path; "
                "the c_fc directed-product update matches the observed low-dimensional "
                "weight trajectory in the dense tangent gauge rather than imposing a "
                "fixed random chart."
            ),
            "composition_evidence": (
                "At 5TPP, adding c_fc to generated attention improved all four fixed "
                "checkpoints by 0.0015-0.0018 CE, so the 20TPP test is a horizon-transfer "
                "confirmation rather than a new structural guess."
            ),
            "residual_write_boundary": (
                "V and both c_proj matrices stay dense because the measured activation "
                "and trajectory geometry, endpoint swaps, and repeated interventions "
                "localized failures to residual-stream writing rather than feature formation."
            ),
            "claim_boundary": (
                "A pass establishes a partial QK+c_fc training architecture only. The "
                "materialized c_fc and custom optimizer state persist outside the ordinary "
                "registered count, so c_fc has no independent realized algorithmic "
                "parameter reduction. Inference parameter/FLOP reduction is zero, and "
                "neither full attention nor full MLP replacement is claimed."
            ),
        },
        "immutable_evidence": {
            key: {"path": f"examples/nanogpt/configs/selection_artifacts/{name}",
                  "sha256": sha256(ARTIFACT_DIR / name)}
            for key, name in evidence.items()
        },
        "accounting_correction": {
            "path": str(ACCOUNTING_CORRECTION.relative_to(ROOT)),
            "sha256": sha256(ACCOUNTING_CORRECTION),
        },
        "candidate": {
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG),
            "parent_config": str(PARENT.relative_to(ROOT)),
            "parent_config_sha256": sha256(PARENT),
            "generated_attention": ["attn.c_attn.qk_headwise"],
            "procedural_mlp": ["mlp.c_fc"],
            "dense_muon": ["attn.c_attn.v", "attn.c_proj", "mlp.c_proj"],
            "horizon": {"planned_tpp": 20.0, "max_iters": 9489,
                        "fixed_evaluation_steps": [2373, 4746, 7119, 9489]},
            "forbidden_changes": [
                "generated V or either c_proj", "different QK or c_fc capacity",
                "optimizer or learning-rate changes", "automatic retry or parallel arm",
                "larger rung or post-result threshold change",
            ],
        },
        "identity": {
            "dataset_manifest_sha256": config["data_manifest_sha256"],
            "fixed_eval_indices_sha256": "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747",
            "entrypoint": "examples.nanogpt.train", "host": "PRO6", "gpu": 0,
            "native_blockfht_required": True,
        },
        "decision_rule": {
            "qk_only_parent_validation_ce": [3.5038, 3.3179, 3.2046, 3.1488],
            "dense_terminal_validation_ce": 3.1547,
            "terminal_validation_ce_maximum": 3.1538,
            "maximum_fixed_curve_gap_to_qk_only_parent": 0.005,
            "all_fixed_losses_must_be_finite": True,
            "pass": (
                "Accept QK-only plus directed-product c_fc as the 124M/20TPP partial "
                "architecture endpoint; do not call it full replacement."
            ),
            "fail": (
                "Reject 20TPP transfer of c_fc composition and retain QK-only as the "
                "accepted endpoint; launch no variant sweep."
            ),
            "threshold_changed_after_measurement": False,
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2, "warmup_updates": 1, "timed_updates": 8,
            "all_losses_finite": True, "direct_foreground_polling": True,
            "watchdog": False, "callbacks": False,
        },
        "monitoring": {
            "expected_duration": "approximately 5-6 hours from exact-config preflight",
            "policy": "one idempotent aggregate watchdog",
            "callbacks": [20, 50, 100, "error or actionable stall"],
            "heartbeat_minutes": 90,
            "heartbeat_resets_after_progress_callback": True,
            "callback_endpoint": "http://127.0.0.1:8766/send-opencode-test",
            "callback_action_prompt": (
                "@Codex verify live or terminal identity, fixed losses, GPU health, "
                "provenance, and checkpoint hashes; update durable notes, classify the "
                "frozen gate, and continue autonomously."
            ),
        },
        "authorization": {
            "one_exact_config_mfu_preflight": True,
            "one_scientific_run_after_mfu_pass": True,
            "automatic_rerun": False, "parallel_arm": False, "larger_rung": False,
        },
    }
    dump(PLAN, plan)


if __name__ == "__main__":
    main()
