#!/usr/bin/env python3
"""Generate the immutable two-pass fresh-hidden88 config and MFU plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
PARENT_CONFIG = (
    CONFIGS
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "statelessfresh64_0p5tpp.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "twopassfresh88_0p5tpp.json"
)
OUTPUT_PLAN = (
    ARTIFACTS
    / "124m_mlp_two_pass_fresh_hidden88_mfu_plan.json"
)
OUTPUT_MFU_RESULT = (
    ARTIFACTS
    / "124m_mlp_two_pass_fresh_hidden88_mfu_result.json"
)
OUTPUT_TRAINING_PLAN = (
    ARTIFACTS
    / "124m_mlp_two_pass_fresh_hidden88_training_plan.json"
)
OUTPUT_TRAINING_RESULT = (
    ARTIFACTS
    / "124m_mlp_two_pass_fresh_hidden88_training_result.json"
)
IMPLEMENTATION_COMMIT = (
    "1f7e1c6640450b0938545bc7efe3c911f1e0ac33"
)
PARENT_CONFIG_SHA256 = (
    "9051dc9535f5a543d006dad458f25190a81592a1166f3df9694f2e734723e1d4"
)
CAPACITY_RESULT = (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_fast_fresh_hidden_capacity_holdout_result.json"
)
CAPACITY_RESULT_SHA256 = (
    "cab549170a3cbfb1a0228e374bca056834f0d64e6a242632783e511520b437b3"
)
SOURCE_HASHES = {
    "examples/nanogpt/csrc/task_edge_coloring.cpp": (
        "6b9e0c795e601eb0a53934bdaa79f898c7ffcd51988b0e5891f189cc27f2e67e"
    ),
    "examples/nanogpt/fast_task_matching.py": (
        "1e451f3d0309299d8f9dfd4494f1087fb35f282b7a1fca49bb60322dd3529ce1"
    ),
    "examples/nanogpt/model.py": (
        "311a9f3b3f2c210602eab6d6bdca93194218e493c150d8428dc545e1ef565295"
    ),
    "examples/nanogpt/muon.py": (
        "532e172d91306d12284507c96aa3176792b33eb657f568512ce278bb5a9874ff"
    ),
    "examples/nanogpt/muon_matched_givens.py": (
        "ce6490252e9c6d5b0b768a93df7924872a951ef19ff442e14a8257bc899aaa04"
    ),
    "examples/nanogpt/parameter_trajectory.py": (
        "2bcbcb806d0e52652f2ac0ecbcc9ae9b8ca51800bfa8a57cfcc4cb0ac5113f1d"
    ),
    "examples/nanogpt/train.py": (
        "a59f76f61d2a91070198693e519bea3899c850bba540eafe07f0f1d4b7effd6b"
    ),
    "latent_weight_lab/block_fht.py": (
        "5646a93f518f586417496332196d41ddd547fb8a5606ff308e3b116204482b1b"
    ),
}
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_mlp_two_pass_fresh_screen"
)
RUN_NAME = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "twopassfresh88_0p5tpp"
)
CERTIFICATE = (
    f"{OUTPUT_ROOT}/performance_preflight_two_pass_fresh_hidden88.json"
)
CERTIFICATE_SHA256 = (
    "b459a0c0baf5b0b3b0b7df47d3b2e233f37a07e85f3ce917c8183cb4dce7904c"
)
MFU_PLAN_SHA256 = (
    "530be5283129ea3fa84d1725355fa2dcacbea4ff531f4f19c1fd7b8e9a5a816c"
)
TRAINING_PLAN_SHA256 = (
    "17156740a9d8fa54857cccb53521ac770bff67c1a3138ff29036b728c95fbb84"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()


def validate_inputs() -> None:
    if sha256_file(PARENT_CONFIG) != PARENT_CONFIG_SHA256:
        raise RuntimeError("parent config hash drifted")
    if sha256_file(ROOT / CAPACITY_RESULT) != CAPACITY_RESULT_SHA256:
        raise RuntimeError("capacity result hash drifted")
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"runtime source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(PARENT_CONFIG.read_text())
    config.update(
        {
            "block_fht_mlp_cproj_muon_matched_givens_residual_stages": 24,
            "candidate_scope": (
                "selected full-attention replacement plus folded-base "
                "two-pass fresh hidden88 mlp.c_proj chart: a 64-stage "
                "parent fit to the exact scheduled Muon update followed by "
                "a newly selected 24-stage fit to the materialized parent "
                "residual on every update; mlp.c_fc remains dense"
            ),
            "failed_mfu_preflight": None,
            "hpo_stage": (
                "causal_two_pass_fresh_hidden88_integration_and_"
                "mfu_qualification"
            ),
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": SOURCE_HASHES,
            "implementation_test_evidence": {
                "focused_resume_rng": (
                    "42 passed, 40 subtests passed"
                ),
                "test_paths": [
                    "examples/nanogpt/test_muon_matched_givens.py",
                    "examples/nanogpt/test_fast_task_matching.py",
                    "examples/nanogpt/test_train_rng.py",
                    (
                        "examples/nanogpt/"
                        "test_verify_resume_checkpoint_envelope.py"
                    ),
                ],
            },
            "ladder_role": (
                "mlp_cproj_two_pass_fresh_hidden88_smallest_rung_screen"
            ),
            "mfu_measurement_protocol": (
                "foreground real-training preflight with 1 warmup and 8 "
                "timed updates; exact Muon direction, 64-stage parent "
                "selection/fit/materialization, fresh 24-stage residual "
                "selection/fit, native validation, and folded update all "
                "execute on every measured update"
            ),
            "mfu_preflight_certificate": CERTIFICATE,
            "muon_matched_givens_representation": {
                "angle_formula": (
                    "closed-form diagonal tangent metric; parent fits the "
                    "current scheduled exact-Muon update and residual pass "
                    "fits the exact post-parent residual"
                ),
                "coordinate_fraction_per_cproj": 88 / 1536,
                "coordinates_per_layer": 88 * 1536,
                "dense_folded_base_buffer": True,
                "dense_residual": False,
                "exact_muon_momentum": True,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "matching_neighbors": 64,
                "matching_policy": (
                    "stateless two-pass topology on every update: fresh "
                    "hidden64 from the exact current Muon direction, then "
                    "fresh hidden24 from the materialized parent residual"
                ),
                "matching_refresh_updates": 1,
                "parent_matching_stages": 64,
                "residual_matching_stages": 24,
                "total_matching_stages": 88,
                "native_matcher_source_sha256": SOURCE_HASHES[
                    "examples/nanogpt/csrc/task_edge_coloring.cpp"
                ],
                "native_output_validation": True,
                "persistent_resume_state": [
                    "folded weight",
                    "parent and residual last angles",
                    "optimizer step",
                    (
                        "last parent and residual selected permutations "
                        "and inverse permutations for exact observability"
                    ),
                    (
                        "last refresh step and refresh count for exact "
                        "observability"
                    ),
                    "matching-valid flag",
                    "exact Muon momentum buffer",
                ],
                "persistent_selected_connectivity": False,
                "scope_limit": (
                    "compresses trainable coordinates, not materialized "
                    "checkpoint size or c_proj inference FLOPs"
                ),
            },
            "optimizer_assignment_expected": (
                "dense GPT matrices use Muon; attention BlockFHT latents "
                "and ordinary scalars use AdamW fallback; each mlp.c_proj "
                "folded buffer receives exact Muon momentum/polar direction "
                "projected first to a fresh 64-stage chart and then to a "
                "fresh 24-stage chart of the exact post-parent residual"
            ),
            "out_dir": f"{OUTPUT_ROOT}/{RUN_NAME}",
            "parent_capacity_result": CAPACITY_RESULT,
            "parent_capacity_result_sha256": CAPACITY_RESULT_SHA256,
            "registered_resume_protocol": (
                "atomic_latest_checkpoint_v2_with_pre_current_batch_and_"
                "full_rng_state_plus_parent_and_residual_muon_matched_"
                "givens_buffers_and_momentum"
            ),
            "screen_only_resolution": (
                "the immutable config is executable only for the "
                "preregistered directly polled MFU qualification; no "
                "scientific training is authorized until a passing "
                "exact-config certificate and a separate registered "
                "training plan exist"
            ),
        }
    )
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = (
        "/root/userdata/MappingNetworks/"
        "latent-weight-lab-mlp-product-fht/"
        "examples/nanogpt/configs/"
        + OUTPUT_CONFIG.name
    )
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=1",
        "/root/userdata/MappingNetworks/.venv-gpt2/bin/python",
        "-u",
        "-m",
        "examples.nanogpt.mfu_preflight",
        "--config",
        remote_config,
        "--output",
        CERTIFICATE,
        "--min-fraction",
        "0.2",
        "--warmup-updates",
        "1",
        "--timed-updates",
        "8",
    ]
    return {
        "schema_version": (
            "mai_124m_mlp_two_pass_fresh_hidden88_mfu_plan_v1"
        ),
        "status": (
            "registered_after_heldout_geometry_selection_and_production_"
            "implementation_before_real_training_path_mfu_measurement"
        ),
        "scientific_question": (
            "Can the held-out-selected two-pass fresh hidden88 c_proj "
            "optimizer execute its complete real training update at or "
            "above the mandatory 20% model-FLOP utilization floor?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "model_tier": "124m",
            "planned_tpp": 0.5,
            "parent_stages": 64,
            "residual_stages": 24,
            "matching_neighbors": 64,
            "coordinates_per_layer": 88 * 1536,
            "coordinate_fraction_per_cproj": 88 / 1536,
        },
        "parent_evidence": {
            "heldout_capacity_result": CAPACITY_RESULT,
            "heldout_capacity_result_sha256": CAPACITY_RESULT_SHA256,
            "decision": (
                "SELECT_TWO_PASS_FRESH_HIDDEN_FOR_IMPLEMENTATION_PREFLIGHT"
            ),
            "selected_candidate": "fresh_hidden88",
            "scope": (
                "authorizes production implementation and this MFU "
                "preflight only, not scientific training"
            ),
        },
        "measured_path": {
            "optimizer": "exact Muon momentum and polar direction",
            "parent": (
                "fresh 64-stage native-selected hidden chart and "
                "diagonal-metric finite rotation"
            ),
            "residual": (
                "materialize parent, subtract it from the exact scheduled "
                "update, then fresh-select and fit 24 residual stages"
            ),
            "weight_decay": "same decoupled folded-weight path as control",
            "forward_backward": "real CUDA BF16 124M training batches",
            "future_information_used": False,
            "learned_dense_basis": False,
            "dense_residual": False,
            "lora_adapter": False,
        },
        "protocol": {
            "host": "Y400",
            "gpu": 1,
            "warmup_updates": 1,
            "timed_updates": 8,
            "minimum_mfu_fraction": 0.2,
            "denominator": (
                "same-host empirical BF16 8192-square tensor-core GEMM peak"
            ),
            "execution": (
                "direct foreground process polled by the agent through "
                "terminal exit"
            ),
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "heartbeat": False,
            "pro6": False,
            "do_not_disturb": (
                "existing 985M full-attention run on Y400 GPU0"
            ),
            "command": command,
            "certificate": CERTIFICATE,
        },
        "decision_rule": {
            "pass": (
                "exit zero, finite complete certificate bound to the exact "
                "config hash, all 8 timed real-training updates present, "
                "native matcher outputs validated for both passes, and "
                "measured MFU >= 0.20"
            ),
            "reject": (
                "MFU < 0.20, nonfinite/divergent training path, incomplete "
                "timing, native validation failure, provenance mismatch, "
                "or nonzero exit"
            ),
            "threshold_changes_after_measurement": False,
        },
        "authorization": {
            "scope": (
                "exactly one directly polled MFU preflight under the pinned "
                "hidden88 config"
            ),
            "scientific_training_authorized": False,
            "larger_rung_authorized": False,
            "additional_structure_authorized": False,
        },
    }


def make_mfu_result(config_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": (
            "mai_124m_mlp_two_pass_fresh_hidden88_mfu_result_v1"
        ),
        "recorded_at": "2026-07-30",
        "plan": {
            "path": str(OUTPUT_PLAN.relative_to(ROOT)),
            "sha256": MFU_PLAN_SHA256,
        },
        "config": {
            "path": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "sha256": config_sha256,
        },
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "execution": (
            "Direct foreground Y400 GPU1 real CUDA BF16 training-path "
            "preflight, polled through exit; no watchdog, callback, queue "
            "worker, heartbeat, or PRO6 execution."
        ),
        "certificate": {
            "path": CERTIFICATE,
            "sha256": CERTIFICATE_SHA256,
            "preflight_log_sha256": (
                "bdeeb86f903dfa7d9d040eb2b94a047580bd5269782ceb233e150846dacdb8d0"
            ),
            "schema_version": "nanogpt_mfu_preflight_v1",
        },
        "protocol": {
            "warmup_updates": 1,
            "timed_updates": 8,
            "parent_matching_stages": 64,
            "residual_matching_stages": 24,
            "matching_refresh_interval_updates": 1,
            "minimum_mfu_fraction": 0.2,
            "denominator": (
                "same-host empirical BF16 8192-square tensor-core GEMM peak"
            ),
        },
        "measurement": {
            "passed": True,
            "empirical_bf16_peak_tflops": 657.0719597441285,
            "active_params_6n_estimate": 124475904,
            "tokens_per_second": 199744.51875,
            "average_iter_ms": 1312.44375,
            "model_tflops": 149.18027724270718,
            "mfu_fraction": 0.22703795989224637,
            "headroom_over_gate_percentage_points": (
                2.7037959892246364
            ),
            "peak_mib": 31688.21,
            "timing_breakdown_ms": {
                "data": 1.65375,
                "flush": 9.96375,
                "forward_backward": 883.2487500000001,
                "gradient_postprocess": 2.84375,
                "optimizer": 404.25499999999994,
                "other": 2.0237499999999997,
                "prepare": 8.458749999999998,
            },
        },
        "native_confirmation": {
            "parent_selector": "fast_fresh_single_pass",
            "residual_selector": "fast_fresh_residual_pass",
            "parent_and_residual_output_validated": True,
            "source_sha256": SOURCE_HASHES[
                "examples/nanogpt/csrc/task_edge_coloring.cpp"
            ],
            "library_sha256": (
                "5aead912a93ca2ff4000632cd906b3a6622fa8cba6e82e0aa86a6fae94302e73"
            ),
            "coordinates_per_layer": 88 * 1536,
        },
        "decision": {
            "classification": (
                "AUTHORIZE_ONE_124M_0P5TPP_TWO_PASS_FRESH_HIDDEN88_RUN"
            ),
            "registered_mfu_gate_passed": True,
            "scope": (
                "Exactly one directly polled 238-update run under the "
                "immutable config and a separately registered terminal-CE "
                "plan; no other MLP structure or larger rung is authorized."
            ),
        },
    }


def make_training_plan(
    config_sha256: str,
    mfu_result_sha256: str,
) -> dict[str, Any]:
    remote_repo = (
        "/root/userdata/MappingNetworks/"
        "latent-weight-lab-mlp-product-fht"
    )
    remote_config = (
        f"{remote_repo}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    run_id = "y400_mai_v3_124m_twopassfresh88_scientific_v1"
    command = [
        "env",
        (
            "PYTHON_BIN=/root/userdata/MappingNetworks/"
            ".venv-gpt2/bin/python"
        ),
        "bash",
        f"{remote_repo}/examples/nanogpt/launch_y400_ladder_detached.sh",
        "--foreground",
        remote_config,
        "1",
        run_id,
        "/root/userdata/MappingNetworks",
    ]
    return {
        "schema_version": (
            "mai_124m_mlp_two_pass_fresh_hidden88_training_plan_v1"
        ),
        "status": (
            "registered_after_heldout_geometry_and_exact_config_mfu_"
            "qualification_before_scientific_training"
        ),
        "scientific_question": (
            "Does the held-out-selected two-pass fresh hidden88 c_proj "
            "chart close the remaining 124M loss gap to within +0.1 CE of "
            "the accepted full-attention-only control without a learned "
            "dense basis, dense residual, or LoRA adapter?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "model_tier": "124m",
            "planned_tpp": 0.5,
            "max_iters": 238,
            "parent_stages": 64,
            "residual_stages": 24,
            "matching_neighbors": 64,
            "refresh_interval_updates": 1,
            "coordinates_per_layer": 88 * 1536,
            "coordinate_fraction_per_cproj": 88 / 1536,
        },
        "causal_optimizer_protocol": {
            "matching": (
                "At every update, score hidden-channel edges from that "
                "update's exact coherent Muon polar direction, select and "
                "fit 64 parent stages, materialize the finite parent update, "
                "then select and fit 24 new stages from its exact residual."
            ),
            "state_transition": (
                "Apply both finite Givens flows, apply decoupled weight "
                "decay, and fold the result into the persistent base buffer."
            ),
            "future_information_used": False,
            "persistent_selected_connectivity": False,
            "native_output_validation": True,
            "learned_dense_basis": False,
            "dense_residual": False,
            "lora_adapter": False,
            "mlp_cfc_replaced": False,
        },
        "parent_evidence": {
            "heldout_capacity_result": CAPACITY_RESULT,
            "heldout_capacity_result_sha256": CAPACITY_RESULT_SHA256,
            "mfu_result": str(OUTPUT_MFU_RESULT.relative_to(ROOT)),
            "mfu_result_sha256": mfu_result_sha256,
            "stateless_fresh64_training_result": (
                "examples/nanogpt/configs/selection_artifacts/"
                "124m_mlp_fast_fresh_training_result.json"
            ),
            "stateless_fresh64_validation_ce": 5.612201690673828,
        },
        "mfu_gate": {
            "minimum_fraction": 0.2,
            "result": str(OUTPUT_MFU_RESULT.relative_to(ROOT)),
            "result_sha256": mfu_result_sha256,
            "certificate": CERTIFICATE,
            "certificate_sha256": CERTIFICATE_SHA256,
            "measured_fraction": 0.22703795989224637,
            "passed": True,
        },
        "decision_rule": {
            "primary_metric": (
                "terminal fixed-window validation cross entropy at update 238"
            ),
            "accepted_attention_only_control": 5.4918,
            "accepted_attention_gap": 0.1,
            "success_ce_maximum": 5.5918,
            "stateless_fresh64_control": 5.612201690673828,
            "success": (
                "Stable terminal validation CE at most 5.5918."
            ),
            "directional_only": (
                "Stable terminal validation CE below 5.612201690673828 "
                "but above 5.5918."
            ),
            "reject": (
                "Terminal validation CE at least 5.612201690673828, "
                "NaN/Inf/divergence, incomplete terminal evaluation, "
                "provenance mismatch, or incomplete exact-resume state."
            ),
            "threshold_changes_after_result": False,
        },
        "execution": {
            "run_name": run_id,
            "host": "Y400",
            "gpu": 1,
            "command": command,
            "entrypoint": "examples.nanogpt.train",
            "launcher": (
                "examples/nanogpt/launch_y400_ladder_detached.sh "
                "--foreground"
            ),
            "artifact_prefix": (
                "/root/userdata/MappingNetworks/outputs/y400_ladder_runs/"
                f"{{logs,status,provenance}}/{run_id}_"
            ),
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "heartbeat": False,
            "pro6": False,
            "do_not_disturb": (
                "existing 985M full-attention run on Y400 GPU0"
            ),
        },
        "authorization": {
            "scope": (
                "Exactly one immutable 124M/0.5TPP two-pass fresh hidden88 "
                "candidate."
            ),
            "larger_rung_authorized": False,
            "additional_structure_authorized": False,
        },
        "scope_limit": (
            "This is a trainable-coordinate/manifold test. The dense folded "
            "c_proj base remains materialized in checkpoints and forward "
            "passes, so it does not claim checkpoint-size or inference-FLOP "
            "compression."
        ),
    }


def make_training_result(
    config_sha256: str,
    mfu_result_sha256: str,
) -> dict[str, Any]:
    run = (
        "y400_mai_v3_124m_twopassfresh88_scientific_v1_"
        "20260729T215022Z_114146"
    )
    ladder_root = (
        "/root/userdata/MappingNetworks/outputs/y400_ladder_runs"
    )
    output = (
        "/root/userdata/MappingNetworks/outputs/"
        "y400_mai_v3_mlp_two_pass_fresh_screen/"
        f"{RUN_NAME}"
    )
    terminal = 5.592058181762695
    attention = 5.4918
    threshold = 5.5918
    parent = 5.612201690673828
    return {
        "schema_version": (
            "mai_124m_mlp_two_pass_fresh_hidden88_training_result_v1"
        ),
        "recorded_at": "2026-07-30",
        "plan": {
            "path": str(OUTPUT_TRAINING_PLAN.relative_to(ROOT)),
            "sha256": TRAINING_PLAN_SHA256,
        },
        "parent_evidence": {
            "heldout_capacity_result": CAPACITY_RESULT,
            "heldout_capacity_result_sha256": CAPACITY_RESULT_SHA256,
            "mfu_result": str(OUTPUT_MFU_RESULT.relative_to(ROOT)),
            "mfu_result_sha256": mfu_result_sha256,
            "stateless_fresh64_validation": parent,
        },
        "config": {
            "path": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "sha256": config_sha256,
        },
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "execution_commit": (
            "c2cb932a29b2aafca58315d487937b36b8296e15"
        ),
        "execution": {
            "run_id": run,
            "run_name": (
                "y400_mai_v3_124m_twopassfresh88_scientific_v1"
            ),
            "host": "Y400",
            "gpu": 1,
            "mode": (
                "provenance-enforcing foreground run, directly polled "
                "through terminal evaluation"
            ),
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "heartbeat": False,
            "pro6": False,
            "parameter_updates": 238,
            "entrypoint": "examples.nanogpt.train",
            "command": [
                "/root/userdata/MappingNetworks/.venv-gpt2/bin/python",
                "-u",
                "-m",
                "examples.nanogpt.train",
                "--config",
                f"{ladder_root}/provenance/{run}.config.json",
            ],
            "polling_transport_events": [
                {
                    "event": (
                        "the originating SSH polling connection closed "
                        "after step 60"
                    ),
                    "training_process_interrupted": False,
                    "resume_or_relaunch": False,
                    "evidence": (
                        "the same PID/PGID continued through step 238 and "
                        "the launcher recorded one clean execution"
                    ),
                }
            ],
        },
        "identity": {
            "repository": (
                "/root/userdata/MappingNetworks/"
                "latent-weight-lab-mlp-product-fht"
            ),
            "dataset_manifest": (
                "/root/userdata/MappingNetworks/data/"
                "finewebedu_20b/manifest.json"
            ),
            "dataset_manifest_sha256": (
                "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
            ),
            "fixed_eval_indices_sha256": (
                "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
            ),
            "resolved_config_sha256": (
                "6837049deeb66455fbb298f645be02e5bb15f1088399f3f524aae22c715a989c"
            ),
            "provenance": f"{ladder_root}/provenance/{run}.json",
            "provenance_sha256": (
                "d51b762d873b1d0eca12fd09be09bc552369e2cb7ff77588ddf92da4e175c865"
            ),
            "runtime_source_hashes": SOURCE_HASHES,
        },
        "performance_gate": {
            "certificate": CERTIFICATE,
            "certificate_sha256": CERTIFICATE_SHA256,
            "minimum_mfu_fraction": 0.2,
            "measured_mfu_fraction": 0.22703795989224637,
            "tokens_per_second": 199744.51875,
            "average_iter_ms": 1312.44375,
            "optimizer_ms": 404.25499999999994,
            "peak_mib": 31688.21,
            "passed": True,
        },
        "loss": {
            "fixed_evaluations": [
                {"step": 0, "train": 10.9672, "validation": 10.9669},
                {"step": 60, "train": 6.3043, "validation": 6.3141},
                {"step": 120, "train": 5.8487, "validation": 5.8533},
                {"step": 180, "train": 5.6699, "validation": 5.6733},
                {
                    "step": 238,
                    "train": 5.5889,
                    "validation": terminal,
                },
            ],
            "attention_only_control_validation": attention,
            "accepted_attention_gap": 0.1,
            "registered_success_threshold": threshold,
            "stateless_fresh64_control_validation": parent,
            "candidate_minus_attention": terminal - attention,
            "candidate_minus_success_threshold": terminal - threshold,
            "candidate_minus_stateless_fresh64": terminal - parent,
            "fraction_of_stateless_fresh64_to_attention_gap_closed": (
                (parent - terminal) / (parent - attention)
            ),
        },
        "causal_structure": {
            "mlp_cfc_replaced": False,
            "mlp_cproj_replaced": True,
            "matching_refresh_interval_updates": 1,
            "parent_stages": 64,
            "residual_stages": 24,
            "matching_neighbors": 64,
            "coordinates_per_layer": 88 * 1536,
            "coordinate_fraction_per_cproj": 88 / 1536,
            "persistent_selected_connectivity": False,
            "learned_dense_basis": False,
            "dense_residual": False,
            "lora_adapter": False,
            "terminal_optimizer_steps": "12/12 equal 238",
            "terminal_refresh_counts": "12/12 equal 238",
            "terminal_last_refresh_steps": "12/12 equal 237",
            "terminal_matching_valid": "12/12 true",
        },
        "terminal_checkpoint": {
            "path": f"{output}/ckpt.pt",
            "sha256": (
                "423029ea02046b95a5f8f4b9f80c000816b0ebf51f6a9dfd66dcd84da0ed47c6"
            ),
            "metadata_sha256": (
                "b2930a84113e5d25f1302c81d966eb5b34e5155b752b7a39b15dcfa7219c4458"
            ),
            "schema": "nanogpt_exact_resume_v2",
            "next_iter": 238,
            "checkpoint_reason": "evaluation",
            "best_val_loss": terminal,
            "custom_module_count": 12,
            "parent_selected_permutations_per_layer_shape": [
                64,
                3072,
            ],
            "residual_selected_permutations_per_layer_shape": [
                24,
                3072,
            ],
            "parent_last_angles_per_layer_shape": [64, 1536],
            "residual_last_angles_per_layer_shape": [24, 1536],
            "recursive_momentum_buffer_count": 24,
            "exact_resume_state_complete": True,
            "execution_provenance_embedded": True,
        },
        "artifacts": {
            "foreground_log": f"{ladder_root}/logs/{run}.log",
            "foreground_log_sha256": (
                "ad5e68882af61af7cff7f8330b5da9d413c7178b90d2e22a5cf4f6f6ab27d304"
            ),
            "status": f"{ladder_root}/status/{run}.json",
            "status_sha256": (
                "7da7ef75d7d1f12d8b02d0a8c23506b0733849fa762b1a252d90e0cb9699f902"
            ),
            "archived_config": f"{ladder_root}/provenance/{run}.config.json",
            "archived_config_sha256": config_sha256,
            "archived_mfu_certificate": (
                f"{ladder_root}/provenance/{run}.mfu.json"
            ),
            "archived_mfu_certificate_sha256": CERTIFICATE_SHA256,
            "workspace_after_run_gib": 240,
        },
        "decision": {
            "classification": "DIRECTIONAL_ONLY",
            "registered_success_gate_passed": False,
            "registered_directional_gate_passed": True,
            "larger_rung_authorized": False,
            "additional_training_candidate_authorized": False,
            "interpretation": [
                (
                    "The held-out-selected residual 24-stage pass causally "
                    "improves validation CE by 0.0201435 over the exact "
                    "stateless-fresh64 parent."
                ),
                (
                    "The candidate is 0.1002582 CE behind attention-only "
                    "and misses the fixed +0.1 success boundary by only "
                    "0.0002582 CE; this is practically boundary-level but "
                    "must remain DIRECTIONAL_ONLY under the registered rule."
                ),
                (
                    "Hidden-side coordinate capacity was therefore a real "
                    "part of the c_proj deficit after direction staleness "
                    "was removed."
                ),
                (
                    "The run leaves c_fc dense, so it does not validate a "
                    "full MLP replacement or resolve the pre-GELU activation "
                    "spectrum problem."
                ),
            ],
            "next_action": (
                "Use this two-pass hidden88 c_proj as the fixed reference "
                "for a separately preregistered no-training c_fc manifold "
                "gate whose metric is activation/Jacobian-aware before any "
                "new 124M training run or larger rung."
            ),
        },
        "scope_limit": (
            "The folded c_proj base remains materialized in checkpoints and "
            "forward passes; this result concerns trainable-coordinate "
            "geometry and does not claim checkpoint-size or inference-FLOP "
            "compression."
        ),
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    config_data = json_bytes(config)
    OUTPUT_CONFIG.write_bytes(config_data)
    plan = make_plan(sha256_bytes(config_data))
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    if sha256_file(OUTPUT_PLAN) != MFU_PLAN_SHA256:
        raise RuntimeError("registered MFU plan hash drifted")
    mfu_result = make_mfu_result(sha256_bytes(config_data))
    mfu_result_data = json_bytes(mfu_result)
    OUTPUT_MFU_RESULT.write_bytes(mfu_result_data)
    training_plan = make_training_plan(
        sha256_bytes(config_data),
        sha256_bytes(mfu_result_data),
    )
    OUTPUT_TRAINING_PLAN.write_bytes(json_bytes(training_plan))
    if sha256_file(OUTPUT_TRAINING_PLAN) != TRAINING_PLAN_SHA256:
        raise RuntimeError("registered training plan hash drifted")
    training_result = make_training_result(
        sha256_bytes(config_data),
        sha256_bytes(mfu_result_data),
    )
    OUTPUT_TRAINING_RESULT.write_bytes(json_bytes(training_result))
    print(
        json.dumps(
            {
                "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
                "config_sha256": sha256_bytes(config_data),
                "plan": str(OUTPUT_PLAN.relative_to(ROOT)),
                "plan_sha256": sha256_file(OUTPUT_PLAN),
                "mfu_result": str(
                    OUTPUT_MFU_RESULT.relative_to(ROOT)
                ),
                "mfu_result_sha256": sha256_file(OUTPUT_MFU_RESULT),
                "training_plan": str(
                    OUTPUT_TRAINING_PLAN.relative_to(ROOT)
                ),
                "training_plan_sha256": sha256_file(
                    OUTPUT_TRAINING_PLAN
                ),
                "training_result": str(
                    OUTPUT_TRAINING_RESULT.relative_to(ROOT)
                ),
                "training_result_sha256": sha256_file(
                    OUTPUT_TRAINING_RESULT
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
