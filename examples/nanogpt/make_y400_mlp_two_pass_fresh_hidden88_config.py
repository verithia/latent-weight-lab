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


def main() -> None:
    validate_inputs()
    config = make_config()
    config_data = json_bytes(config)
    OUTPUT_CONFIG.write_bytes(config_data)
    plan = make_plan(sha256_bytes(config_data))
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    print(
        json.dumps(
            {
                "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
                "config_sha256": sha256_bytes(config_data),
                "plan": str(OUTPUT_PLAN.relative_to(ROOT)),
                "plan_sha256": sha256_file(OUTPUT_PLAN),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
