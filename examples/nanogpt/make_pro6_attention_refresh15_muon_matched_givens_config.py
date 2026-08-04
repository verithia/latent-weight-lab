#!/usr/bin/env python3
"""Register the causal refresh-15 attention Givens screen."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_"
    "cproj8_targeted_bilateral_fullcayleylr_0p5tpp_lr24e4.json"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_refresh15_muon_matched_givens_"
    "0p5tpp_lr24e4.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_refresh15_muon_matched_givens_plan.json"
)
EXPECTED_SOURCE_SHA256 = (
    "0c1bf038cbcf2f3d81570987912a5d1fe9a338b1b2ce4cb0e5ed2c54926660de"
)
PARENT_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_targeted_bilateral_full_cayley_lr_0p5tpp_result.json"
)
EXPECTED_PARENT_RESULT_SHA256 = (
    "e7c3705ae5ea05042ad216f9bc559ca098c576406a4269197adbad2091375bce"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
REMOTE_REPO = Path("/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab")
RUN_NAME = (
    "pro6_mai_v3_124m_fullattn_refresh15_muon_matched_givens_0p5tpp"
)
OUTPUT_ROOT = (
    Path("/mnt/ssd-data/orj/MappingNetworks/outputs")
    / "pro6_mai_v3_attention_refresh15_candidate"
    / RUN_NAME
)
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
TARGETS = ["attn.c_attn.qk", "attn.c_attn.v", "attn.c_proj"]
STAGES = 64
NEIGHBORS = 128
REFRESH_INTERVAL = 15
MATCHING_SEED = 161803
SEED_STEP_STRIDE = 8192
MATERIALIZED_ATTENTION_WEIGHTS = 28_311_552
UPDATE_COORDINATES = 1_769_472


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(SOURCE_CONFIG) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("immutable attention parent config drifted")
    if sha256_file(PARENT_RESULT) != EXPECTED_PARENT_RESULT_SHA256:
        raise RuntimeError("immutable attention parent result drifted")
    config = copy.deepcopy(source)
    for key in tuple(config):
        if key.startswith("block_fht_attn_cayley_"):
            config.pop(key)
    config.update(
        {
            "block_fht_targets": TARGETS,
            "block_fht_attn_muon_matched_givens_targets": TARGETS,
            "block_fht_attn_muon_matched_givens_stages": STAGES,
            "block_fht_attn_muon_matched_givens_neighbors": NEIGHBORS,
            "block_fht_attn_muon_matched_givens_refresh_interval": (
                REFRESH_INTERVAL
            ),
            "block_fht_attn_muon_matched_givens_fast_matching": True,
            "block_fht_attn_muon_matched_givens_seed": MATCHING_SEED,
            "block_fht_attn_muon_matched_givens_seed_step_stride": (
                SEED_STEP_STRIDE
            ),
            "block_fht_output_gain_targets": [],
            "block_fht_input_gain_targets": [],
            "block_fht_cache_weights": False,
            "block_fht_native_extension_required": False,
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "execution_host": "PRO6",
            "host_transfer_source_config": str(
                SOURCE_CONFIG.relative_to(ROOT)
            ),
            "host_transfer_policy": (
                "preserve the parent's 124M/0.5-TPP data, model, Muon, "
                "learning-rate, warmup, decay, and fixed-evaluation settings; "
                "replace only the attention generator/chart and PRO6 paths"
            ),
            "hpo_stage": (
                "attention_causal_refresh15_muon_matched_givens_124m_0p5tpp"
            ),
            "ladder_slot": "refresh15_muon_matched_givens",
            "confirmation_slot": "refresh15_muon_matched_givens",
            "confirmation_source": (
                "the preregistered dense replay/oracle selected 15 updates: "
                "15.0759% future-chord recovery, 2.1804x equal-coordinate "
                "random, and 10.4498% minimum target recovery"
            ),
            "candidate_scope": (
                "Fold each full attention matrix through sparse task-selected "
                "Givens rotations. Refresh integer connectivity every 15 "
                "optimizer updates and refit rotation angles every update. "
                "No learned dense basis, additive adapter, LoRA branch, "
                "Cayley factor, output gain, or Mapping Loss is enabled."
            ),
            "candidate_parameter_accounting": {
                "materialized_attention_weight_elements": (
                    MATERIALIZED_ATTENTION_WEIGHTS
                ),
                "persistent_dense_weight_buffer": True,
                "dense_weight_gradient": True,
                "dense_muon_momentum": True,
                "sparse_update_coordinates": UPDATE_COORDINATES,
                "sparse_coordinate_fraction_of_materialized_attention": (
                    UPDATE_COORDINATES / MATERIALIZED_ATTENTION_WEIGHTS
                ),
                "claim": (
                    "compact structured update family only; this screen does "
                    "not claim optimizer-state, training-memory, or inference "
                    "parameter compression"
                ),
            },
            "native_task_matching_required": True,
            "native_task_matching_failure_policy": (
                "fail closed; no Python or random-connectivity fallback"
            ),
            "monitoring_policy": (
                "Short 124M preflight and scientific screen are directly "
                "foreground-polled; no watchdog, callback, or heartbeat."
            ),
            "practical_equivalence_policy": (
                "require exact-config MFU >=20%, finite fixed evaluations, "
                "and terminal validation CE <=5.3924 to promote to 5TPP"
            ),
            "screen_only": True,
            "screen_only_resolution": (
                "promote only if terminal validation CE <=5.3924; otherwise "
                "reject refresh-15 sparse-orbit training without a posthoc "
                "refresh or stage sweep"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "smallest-rung causal implementation test of the "
                    "pre-registered attention refresh-cadence oracle"
                ),
                "recorded_at": "2026-08-05",
                "scope": "124M/0.5TPP attention-only replacement",
            },
        }
    )
    preserved = {
        "batch_size",
        "beta1",
        "beta2",
        "block_size",
        "data_manifest_sha256",
        "dropout",
        "eval_batch_size",
        "eval_interval",
        "eval_iters",
        "eval_protocol_id",
        "eval_seed",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_decay_iters",
        "max_iters",
        "min_lr",
        "model_seed",
        "muon_adamw_lr_scale",
        "muon_momentum",
        "muon_ns_steps",
        "n_embd",
        "n_head",
        "n_layer",
        "optimizer",
        "planned_tokens",
        "scheduled_tokens",
        "train_data_seed",
        "warmup_iters",
        "weight_decay",
    }
    if any(config[key] != source[key] for key in preserved):
        raise RuntimeError("parent scientific setting changed")
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    python = WORKSPACE / ".venv/bin/python"
    return {
        "schema_version": (
            "mai_124m_attention_refresh15_muon_matched_givens_plan_v1"
        ),
        "status": "registered_before_preflight_and_training",
        "recorded_at": "2026-08-05",
        "scientific_question": (
            "Does causally refreshing task-selected sparse attention "
            "connectivity every 15 updates turn the oracle's local direction "
            "recovery into a terminal CE improvement over the QK32 parent?"
        ),
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": EXPECTED_SOURCE_SHA256,
            "source_result": str(PARENT_RESULT.relative_to(ROOT)),
            "source_result_sha256": EXPECTED_PARENT_RESULT_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "refresh_oracle_raw_result_sha256": (
                "a7cff8d14b8445c7b8c34a4e6c4f0e059441bdf7d2987bbe36618b66624d2d0e"
            ),
            "refresh_oracle_cells_sha256": (
                "b51635ae932f6a1fe8c3ee3c44f04cc8ca7198003218765e0cb6ea4051601f6b"
            ),
            "refresh_oracle_connectivity_sha256": (
                "7b84f242932c19bfedbec404eae06f21453939b51230e34f2f1bc0b9aa145ffc"
            ),
        },
        "candidate": {
            "targets": TARGETS,
            "input_stages": {
                "attn.c_attn.qk": STAGES,
                "attn.c_attn.v": STAGES,
                "attn.c_proj": 0,
            },
            "output_stages": {target: STAGES for target in TARGETS},
            "neighbors": NEIGHBORS,
            "refresh_interval": REFRESH_INTERVAL,
            "matching_seed": MATCHING_SEED,
            "seed_step_stride": SEED_STEP_STRIDE,
            "materialized_attention_weight_elements": (
                MATERIALIZED_ATTENTION_WEIGHTS
            ),
            "sparse_update_coordinates": UPDATE_COORDINATES,
            "sparse_coordinate_fraction": (
                UPDATE_COORDINATES / MATERIALIZED_ATTENTION_WEIGHTS
            ),
            "learned_dense_basis": False,
            "learned_additive_adapter": False,
            "persistent_dense_weight_and_muon_state": True,
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "warmup_updates": 1,
            "timed_updates": 16,
            "reason": (
                "the timed window includes the expensive connectivity "
                "refresh at optimizer update 15"
            ),
            "direct_foreground_polling": True,
            "watchdog": False,
            "callbacks": False,
            "command": [
                "env",
                "CUDA_VISIBLE_DEVICES=0",
                f"CUDA_HOME={WORKSPACE / '.cuda-12.8'}",
                str(python),
                "-u",
                "-m",
                "examples.nanogpt.mfu_preflight",
                "--config",
                str(REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)),
                "--output",
                str(CERTIFICATE),
                "--log-output",
                str(OUTPUT_ROOT / "performance_preflight.log"),
                "--min-fraction",
                "0.2",
                "--warmup-updates",
                "1",
                "--timed-updates",
                "16",
            ],
        },
        "scientific_run": {
            "authorized_count": 1,
            "host": "PRO6",
            "gpu": 0,
            "max_iters": 238,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callbacks": False,
            "command": [
                "env",
                f"PYTHON_BIN={python}",
                f"MFU_PREFLIGHT_CERTIFICATE_OVERRIDE={CERTIFICATE}",
                str(
                    REMOTE_REPO
                    / "examples/nanogpt/launch_y400_ladder_detached.sh"
                ),
                "--foreground",
                str(REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)),
                "0",
                RUN_NAME,
                str(WORKSPACE),
            ],
        },
        "decision_rule": {
            "mfu_must_pass_before_training": True,
            "terminal_validation_ce_maximum": 5.3924,
            "parent_terminal_validation_ce": 5.4024,
            "historical_dense_terminal_validation_ce": 5.4890,
            "automatic_larger_rung_authorized": False,
            "no_posthoc_stage_or_refresh_sweep": True,
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_raw = json_bytes(config)
    OUTPUT_CONFIG.write_bytes(config_raw)
    OUTPUT_PLAN.write_bytes(
        json_bytes(make_plan(sha256_bytes(config_raw)))
    )
    print(OUTPUT_CONFIG)
    print(OUTPUT_PLAN)


if __name__ == "__main__":
    main()
