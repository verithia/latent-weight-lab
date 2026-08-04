#!/usr/bin/env python3
"""Register the matched BlockFHT-attention parent c_proj trajectory on PRO6."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "y400_mai_v2_124m_fullattn_blockfht_0p5tpp_mult1p00.json"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_parent_cproj_stepwise_trajectory_0p5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_matched_fullattn_parent_trajectory_plan.json"
)
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "a34024398271ca0d40e41b3b136ed0e471bce22ce796b402e46608624793fac7"
)
CAUSAL_RESULT_SHA256 = (
    "c2094f6098ca4ea3b2bb9a85f2e9493868cb9e288e9f73ce4fdc58aba339286d"
)
CORRECTED_DENSE_TEACHER_ORACLE_SHA256 = (
    "f91f831734f6c171bd3ca62668a46f8052cd4b11b269945dc1f38d92645724b0"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
REMOTE_REPO = Path("/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab")
RUN_NAME = "pro6_mai_v3_124m_fullattn_parent_cproj_stepwise_trajectory_0p5tpp"
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_manifold" / RUN_NAME
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
SNAPSHOT_LAYERS = [0, 3, 6, 9, 11]


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
    if sha256_file(SOURCE_CONFIG) != EXPECTED_SOURCE_CONFIG_SHA256:
        raise RuntimeError("immutable BlockFHT-attention source config drifted")
    config = copy.deepcopy(source)
    config.update(
        {
            "data_dir": str(WORKSPACE / "data/finewebedu_20b"),
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "trajectory_snapshot_interval": 1,
            "trajectory_snapshot_targets": ["mlp.c_proj"],
            "trajectory_snapshot_layers": SNAPSHOT_LAYERS,
            "trajectory_snapshot_dtype": "float32",
            "diagnostic_protocol": (
                "Run the exact original BlockFHT-attention/dense-MLP parent and "
                "record dense mlp.c_proj at step 0 and after every optimizer update "
                "for layers 0,3,6,9,11. The snapshots are diagnostic side effects; "
                "optimizer, model, data, schedule, and fixed evaluation are unchanged."
            ),
            "diagnostic_caveat": (
                "This is a matched-parent trajectory oracle, not a global manifold "
                "dimension estimate and not a causal compressed-c_proj result."
            ),
            "estimated_trajectory_payload_bytes": 11277434880,
            "matched_parent_replay_provenance": {
                "classification": "new_cross_host_replay_not_resume",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
                "original_terminal_validation_ce": 5.4918,
                "original_step60_validation_ce": 6.2184,
                "causal_bilateral_result_sha256": CAUSAL_RESULT_SHA256,
                "corrected_dense_teacher_oracle_sha256": (
                    CORRECTED_DENSE_TEACHER_ORACLE_SHA256
                ),
                "scientific_settings_changed": False,
                "allowed_changes": [
                    "PRO6 data/output/certificate paths",
                    "per-update c_proj trajectory diagnostics",
                    "diagnostic and replay provenance fields",
                ],
            },
            "monitoring_policy": (
                "Short 124M diagnostic: foreground polling only for preflight and "
                "scientific run; no watchdog, callback, queue worker, or heartbeat."
            ),
        }
    )
    allowed = {
        "data_dir",
        "out_dir",
        "mfu_preflight_certificate",
        "trajectory_snapshot_interval",
        "trajectory_snapshot_targets",
        "trajectory_snapshot_layers",
        "trajectory_snapshot_dtype",
        "diagnostic_protocol",
        "diagnostic_caveat",
        "estimated_trajectory_payload_bytes",
        "matched_parent_replay_provenance",
        "monitoring_policy",
    }
    changed = {
        key
        for key in set(source) | set(config)
        if source.get(key) != config.get(key)
    }
    unexpected = changed - allowed
    if unexpected:
        raise RuntimeError(f"scientific config changed unexpectedly: {unexpected}")
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    python = WORKSPACE / ".venv/bin/python"
    trajectory_dir = SCIENTIFIC_OUT / "parameter_trajectory"
    checkpoint = SCIENTIFIC_OUT / "ckpt.pt"
    oracle_out = OUTPUT_ROOT / "matched_parent_endpoint_oracle"
    return {
        "schema_version": "mai_124m_mlp_cproj_matched_fullattn_parent_trajectory_plan_v1",
        "status": "registered_before_performance_preflight_or_execution",
        "recorded_at": "2026-08-04",
        "scientific_question": (
            "Does the bilateral c_proj endpoint gain survive when the teacher "
            "trajectory and evaluated checkpoint use the same original BlockFHT "
            "attention operator as the failed causal c_proj candidate?"
        ),
        "confound": (
            "The accepted output32 oracle replayed a fully dense-attention teacher, "
            "but the causal candidate used the original BlockFHT attention-only "
            "parent from initialization onward. The optimizer-selector correction "
            "did not remove this parent-model mismatch."
        ),
        "authorization": {
            "trajectory": "one exact 238-update matched-parent replay after MFU pass",
            "endpoint_oracle": "one zero-update right/output32/output64 fixed-eval replay after trajectory acceptance",
            "language_model_candidate_training": False,
            "watchdog": False,
            "callbacks": False,
        },
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "causal_bilateral_result_sha256": CAUSAL_RESULT_SHA256,
            "corrected_dense_teacher_oracle_sha256": (
                CORRECTED_DENSE_TEACHER_ORACLE_SHA256
            ),
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "execution_commit_rule": (
                "record the exact clean pushed commit containing this immutable plan"
            ),
        },
        "matched_parent": {
            "method": "block_fht",
            "block_fht_targets": [
                "attn.c_attn.qk_headwise",
                "attn.c_attn.v",
                "attn.c_proj",
            ],
            "mlp_c_fc": "dense",
            "mlp_c_proj": "dense teacher trajectory",
            "original_terminal_validation_ce": 5.4918,
            "original_step60_validation_ce": 6.2184,
            "snapshot_steps": "0 through 238 inclusive",
            "snapshot_layers": SNAPSHOT_LAYERS,
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "include_diagnostic_io": True,
            "warmup_updates": 1,
            "timed_updates": 8,
            "direct_foreground_polling": True,
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
                "8",
                "--include-diagnostic-io",
            ],
        },
        "trajectory_execution": {
            "host": "PRO6",
            "gpu": 0,
            "out_dir": str(SCIENTIFIC_OUT),
            "direct_foreground_polling": True,
            "command": [
                "env",
                "CUDA_VISIBLE_DEVICES=0",
                f"CUDA_HOME={WORKSPACE / '.cuda-12.8'}",
                "PYTHONPATH=.",
                str(python),
                "-u",
                "-m",
                "examples.nanogpt.train",
                "--config",
                str(REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)),
            ],
        },
        "trajectory_acceptance": {
            "required": [
                "MFU >= 0.20 with snapshot I/O charged",
                "clean terminal step 238 and complete exact-resume checkpoint",
                "239 snapshots for layers 0,3,6,9,11",
                "dataset and fixed-evaluation digests match",
                "all losses and floating optimizer/model state finite",
                "absolute terminal validation CE delta from 5.4918 <= 0.03",
                "absolute step-60 validation CE delta from 6.2184 <= 0.03",
            ],
            "threshold_changes_after_measurement": False,
        },
        "endpoint_oracle": {
            "trajectory_dir": str(trajectory_dir),
            "checkpoint": str(checkpoint),
            "output_dir": str(oracle_out),
            "connectivity_target": "production_nondecay",
            "variants": [
                "matched_parent_endpoint",
                "hidden88_full_carry",
                "hidden88_output32_full_carry",
                "hidden88_output64_full_carry",
            ],
            "candidate_pass": (
                "finite candidate validation CE at least 0.002 below right-only, "
                "candidate train CE no worse, and candidate validation distance "
                "from matched parent smaller than right-only"
            ),
            "selection_order": [
                "hidden88_output32_full_carry",
                "hidden88_output64_full_carry",
            ],
        },
        "decision_rule": {
            "neither_candidate_passes": (
                "classify the old bilateral direction as dense-attention-specific; "
                "do not implement an activation metric around that direction"
            ),
            "candidate_passes_but_known_causal_run_fails": (
                "classify true matched-parent closed-loop direction drift; then "
                "preregister an activation-weighted/current-task selector"
            ),
            "new_training_before_classification": False,
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config_data = json_bytes(make_config(source))
    plan_data = json_bytes(make_plan(sha256_bytes(config_data)))
    OUTPUT_CONFIG.write_bytes(config_data)
    OUTPUT_PLAN.write_bytes(plan_data)
    print(
        json.dumps(
            {
                "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
                "config_sha256": sha256_bytes(config_data),
                "plan": str(OUTPUT_PLAN.relative_to(ROOT)),
                "plan_sha256": sha256_bytes(plan_data),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
