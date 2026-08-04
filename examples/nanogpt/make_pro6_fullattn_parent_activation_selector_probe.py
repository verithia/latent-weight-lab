#!/usr/bin/env python3
"""Register matched-parent phase states for an activation-weighted c_proj selector."""

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
    "pro6_mai_v3_124m_fullattn_parent_activation_selector_probe_0p5tpp.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_activation_weighted_output_selector_plan.json"
)
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "a34024398271ca0d40e41b3b136ed0e471bce22ce796b402e46608624793fac7"
)
MATCHED_PARENT_ENDPOINT_RESULT_SHA256 = (
    "7041249de559c82df785ec439308e5849f6c38b9d52eb885afdea9b0bf2b3760"
)
MATCHED_PARENT_TRAJECTORY_RESULT_SHA256 = (
    "f261dc43f44ae73bbc01ae32cdae317897370b4382564f0994d0178711f724c0"
)
CAUSAL_OUTPUT32_RESULT_SHA256 = (
    "c2094f6098ca4ea3b2bb9a85f2e9493868cb9e288e9f73ce4fdc58aba339286d"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
REMOTE_REPO = Path("/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab")
RUN_NAME = "pro6_mai_v3_124m_fullattn_parent_activation_selector_probe_0p5tpp"
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_manifold" / RUN_NAME
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
PHASE_BOUNDARIES = [0, 60, 120, 180, 238]
PROBE_STEPS = PHASE_BOUNDARIES[:-1]
PROBE_LAYERS = [0, 3, 6, 9, 11]


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
            "trajectory_snapshot_interval": 60,
            "trajectory_snapshot_targets": [],
            "trajectory_snapshot_layers": None,
            "trajectory_snapshot_all_parameters": True,
            "trajectory_snapshot_dtype": "float32",
            "optimizer_probe_steps": PROBE_STEPS,
            "optimizer_probe_targets": ["mlp.c_proj"],
            "optimizer_probe_layers": PROBE_LAYERS,
            "optimizer_probe_dtype": "float32",
            "diagnostic_protocol": (
                "Replay the exact original BlockFHT-attention/dense-MLP parent; "
                "save every named parameter at steps 0,60,120,180,238 and the "
                "exact pre-step dense c_proj Muon state/direction at steps "
                "0,60,120,180 for layers 0,3,6,9,11. These are diagnostic side "
                "effects only; optimizer, model, data, schedule, and evaluation "
                "remain unchanged."
            ),
            "diagnostic_caveat": (
                "This acquisition supports a zero-update current-phase selector "
                "comparison. It is neither a global manifold-dimension estimate "
                "nor authorization for compressed-c_proj language-model training."
            ),
            "estimated_trajectory_payload_bytes": 2487475200,
            "estimated_optimizer_probe_payload_bytes": 1132462080,
            "activation_selector_probe_provenance": {
                "classification": "new_cross_host_replay_not_resume",
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "source_config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
                "matched_parent_endpoint_result_sha256": (
                    MATCHED_PARENT_ENDPOINT_RESULT_SHA256
                ),
                "matched_parent_trajectory_result_sha256": (
                    MATCHED_PARENT_TRAJECTORY_RESULT_SHA256
                ),
                "causal_output32_result_sha256": CAUSAL_OUTPUT32_RESULT_SHA256,
                "original_terminal_validation_ce": 5.4918,
                "original_step60_validation_ce": 6.2184,
                "scientific_settings_changed": False,
                "allowed_changes": [
                    "PRO6 data/output/certificate paths",
                    "sparse all-parameter phase snapshots",
                    "phase-start c_proj Muon optimizer probes",
                    "diagnostic and replay provenance fields",
                ],
            },
            "monitoring_policy": (
                "Short 124M diagnostic: direct foreground polling for preflight "
                "and scientific acquisition; no watchdog, callback, queue worker, "
                "or heartbeat."
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
        "trajectory_snapshot_all_parameters",
        "trajectory_snapshot_dtype",
        "optimizer_probe_steps",
        "optimizer_probe_targets",
        "optimizer_probe_layers",
        "optimizer_probe_dtype",
        "diagnostic_protocol",
        "diagnostic_caveat",
        "estimated_trajectory_payload_bytes",
        "estimated_optimizer_probe_payload_bytes",
        "activation_selector_probe_provenance",
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
    snapshot_dir = SCIENTIFIC_OUT / "parameter_trajectory"
    probe_dir = SCIENTIFIC_OUT / "optimizer_probes"
    analysis_out = OUTPUT_ROOT / "activation_weighted_output_selector"
    return {
        "schema_version": "mai_124m_mlp_cproj_activation_weighted_output_selector_plan_v1",
        "status": "registered_before_acquisition_preflight_or_selector_implementation",
        "recorded_at": "2026-08-04",
        "scientific_question": (
            "At identical output32 chart size, does selecting c_proj output pairs "
            "and angles in the empirical post-GELU residual-stream metric beat "
            "the current Frobenius weight-space selector on unseen tokens and CE?"
        ),
        "causal_diagnosis": (
            "The matched BlockFHT-attention endpoint oracle retained 92.35% of "
            "the dense-attention output32 gain, while the causal output32 run "
            "worsened its right-only parent. Parent identity, output capacity, "
            "and the earlier weight-decay selector mismatch are therefore ruled "
            "out; closed-loop task-direction selection remains the live cause."
        ),
        "authorization": {
            "acquisition": "one exact 238-update matched-parent replay after MFU pass",
            "zero_update_selector_analysis": True,
            "selector_implementation_before_acquisition_acceptance": False,
            "language_model_candidate_training": False,
            "watchdog": False,
            "callbacks": False,
        },
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "matched_parent_endpoint_result_sha256": (
                MATCHED_PARENT_ENDPOINT_RESULT_SHA256
            ),
            "matched_parent_trajectory_result_sha256": (
                MATCHED_PARENT_TRAJECTORY_RESULT_SHA256
            ),
            "causal_output32_result_sha256": CAUSAL_OUTPUT32_RESULT_SHA256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
            "execution_commit_rule": (
                "record a clean pushed commit containing this immutable plan"
            ),
        },
        "acquisition": {
            "host": "PRO6",
            "gpu": 0,
            "phase_boundaries": PHASE_BOUNDARIES,
            "optimizer_probe_steps": PROBE_STEPS,
            "optimizer_probe_layers": PROBE_LAYERS,
            "snapshot_scope": "all named model parameters in float32",
            "optimizer_probe_scope": (
                "weight, clipped gradient, old momentum, combined momentum, "
                "polar update, and exact applied direction per LR for mlp.c_proj"
            ),
            "snapshot_dir": str(snapshot_dir),
            "probe_dir": str(probe_dir),
            "out_dir": str(SCIENTIFIC_OUT),
            "direct_foreground_polling": True,
            "command": [
                str(REMOTE_REPO / "examples/nanogpt/launch_y400_ladder_detached.sh"),
                "--foreground",
                str(REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)),
                "0",
                RUN_NAME,
                str(WORKSPACE),
            ],
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
        "acquisition_acceptance": {
            "required": [
                "MFU >= 0.20 with all-parameter snapshot and optimizer-probe I/O charged",
                "clean terminal step 238 and complete exact-resume checkpoint",
                "all-parameter snapshots exactly at 0,60,120,180,238",
                "c_proj optimizer probes exactly at 0,60,120,180 for layers 0,3,6,9,11",
                "one common run identity across snapshots, probes, and checkpoint",
                "dataset and fixed-evaluation digests match",
                "all losses and floating model/optimizer/probe tensors finite",
                "absolute terminal validation CE delta from 5.4918 <= 0.03",
                "absolute step-60 validation CE delta from 6.2184 <= 0.03",
            ],
            "threshold_changes_after_measurement": False,
        },
        "selector_analysis": {
            "output_dir": str(analysis_out),
            "parameter_updates": 0,
            "phases": [[0, 60], [60, 120], [120, 180], [180, 238]],
            "layers": PROBE_LAYERS,
            "fit_window": {
                "split": "validation",
                "seed": 20260804,
                "batch_size": 2,
                "block_size": 256,
                "batches": 4,
                "rows_per_layer": 2048,
            },
            "holdout_window": {
                "split": "validation",
                "seed": 20260805,
                "batch_size": 2,
                "block_size": 256,
                "batches": 4,
                "rows_per_layer": 2048,
            },
            "shared_chart": {
                "hidden_parent_stages": 64,
                "hidden_residual_stages": 24,
                "output_stages": 32,
                "neighbors": 64,
                "matching_seed": 20260804,
                "coordinate_count_per_layer": 147456,
                "feedback": "zero for this one-step prospective diagnostic",
                "weight_decay_application": "identical production ordering in both arms",
            },
            "control": (
                "Frobenius output32: choose output pairs and closed-form angles "
                "from output_source=W_after_hidden^T and output residual^T"
            ),
            "candidate": (
                "Activation output32: on fit tokens form H@output_source and "
                "H@output_residual, choose the same number of output-channel "
                "pairs and fit angles there, then apply those discrete pairs and "
                "angles to the unprojected output_source"
            ),
            "prohibited": [
                "learned basis",
                "inverse JtJ or conjugate-gradient pullback",
                "dense residual",
                "extra chart coordinates",
                "selection on holdout activations",
            ],
            "measurements": [
                "fit and holdout ||H(DeltaW_target-DeltaW_chart)^T||_F^2",
                "fit and holdout fixed-scale and positive-line output recovery",
                "recorded-train and fit/holdout task-gradient predicted CE descent",
                "simultaneous five-layer finite-step CE on fit and holdout batches",
                "weight-space recovery and chart timing as secondary diagnostics",
            ],
        },
        "decision_rule": {
            "pass_all": [
                "all outputs and metrics finite",
                "aggregate holdout activation-output residual energy <= 0.95 times Frobenius control",
                "fit activation residual-energy advantage = 1-candidate/control is positive",
                "holdout-to-fit activation residual-energy advantage ratio >= 0.80",
                "aggregate holdout task-gradient predicted CE descent >= Frobenius control",
                "activation selector finite-step CE beats Frobenius in at least 6 of 8 phase-window comparisons",
                "mean finite-step CE across all 8 comparisons <= Frobenius control",
            ],
            "pass_action": (
                "preregister a production activation-weighted output32 path, add "
                "focused correctness tests, and require a separate >=20% MFU "
                "preflight before considering exactly one 124M causal run"
            ),
            "fail_action": (
                "reject activation-weighted pair selection without training and "
                "move to a directly task-gradient-aware output selector"
            ),
            "language_model_training_authorized_by_this_plan": False,
            "advantage_formula": (
                "advantage(window)=1-E_activation(window)/E_frobenius(window); "
                "retention=advantage(holdout)/advantage(fit)"
            ),
            "threshold_changes_after_measurement": False,
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
