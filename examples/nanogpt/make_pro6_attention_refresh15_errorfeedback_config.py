#!/usr/bin/env python3
"""Register the decay-0.5 temporal repair for refresh-15 attention."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_refresh15_muon_matched_givens_"
    "0p5tpp_lr24e4.json"
)
SOURCE_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_refresh15_muon_matched_givens_result.json"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_refresh15_muon_matched_givens_"
    "errorfeedback_decay0p5_0p5tpp_lr24e4.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_refresh15_errorfeedback_decay0p5_plan.json"
)
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "d1c3bf1c7927083dedd5089f2f97bbddebf024b5ca9cac3cc37490f77b800faa"
)
EXPECTED_SOURCE_RESULT_SHA256 = (
    "2e82f111d2fb747ac17b39708cc45fc41ff08fa0d7e9250a5c6abde1e4d7ad2f"
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
    "pro6_mai_v3_124m_fullattn_refresh15_muon_matched_givens_"
    "errorfeedback_decay0p5_0p5tpp"
)
OUTPUT_ROOT = (
    Path("/mnt/ssd-data/orj/MappingNetworks/outputs")
    / "pro6_mai_v3_attention_refresh15_errorfeedback"
    / RUN_NAME
)
SCIENTIFIC_OUT = OUTPUT_ROOT / "scientific"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight.json"
ATTENTION_WEIGHT_ELEMENTS = 28_311_552
FP32_FEEDBACK_BYTES = ATTENTION_WEIGHT_ELEMENTS * 4


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
        raise RuntimeError("immutable refresh-15 parent config drifted")
    if sha256_file(SOURCE_RESULT) != EXPECTED_SOURCE_RESULT_SHA256:
        raise RuntimeError("immutable refresh-15 parent result drifted")
    config = copy.deepcopy(source)
    config.update(
        {
            "block_fht_attn_muon_matched_givens_error_feedback": True,
            "block_fht_attn_muon_matched_givens_error_feedback_decay": 0.5,
            "block_fht_attn_muon_matched_givens_error_feedback_max_nominal_steps": None,
            "out_dir": str(SCIENTIFIC_OUT),
            "mfu_preflight_certificate": str(CERTIFICATE),
            "hpo_stage": (
                "attention_refresh15_errorfeedback_decay0p5_124m_0p5tpp"
            ),
            "ladder_slot": "refresh15_errorfeedback_decay0p5",
            "confirmation_slot": "refresh15_errorfeedback_decay0p5",
            "confirmation_source": (
                "memoryless refresh-15 retained 23.29% mean instantaneous "
                "recovery but its discarded residual grew from 0.353 to "
                "0.620 nominal Muon steps; MLP c_proj evidence selected "
                "bounded decay 0.5 over unstable full accumulation"
            ),
            "candidate_scope": (
                "Keep the exact refresh-15 sparse attention chart and add "
                "standard temporal error feedback with fixed decay 0.5. "
                "No basis, adapter, gain, Mapping Loss, stage change, "
                "refresh change, or decay sweep is enabled."
            ),
            "candidate_error_feedback_accounting": {
                "attention_weight_elements": ATTENTION_WEIGHT_ELEMENTS,
                "fp32_feedback_state_bytes": FP32_FEEDBACK_BYTES,
                "fp32_feedback_state_mib": FP32_FEEDBACK_BYTES / 2**20,
                "persistent_dense_feedback_state": True,
                "inference_effect": "none after weights are folded",
                "claim": (
                    "temporal compression-bias repair only; this adds dense "
                    "training state and is not optimizer-memory compression"
                ),
            },
            "monitoring_policy": (
                "Short 124M preflight and scientific screen are directly "
                "foreground-polled; no watchdog, callback, or heartbeat."
            ),
            "practical_equivalence_policy": (
                "require exact-config MFU >=20%, finite fixed evaluations, "
                "and terminal validation CE <=5.3924 to promote to 5TPP"
            ),
            "screen_only_resolution": (
                "promote only if terminal validation CE <=5.3924; otherwise "
                "reject this temporal repair without a posthoc decay, stage, "
                "or refresh sweep"
            ),
        }
    )
    changed = {
        "block_fht_attn_muon_matched_givens_error_feedback",
        "block_fht_attn_muon_matched_givens_error_feedback_decay",
        "block_fht_attn_muon_matched_givens_error_feedback_max_nominal_steps",
        "out_dir",
        "mfu_preflight_certificate",
        "hpo_stage",
        "ladder_slot",
        "confirmation_slot",
        "confirmation_source",
        "candidate_scope",
        "candidate_error_feedback_accounting",
        "monitoring_policy",
        "practical_equivalence_policy",
        "screen_only_resolution",
    }
    for key, value in source.items():
        if key not in changed and config[key] != value:
            raise RuntimeError(f"parent setting changed: {key}")
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    python = WORKSPACE / ".venv/bin/python"
    config_path = REMOTE_REPO / OUTPUT_CONFIG.relative_to(ROOT)
    return {
        "schema_version": "mai_124m_attention_refresh15_errorfeedback_plan_v1",
        "status": "registered_before_preflight_and_training",
        "recorded_at": "2026-08-05",
        "scientific_question": (
            "Does decay-0.5 temporal error feedback prevent discarded sparse "
            "attention updates from accumulating into the observed late-run "
            "loss regression?"
        ),
        "identity": {
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
            "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "source_result_sha256": EXPECTED_SOURCE_RESULT_SHA256,
            "candidate_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "candidate_config_sha256": config_sha256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
        },
        "isolated_change": {
            "error_feedback": True,
            "error_feedback_decay": 0.5,
            "error_feedback_max_nominal_steps": None,
            "feedback_state_bytes": FP32_FEEDBACK_BYTES,
            "learned_dense_basis": False,
            "learned_additive_adapter": False,
        },
        "performance_gate": {
            "minimum_mfu_fraction": 0.2,
            "warmup_updates": 1,
            "timed_updates": 16,
            "reason": (
                "the timed window activates error feedback and includes the "
                "connectivity refresh at optimizer update 15"
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
                str(config_path),
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
                str(config_path),
                "0",
                RUN_NAME,
                str(WORKSPACE),
            ],
        },
        "decision_rule": {
            "mfu_must_pass_before_training": True,
            "terminal_validation_ce_maximum": 5.3924,
            "parent_terminal_validation_ce": 5.4024,
            "memoryless_refresh15_terminal_validation_ce": 5.5884,
            "automatic_larger_rung_authorized": False,
            "no_posthoc_decay_stage_or_refresh_sweep": True,
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config = make_config(source)
    config_raw = json_bytes(config)
    OUTPUT_CONFIG.write_bytes(config_raw)
    OUTPUT_PLAN.write_bytes(json_bytes(make_plan(sha256_bytes(config_raw))))
    print(OUTPUT_CONFIG)
    print(OUTPUT_PLAN)


if __name__ == "__main__":
    main()
