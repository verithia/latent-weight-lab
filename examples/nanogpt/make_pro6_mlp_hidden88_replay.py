#!/usr/bin/env python3
"""Register the PRO6 replay of the exact 124M hidden-88 MLP reference."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "c2cb932a29b2aafca58315d487937b36b8296e15"
BRANCH = "experiment/pro6-hidden88-replay-20260801"
SOURCE_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp.json"
)
OUTPUT_CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
)
OUTPUT_PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_two_pass_fresh_hidden88_pro6_replay_plan.json"
)
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
RUN_NAME = "pro6_mai_v3_124m_twopassfresh88_replay1"
OUTPUT_ROOT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay"
CERTIFICATE = OUTPUT_ROOT / "performance_preflight_hidden88_replay1.json"
REFERENCE_TERMINAL_CE = 5.592058181762695
REFERENCE_CURVE = {
    "0": 10.9669,
    "60": 6.3141,
    "120": 5.8533,
    "180": 5.6733,
    "238": REFERENCE_TERMINAL_CE,
}
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "bdbe54a8aaaba713dc76bac41d6a789c6f739cf42351786b185a425956d225ac"
)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def validate_original_identity(source: dict[str, Any]) -> None:
    if sha256_file(SOURCE_CONFIG) != EXPECTED_SOURCE_CONFIG_SHA256:
        raise RuntimeError("the immutable Y400 source config drifted")
    expected = source["implementation_source_hashes"]
    mismatches = {
        relative: {
            "expected": digest,
            "actual": sha256_file(ROOT / relative),
        }
        for relative, digest in expected.items()
        if sha256_file(ROOT / relative) != digest
    }
    if mismatches:
        raise RuntimeError(f"original execution source mismatch: {mismatches}")


def make_config(source: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(source)
    config["data_dir"] = str(WORKSPACE / "data/finewebedu_20b")
    config["out_dir"] = str(OUTPUT_ROOT / RUN_NAME)
    config["mfu_preflight_certificate"] = str(CERTIFICATE)
    config["monitoring_policy"] = (
        "PRO6 GPU0 host-local MFU qualification and the 238-update replay are "
        "foreground-polled directly; no watchdog, callback, queue worker, or "
        "heartbeat is attached to this short run"
    )
    config["replay_provenance"] = {
        "reason": (
            "Y400 filesystem availability is unknown and the exact hidden88 "
            "checkpoint is absent on PRO6; downstream MLP diagnostics require "
            "a host-local replacement reference"
        ),
        "classification": "new_replay_not_resume",
        "source_execution_commit": SOURCE_COMMIT,
        "registration_branch": BRANCH,
        "original_checkpoint_sha256": (
            "423029ea02046b95a5f8f4b9f80c000816b0ebf51f6a9dfd66dcd84da0ed47c6"
        ),
        "original_checkpoint_available": False,
        "scientific_settings_changed": False,
        "host_only_changes": [
            "data_dir",
            "out_dir",
            "mfu_preflight_certificate",
            "monitoring_policy",
        ],
    }
    allowed = {
        "data_dir",
        "out_dir",
        "mfu_preflight_certificate",
        "monitoring_policy",
        "replay_provenance",
    }
    for key in set(source) | set(config):
        if key not in allowed and source.get(key) != config.get(key):
            raise RuntimeError(f"scientific config changed at {key}")
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    repo = WORKSPACE / "latent-weight-lab-hidden88-replay"
    python = WORKSPACE / ".venv/bin/python"
    return {
        "schema_version": "mai_124m_mlp_hidden88_pro6_replay_plan_v1",
        "status": "registered_before_host_preflight_and_replay",
        "authorization": {
            "scope": "one exact-source 124M/0.5TPP hidden88 checkpoint replay",
            "source_recovery": False,
            "new_structure": False,
            "larger_rung": False,
        },
        "reason": (
            "Treat the Y400 filesystem and its checkpoint as unavailable. "
            "Generate a fresh PRO6 reference before the preregistered c_fc "
            "direction/trust-radius diagnostic."
        ),
        "identity": {
            "registration_branch": BRANCH,
            "execution_source_base_commit": SOURCE_COMMIT,
            "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
            "source_config_sha256": EXPECTED_SOURCE_CONFIG_SHA256,
            "replay_config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "replay_config_sha256": config_sha256,
            "dataset_manifest": str(WORKSPACE / "data/finewebedu_20b/manifest.json"),
            "dataset_manifest_sha256": (
                "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
            ),
            "original_checkpoint_sha256": (
                "423029ea02046b95a5f8f4b9f80c000816b0ebf51f6a9dfd66dcd84da0ed47c6"
            ),
            "original_checkpoint_available_on_pro6": False,
        },
        "invariants": {
            "scientific_hyperparameters": "byte-equivalent values to the source config",
            "implementation_source_hashes": "must match the source config exactly",
            "fixed_eval_protocol": "mai_ladder_fixed_eval_indices_v2",
            "fixed_eval_index_runtime_sha256_expected": (
                "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
            ),
            "execution_stack": "eager PyTorch/CUDA BF16 with CUDA_HOME set to the bundled toolkit",
            "checkpoint_identity": "new PRO6 replay; never label it as the Y400 checkpoint",
        },
        "mfu_gate": {
            "host": "PRO6",
            "gpu": 0,
            "minimum_fraction": 0.20,
            "warmup_updates": 1,
            "timed_updates": 8,
            "polling": "direct foreground polling; no watchdog or callback",
            "certificate": str(CERTIFICATE),
            "command": [
                "env",
                "CUDA_VISIBLE_DEVICES=0",
                f"CUDA_HOME={WORKSPACE / '.cuda-12.8'}",
                str(python),
                "-u",
                "-m",
                "examples.nanogpt.mfu_preflight",
                "--config",
                str(repo / OUTPUT_CONFIG.relative_to(ROOT)),
                "--output",
                str(CERTIFICATE),
                "--min-fraction",
                "0.2",
                "--warmup-updates",
                "1",
                "--timed-updates",
                "8",
            ],
        },
        "replay": {
            "host": "PRO6",
            "gpu": 0,
            "run_name": RUN_NAME,
            "max_iters": 238,
            "polling": "direct foreground polling through terminal evaluation",
            "watchdog": False,
            "callback": False,
            "command": [
                "env",
                f"PYTHON_BIN={python}",
                f"CUDA_HOME={WORKSPACE / '.cuda-12.8'}",
                "bash",
                str(repo / "examples/nanogpt/launch_y400_ladder_detached.sh"),
                "--foreground",
                str(repo / OUTPUT_CONFIG.relative_to(ROOT)),
                "0",
                RUN_NAME,
                str(WORKSPACE),
            ],
        },
        "reference": {
            "original_terminal_validation_ce": REFERENCE_TERMINAL_CE,
            "original_validation_curve": REFERENCE_CURVE,
        },
        "decision_rule": {
            "faithful_replay": (
                "exit zero, complete exact-resume custom state, fixed-eval digest match, "
                "absolute terminal CE delta <= 0.03, and validation-curve MAE <= 0.03"
            ),
            "usable_recovery_reference": (
                "exit zero, complete exact-resume custom state, fixed-eval digest match, "
                "finite terminal CE <= 5.65; downstream tests must calibrate against this "
                "new checkpoint and may not mix it with Y400 endpoint values"
            ),
            "reject": (
                "MFU < 0.20, nonzero exit, nonfinite loss, missing terminal evaluation, "
                "fixed-eval mismatch, incomplete custom state, or terminal CE > 5.65"
            ),
            "threshold_changes_after_result": False,
        },
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    validate_original_identity(source)
    config = make_config(source)
    config_data = json_bytes(config)
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
