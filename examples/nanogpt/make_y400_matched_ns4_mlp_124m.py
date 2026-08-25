#!/usr/bin/env python3
"""Register the gated 124M matched-NS4 dense and compact MLP tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
BASE_DENSE = CONFIGS / "pro6_mai_v3_124m_qkonly_densemlp_parent_0p5tpp_lr24e4.json"
BASE_COMPACT = CONFIGS / "pro6_mai_v3_124m_qkonly_pairvq_mlp_lazyretract8_0p5tpp_lr24e4.json"
PLAN = ARTIFACTS / "124m_pair_vq_matched_ns4_mlp_plan.json"
REMOTE_ROOT = "/root/userdata/MappingNetworks"
REMOTE_REPO = f"{REMOTE_ROOT}/latent-weight-lab"
REMOTE_OUTPUT = f"{REMOTE_ROOT}/outputs/y400_mai_v3_matched_ns4_mlp"
IMPLEMENTATION_COMMIT = "57e5447"

NATIVE_NAME = "y400_mai_v3_124m_qkonly_densemlp_native_ns5_0p5tpp_perfcontrol"
DENSE_NAME = "y400_mai_v3_124m_qkonly_densemlp_matched_ns4_0p5tpp_lr24e4"
COMPACT_NAME = "y400_mai_v3_124m_qkonly_pairvq_mlp_matched_ns4_0p5tpp_lr24e4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_command(filename: str) -> str:
    return (
        "CUDA_VISIBLE_DEVICES=0 "
        f"TORCH_EXTENSIONS_DIR={REMOTE_ROOT}/.cache/torch_extensions_sm90 "
        "TORCH_CUDA_ARCH_LIST=9.0 PYTHONPATH=. "
        f"{REMOTE_ROOT}/.venv-gpt2/bin/python -u -m examples.nanogpt.train "
        f"--config examples/nanogpt/configs/{filename}"
    )


def common(config: dict[str, object], *, name: str, filename: str) -> dict[str, object]:
    config.update(
        {
            "data_dir": f"{REMOTE_ROOT}/data/finewebedu_20b",
            "data_manifest_sha256": "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b",
            "out_dir": f"{REMOTE_OUTPUT}/{name}/scientific",
            "literal_command": literal_command(filename),
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.20,
            "mfu_preflight_min_timed_updates": 8,
            "mfu_preflight_certificate": f"{REMOTE_OUTPUT}/preflights/{name}.json",
            "monitoring_policy": (
                "foreground-poll the exact-config MFU gate; scientific training "
                "uses one idempotent terminal/error-only @Codex watchdog with no "
                "healthy progress or heartbeat callbacks"
            ),
        }
    )
    return config


def native() -> dict[str, object]:
    config = common(
        json.loads(BASE_DENSE.read_text()),
        name=NATIVE_NAME,
        filename=f"{NATIVE_NAME}.json",
    )
    config.update(
        {
            "schema_version": "mai_y400_124m_native_ns5_dense_mlp_perfcontrol_v1",
            "experiment_role": "exact-machine native NS5 dense-MLP throughput control; preflight only",
            "launch_ready": False,
            "launch_block_reason": "performance-control config is not a scientific loss run",
            "muon_mlp_ns_steps": 0,
            "muon_mlp_lr_scale": 1.0,
            "muon_mlp_polar_ridge": 0.0,
        }
    )
    return config


def dense() -> dict[str, object]:
    config = common(
        json.loads(BASE_DENSE.read_text()),
        name=DENSE_NAME,
        filename=f"{DENSE_NAME}.json",
    )
    config.update(
        {
            "schema_version": "mai_y400_124m_matched_ns4_dense_mlp_0p5tpp_v1",
            "experiment_role": "matched-optimizer dense-quality gate for four-step MLP Muon",
            "scientific_parent": str(PLAN.relative_to(ROOT)),
            "scientific_parent_sha256": sha256(PLAN),
            "launch_ready": True,
            "muon_mlp_ns_steps": 4,
            "muon_mlp_lr_scale": 1.225,
            "muon_mlp_polar_ridge": 0.0,
            "endpoint_gate": {
                "sealed_native_parent_terminal_validation_ce": 5.361,
                "terminal_validation_ce_max": 5.371,
                "all_fixed_eval_steps_finite": True,
                "same_fixed_eval_protocol_required": True,
                "minimum_mfu_fraction": 0.20,
                "throughput_ratio_vs_native_dense_min": 1.0,
                "automatic_compact_launch": False,
                "automatic_sweep": False,
                "automatic_scale_up": False,
            },
        }
    )
    return config


def compact() -> dict[str, object]:
    config = common(
        json.loads(BASE_COMPACT.read_text()),
        name=COMPACT_NAME,
        filename=f"{COMPACT_NAME}.json",
    )
    config.update(
        {
            "schema_version": "mai_y400_124m_pairvq_matched_ns4_mlp_0p5tpp_v1",
            "experiment_role": "launch-blocked compact full-MLP matched-NS4 gap test",
            "scientific_parent": str(PLAN.relative_to(ROOT)),
            "scientific_parent_sha256": sha256(PLAN),
            "launch_ready": False,
            "launch_block_reason": "requires sealed matched-NS4 dense CE and throughput pass",
            "muon_mlp_ns_steps": 4,
            "muon_mlp_lr_scale": 1.225,
            "muon_mlp_polar_ridge": 0.0,
            "endpoint_gate": {
                "matched_dense_config": f"examples/nanogpt/configs/{DENSE_NAME}.json",
                "matched_dense_result_required": True,
                "candidate_minus_matched_dense_validation_ce_max": 0.01,
                "terminal_candidate_validation_ce_max": 5.381,
                "persistent_momentum_bytes_max": 99090432,
                "persistent_raw_ambient_momentum_tensors": 0,
                "minimum_persistent_mlp_state_compression": 2.8,
                "fixed_model_compute_penalty_max": 1.1,
                "compact_checkpoint_required": True,
                "all_fixed_eval_steps_finite": True,
                "same_fixed_eval_protocol_required": True,
                "automatic_rerun": False,
                "automatic_horizon_transfer": False,
                "automatic_scale_up": False,
                "automatic_sweep": False,
            },
        }
    )
    return config


def main() -> None:
    outputs = {
        CONFIGS / f"{NATIVE_NAME}.json": native(),
        CONFIGS / f"{DENSE_NAME}.json": dense(),
        CONFIGS / f"{COMPACT_NAME}.json": compact(),
    }
    for path, payload in outputs.items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(path.relative_to(ROOT), sha256(path))


if __name__ == "__main__":
    main()
