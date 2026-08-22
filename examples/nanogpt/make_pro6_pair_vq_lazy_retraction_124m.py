#!/usr/bin/env python3
"""Register the sole 124M short-horizon lazy Pair-VQ retraction endpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qkonly_pairvq_mlp_fp16_ambient_momentum_0p5tpp_lr24e4.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_pair_vq_lazy_retraction_plan.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qkonly_pairvq_mlp_lazyretract8_0p5tpp_lr24e4.json"
)
REMOTE_ROOT = "/mnt/ssd-data/orj/MappingNetworks"
RUN_NAME = "pro6_mai_v3_124m_qkonly_pairvq_mlp_lazyretract8_0p5tpp_lr24e4"
REMOTE_OUTPUT = f"{REMOTE_ROOT}/outputs/pro6_mai_v3_pair_vq/{RUN_NAME}"
IMPLEMENTATION_COMMIT = "e4ff6173c"
ADAPTIVE_MOMENTUM_BYTES_MAX = 99_090_432


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "schema_version": "mai_qkonly_pairvq_mlp_lazyretract8_0p5tpp_v1",
            "experiment_role": (
                "sole seed-1 124M short-horizon non-regression endpoint for "
                "eight-step ambient Muon carry with compact-boundary Pair-VQ retraction"
            ),
            "scientific_parent": str(PLAN.relative_to(ROOT)),
            "scientific_parent_sha256": sha256(PLAN),
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "entrypoint": "examples/nanogpt/train.py",
            "literal_command": (
                "CUDA_VISIBLE_DEVICES=0 "
                f"LD_LIBRARY_PATH={REMOTE_ROOT}/.compat/nvidia-595.71.05/root/usr/lib/x86_64-linux-gnu "
                f"TORCH_EXTENSIONS_DIR={REMOTE_ROOT}/.cache/torch_extensions_sm120_exactfamily "
                "TORCH_CUDA_ARCH_LIST=12.0 PYTHONPATH=. "
                f"{REMOTE_ROOT}/.venv/bin/python -u examples/nanogpt/train.py "
                f"--config examples/nanogpt/configs/{OUTPUT.name}"
            ),
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.20,
            "mfu_preflight_min_timed_updates": 8,
            "mfu_preflight_certificate": (
                f"{REMOTE_ROOT}/outputs/pro6_mai_v3_pair_vq/preflights/"
                f"{RUN_NAME}_mfu.json"
            ),
            "persistent_momentum_bytes_max": ADAPTIVE_MOMENTUM_BYTES_MAX,
            "out_dir": f"{REMOTE_OUTPUT}/scientific",
            "block_fht_mlp_pair_vq_fp16_reserved_escape_granularity": (
                "adaptive_block"
            ),
            "block_fht_mlp_pair_vq_lazy_retraction_interval": 8,
            "block_fht_mlp_pair_vq_lazy_retraction_forced_steps": [
                60,
                120,
                180,
                238,
            ],
            "monitoring_policy": (
                "foreground-poll the <=5 minute exact-config MFU gate; after a "
                "pass, use one idempotent terminal/error-only @Codex watchdog "
                "with no milestones or heartbeat"
            ),
            "endpoint_gate": {
                "matched_parent_config": (
                    "examples/nanogpt/configs/"
                    "pro6_mai_v3_124m_qkonly_densemlp_parent_0p5tpp_lr24e4.json"
                ),
                "dense_parent_terminal_validation_ce": 5.3610,
                "terminal_candidate_validation_ce_max": 5.4110,
                "candidate_minus_parent_terminal_validation_ce_max": 0.05,
                "persistent_momentum_bytes_max": ADAPTIVE_MOMENTUM_BYTES_MAX,
                "persistent_raw_ambient_momentum_tensors": 0,
                "lazy_retraction_interval": 8,
                "forced_compact_boundary_steps": [60, 120, 180, 238],
                "compact_checkpoint_required": True,
                "all_fixed_eval_steps_finite": True,
                "same_fixed_eval_protocol_required": True,
                "selection_endpoint": True,
                "automatic_rerun": False,
                "automatic_horizon_transfer": False,
                "automatic_scale_up": False,
                "automatic_sweep": False,
            },
        }
    )
    # The adaptive momentum container has a hard byte ceiling rather than one
    # exact size, so the inherited raw-FP16 equality assertion is inapplicable.
    config.pop("persistent_training_bytes_exact", None)
    return config


def main() -> None:
    OUTPUT.write_text(json.dumps(build_config(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT.relative_to(ROOT))
    print(sha256(OUTPUT))


if __name__ == "__main__":
    main()
