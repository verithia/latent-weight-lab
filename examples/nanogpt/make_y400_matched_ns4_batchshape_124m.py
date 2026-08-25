#!/usr/bin/env python3
"""Register ordered systems-only batch-shape gates for matched NS4 Pair-VQ."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
SOURCE = CONFIGS / "y400_mai_v3_124m_qkonly_pairvq_mlp_matched_ns4_0p5tpp_lr24e4.json"
PLAN = ARTIFACTS / "124m_pair_vq_matched_ns4_batchshape_plan.json"
REMOTE_ROOT = "/root/userdata/MappingNetworks"
REMOTE_OUTPUT = f"{REMOTE_ROOT}/outputs/y400_mai_v3_matched_ns4_mlp/batchshape_preflights"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_for(batch_size: int, gradient_accumulation_steps: int) -> dict[str, object]:
    if batch_size * gradient_accumulation_steps != 256:
        raise ValueError("batch partition must preserve the 256-sequence global batch")
    payload = json.loads(SOURCE.read_text())
    name = (
        "y400_mai_v3_124m_qkonly_pairvq_mlp_matched_ns4_"
        f"b{batch_size}g{gradient_accumulation_steps}_perfonly"
    )
    filename = f"{name}.json"
    payload.update(
        {
            "schema_version": "mai_y400_124m_pairvq_matched_ns4_batchshape_preflight_v1",
            "experiment_role": "systems-only global-batch-preserving Pair-VQ MFU gate",
            "batchshape_plan": str(PLAN.relative_to(ROOT)),
            "batchshape_plan_sha256": sha256(PLAN),
            "batchshape_source_config": str(SOURCE.relative_to(ROOT)),
            "batchshape_source_config_sha256": sha256(SOURCE),
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "global_batch_sequences": 256,
            "tokens_per_update": 262144,
            "out_dir": f"{REMOTE_OUTPUT}/{name}/scratch",
            "mfu_preflight_certificate": f"{REMOTE_OUTPUT}/{name}.json",
            "literal_command": (
                f"CUDA_VISIBLE_DEVICES=0 TORCH_EXTENSIONS_DIR={REMOTE_ROOT}/.cache/torch_extensions_sm90 "
                "TORCH_CUDA_ARCH_LIST=9.0 PYTHONPATH=. "
                f"{REMOTE_ROOT}/.venv-gpt2/bin/python -u -m examples.nanogpt.train "
                f"--config examples/nanogpt/configs/{filename}"
            ),
            "preflight_only": True,
            "launch_ready": False,
            "launch_block_reason": (
                "systems-only preflight; a same-partition dense quality control "
                "must pass before any compact scientific run"
            ),
            "automatic_scientific_launch": False,
            "automatic_next_batchshape": False,
            "mfu_preflight_peak_memory_mib_max": 70000,
        }
    )
    return payload


def main() -> None:
    for batch_size, gradient_accumulation_steps in ((64, 4), (128, 2)):
        payload = config_for(batch_size, gradient_accumulation_steps)
        path = CONFIGS / (
            "y400_mai_v3_124m_qkonly_pairvq_mlp_matched_ns4_"
            f"b{batch_size}g{gradient_accumulation_steps}_perfonly.json"
        )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(path.relative_to(ROOT), sha256(path))


if __name__ == "__main__":
    main()
