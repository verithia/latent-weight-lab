#!/usr/bin/env python3
"""Generate the preregistered 124M attention-c_proj int8-lattice rung."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qkv_only_qk64_outputgain_5tpp_lr24e4.json"
OUTPUT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qkv_cproj_int8_lattice_0p5tpp_lr24e4.json"


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_attn_cproj_int8_lattice": True,
            "block_fht_attn_cproj_int8_lattice_block_size": 4096,
            "block_fht_attn_cproj_int8_lattice_seed": 271828,
            "candidate_scope": (
                "124M/0.5TPP causal full-attention verification: retain the "
                "accepted QK64 and V16 Cayley mappings and replace the last "
                "dense attention c_proj parameter with the preregistered "
                "block-4096 int8 running-max displacement lattice. The "
                "materialized weight, gradient, and Muon momentum remain dense."
            ),
            # Registered deterministic runs require the project-wide two-hour
            # wall-clock checkpoint cadence.  The short smallest rung still
            # writes its terminal checkpoint normally; this value primarily
            # keeps launch validation and later exact-resume identities aligned.
            "checkpoint_wall_clock_seconds": 7200,
            "confirmation_slot": "qkv_plus_cproj_int8_lattice_0p5tpp",
            "eval_interval": 60,
            "hpo_stage": "attention_cproj_int8_lattice_124m_0p5tpp",
            "implementation_commit": "6ea84c5283feb0bac5a6650bf5e5d06d04d53d90",
            "implementation_source_hashes": {
                "examples/nanogpt/mfu_preflight.py": "eac4e899a8be6fb6857940e5070f759becd3a97b59af82bf9b1a2c79fcdeee40",
                "examples/nanogpt/model.py": "3fbc923f0fb992a1fab4179af67f102c017368d7518b974a9e8e1963ad08d5d9",
                "examples/nanogpt/muon.py": "4702dbee85408ab43112acf8c11c9f3e09fecdaf46345c1012a765c516ef44a1",
                "examples/nanogpt/muon_int8_lattice.py": "0bda5ed83a4392507c42c9b09b9bb7e09ca033b2b2236cd3263fb2942f4b70e6",
                "examples/nanogpt/test_muon_int8_lattice.py": "e5bd91118e84d6363cae6ec1a1a3140a456a51865c99fb496509c6080d6c99fa",
                "examples/nanogpt/train.py": "4209f92d8eb7a7d0fdb4fed9182aa49bc8fe8c97acd4b40d34d2574907c290eb",
            },
            "int8_lattice_representation": {
                "base": "reproducible frozen Gaussian initialization",
                "blocks": 1728,
                "code_bytes": 7077888,
                "elements": 7077888,
                "fp16_scale_bytes": 3456,
                "fp32_weight_bytes": 28311552,
                "optimizer_momentum": "dense_fp32_not_in_codec_count",
                "persistent_codec_bytes": 7081344,
                "runtime_base": "transient_dense_fp32",
                "runtime_weight": "transient_dense_fp32",
                "storage_ratio": 0.2501220703125,
                "storage_reduction": 3.998047828208882,
            },
            "ladder_role": "attention_cproj_int8_lattice_smallest_rung",
            "ladder_slot": "qkv_plus_cproj_int8_lattice",
            "launch_block_reason": None,
            "launch_ready": True,
            "lr_decay_iters": 238,
            "max_iters": 238,
            "mfu_measurement_protocol": (
                "foreground exact-config real-training preflight with one "
                "warmup and eight timed updates; includes per-update dense "
                "Muon request, int8 projection, and dense rematerialization"
            ),
            "mfu_min_fraction": 0.2,
            "mfu_preflight_required": True,
            "monitoring_policy": (
                "foreground-poll the <=5 minute MFU gate; after a pass, use "
                "one idempotent terminal/error-only @Codex watchdog for the "
                "238-update run because it may exceed five minutes"
            ),
            "out_dir": (
                "/home/pro6000-9980x/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_qkv_cproj_int8_lattice_0p5tpp_lr24e4/scientific"
            ),
            "planned_tokens": 62186880,
            "planned_tpp": 0.5,
            "practical_equivalence_policy": (
                "This is a causal smallest-rung verification, not a claim of "
                "long-horizon closure. Require finite terminal CE <=5.5890 "
                "before one separately preregistered 5TPP transfer."
            ),
            "registered_resume_protocol": (
                "atomic latest checkpoint with full RNG state, int8 codes, "
                "FP16 running-max scales, reproducible base identity, dense "
                "Muon momentum, and no serialized dense c_proj weight"
            ),
            "scheduled_tokens": 62390272,
            "scheduled_tpp": 0.5016353288667963,
            "screen_only": False,
            "screen_only_resolution": None,
            "selection_endpoint": (
                "terminal step-238 fixed-window validation CE versus dense "
                "124M/0.5TPP CE 5.4890"
            ),
            "warmup_iters": 10,
        }
    )
    for obsolete in (
        "dense_fixed_validation_curve",
        "parent_fixed_validation_curve",
        "parent_dense_token_equivalent_penalty",
        "operator_override",
    ):
        config.pop(obsolete, None)
    return config


def main() -> None:
    OUTPUT.write_text(json.dumps(build_config(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
