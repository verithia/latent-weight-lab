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
            "checkpoint_wall_clock_seconds": 1800,
            "confirmation_slot": "qkv_plus_cproj_int8_lattice_0p5tpp",
            "eval_interval": 60,
            "hpo_stage": "attention_cproj_int8_lattice_124m_0p5tpp",
            "implementation_commit": "c6061b273bec2e59413be8b8c1a8b6e30e02dde4",
            "implementation_source_hashes": {
                "examples/nanogpt/mfu_preflight.py": "4f244e23be072602ef959694095a743306b35d1d2dcb27b7462fdbc002a28303",
                "examples/nanogpt/model.py": "3fbc923f0fb992a1fab4179af67f102c017368d7518b974a9e8e1963ad08d5d9",
                "examples/nanogpt/muon.py": "4702dbee85408ab43112acf8c11c9f3e09fecdaf46345c1012a765c516ef44a1",
                "examples/nanogpt/muon_int8_lattice.py": "647d7a56dbe15be53348c9f1c7b3b480b747468d6cce7e325d187c287b764249",
                "examples/nanogpt/test_muon_int8_lattice.py": "fc6269e3ddc07776edc40c1b92217e2566fe8948ea22879795839d99fe2b33ea",
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
