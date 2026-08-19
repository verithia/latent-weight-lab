#!/usr/bin/env python3
"""Generate the preregistered 124M full-MLP ambient int8-lattice rung."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qkv_cproj_int8_lattice_0p5tpp_lr24e4.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_0p5tpp.json"
)


def build_config() -> dict[str, object]:
    config = json.loads(BASE.read_text())
    config.update(
        {
            "block_fht_mlp_int8_lattice_targets": [
                "mlp.c_fc",
                "mlp.c_proj",
            ],
            "block_fht_mlp_int8_lattice_block_size": 4096,
            "block_fht_mlp_int8_lattice_seed": 314159,
            "candidate_scope": (
                "124M/0.5TPP full-replacement spatial-direction test: keep "
                "the sealed QK64, V16, and attention-c_proj int8-lattice "
                "parent, and replace both dense MLP weights by independent "
                "causal block-4096 signed-int8 running-max displacement "
                "lattices. Dense materialized weights, gradients, and FP32 "
                "Muon momentum remain runtime state."
            ),
            "confirmation_slot": "full_mlp_int8_lattice_0p5tpp",
            "hpo_stage": "full_mlp_int8_lattice_124m_0p5tpp",
            "implementation_source_paths": [
                "examples/nanogpt/model.py",
                "examples/nanogpt/muon_int8_lattice.py",
                "examples/nanogpt/train.py",
            ],
            "mlp_int8_lattice_representation": {
                "base": "independent reproducible frozen Gaussian initialization",
                "blocks": 13824,
                "code_bytes": 56623104,
                "elements": 56623104,
                "fp16_scale_bytes": 27648,
                "fp32_weight_bytes": 226492416,
                "optimizer_momentum": "dense_fp32_not_in_codec_count",
                "persistent_codec_bytes": 56650752,
                "runtime_base": "transient_dense_fp32",
                "runtime_weight": "transient_dense_fp32",
                "storage_ratio": 0.2501220703125,
                "storage_reduction": 3.998047828208882,
            },
            "ladder_role": "full_mlp_ambient_int8_lattice_smallest_rung",
            "ladder_slot": "full_attention_plus_full_mlp_int8_lattice",
            "launch_block_reason": None,
            "launch_ready": True,
            "mfu_measurement_protocol": (
                "foreground exact-config real-training preflight with one "
                "warmup and eight timed updates; includes dense Muon requests, "
                "causal int8 projections, and dense rematerialization for all "
                "attention-c_proj and MLP lattice matrices"
            ),
            "mfu_min_fraction": 0.2,
            "mfu_preflight_required": True,
            "monitoring_policy": (
                "foreground-poll the <=5 minute MFU gate; after a pass, use "
                "one idempotent terminal/error-only @Codex watchdog for the "
                "238-update run"
            ),
            "out_dir": (
                "/home/pro6000-9980x/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_fullattn_fullmlp_int8_lattice_0p5tpp/"
                "scientific"
            ),
            "practical_equivalence_policy": (
                "Require finite terminal validation CE <=5.3117, no more "
                "than +0.0200 behind the sealed full-attention parent 5.2917. "
                "A pass authorizes only a separate 5TPP decision."
            ),
            "registered_resume_protocol": (
                "atomic latest checkpoint with full RNG state, independent "
                "MLP int8 codes, FP16 running-max scales, reproducible base "
                "identities, dense Muon momentum, and no serialized dense MLP "
                "weight"
            ),
            "selection_endpoint": (
                "terminal step-238 fixed-window validation CE versus the "
                "sealed 124M/0.5TPP full-attention parent CE 5.2917"
            ),
        }
    )
    return config


def main() -> None:
    OUTPUT.write_text(json.dumps(build_config(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
