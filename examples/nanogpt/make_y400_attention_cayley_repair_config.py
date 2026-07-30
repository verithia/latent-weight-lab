from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
BASE_PATH = (
    CONFIG_DIR
    / "y400_mai_v2_124m_fullattn_blockfht_0p5tpp_mult1p00.json"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_attention_cayley_repair"
)
TARGETS = [
    "attn.c_attn.qk_headwise",
    "attn.c_attn.v",
    "attn.c_proj",
]


def make_config() -> dict[str, object]:
    config = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    stem = "y400_mai_v3_124m_fullattn_cayley_rank2_0p5tpp_lr24e4"
    config.update(
        {
            "block_fht_output_gain_targets": [],
            "block_fht_input_gain_targets": [],
            "block_fht_attn_cayley_targets": TARGETS,
            "block_fht_attn_cayley_rank": 2,
            "block_fht_attn_cayley_scale": 1.0,
            "block_fht_attn_cayley_seed": 618033,
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "hpo_stage": (
                "attention_direction_repair_lowrank_cayley_124m_0p5tpp"
            ),
            "ladder_role": "attention_direction_repair_screen",
            "ladder_slot": "lowrank_cayley_rank2",
            "screen_only": True,
            "screen_only_resolution": (
                "poll this 238-update run directly; compare terminal fixed "
                "validation CE with attention-only 5.4918 and dense 5.4890; "
                "do not attach the long-run watchdog"
            ),
            "candidate_scope": (
                "selected attention-only BlockFHT chart plus three "
                "identity-initialized rank-2-pair Cayley input rotations per "
                "layer: one shared by all QK heads, one for V, and one before "
                "c_proj; no MLP replacement, learned dense basis, or additive "
                "low-rank residual"
            ),
            "candidate_learning_rate_resolution": (
                "hold the selected attention-only LR at 2.4e-3 and the "
                "registered AdamW fallback multiplier at 0.3; change only "
                "the measured orthogonal chart"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "Exact clipped-gradient analysis found diagonal gain "
                    "recovery near 0.001-0.002, while a rank-4 right-skew "
                    "direction recovered at least 0.117/0.331/0.397 for "
                    "shared-QK/c_proj/V. Rank two Cayley vector pairs are the "
                    "smallest tested structure clearing 0.10 for every target."
                ),
                "recorded_at": "2026-07-30",
                "scope": (
                    "124M/0.5TPP identity-initialized low-rank orthogonal "
                    "attention chart"
                ),
            },
            "practical_equivalence_policy": (
                "require stable terminal CE no worse than attention control "
                "by 0.02 and MFU >=20%; promote to 5TPP only if terminal CE "
                "improves attention control, with preference for a material "
                "gain because the output-gain repair already failed to alter "
                "the longer-budget trajectory"
            ),
            "prelaunch_provenance_requirements": (
                "record commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed "
                "evaluation digest, and synchronous MFU certificate"
            ),
        }
    )
    return config


def main() -> None:
    path = (
        CONFIG_DIR
        / "y400_mai_v3_124m_fullattn_cayley_rank2_0p5tpp_lr24e4.json"
    )
    path.write_text(
        json.dumps(make_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
