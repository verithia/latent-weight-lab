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
HORIZON_CAPACITY_PATH = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_cproj8_5tpp_lr24e4.json"
)


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


def make_promotion_config() -> dict[str, object]:
    config = make_config()
    stem = "y400_mai_v3_124m_fullattn_cayley_rank2_5tpp_lr24e4"
    config.update(
        {
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "eval_interval": 594,
            "max_iters": 2373,
            "lr_decay_iters": 2373,
            "warmup_iters": 23,
            "planned_tokens": 621868800,
            "planned_tpp": 5.0,
            "scheduled_tokens": 622067712,
            "scheduled_tpp": 5.001599308407175,
            "hpo_stage": (
                "attention_direction_repair_lowrank_cayley_124m_5tpp"
            ),
            "ladder_role": "attention_direction_repair_confirmation",
            "ladder_slot": "lowrank_cayley_rank2_top1",
            "screen_only": False,
            "screen_only_resolution": None,
            "confirmation_slot": "top1",
            "confirmation_source": (
                "registered 124M/0.5TPP screen: Cayley val 5.4822, "
                "attention control 5.4918, dense 5.4890"
            ),
            "candidate_learning_rate_resolution": (
                "hold the selected attention-only 5TPP recipe at 2.4e-3 "
                "and AdamW fallback multiplier at 0.3; change only the "
                "task-gradient-oracle-selected orthogonal chart"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "The exact 124M/0.5TPP rank-2-pair Cayley screen "
                    "finished at val 5.4822, improving the attention control "
                    "by 0.0096 and dense by 0.0068. Promote to the informative "
                    "5TPP budget to test closure of the long-horizon gap."
                ),
                "recorded_at": "2026-07-30",
                "scope": (
                    "124M/5TPP low-rank Cayley attention-only confirmation"
                ),
            },
            "practical_equivalence_policy": (
                "compare terminal fixed validation CE against dense 3.5401 "
                "and attention control 3.6744; closure requires val <=3.6401 "
                "and a material improvement over the control"
            ),
        }
    )
    return config


def make_targetwise_config() -> dict[str, object]:
    config = make_config()
    stem = (
        "y400_mai_v3_124m_fullattn_cayley_targetwise_rank2_"
        "0p5tpp_lr24e4"
    )
    config.update(
        {
            "block_fht_attn_cayley_output_targets": [
                "attn.c_attn.qk_headwise",
            ],
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "hpo_stage": (
                "attention_direction_repair_targetwise_cayley_124m_0p5tpp"
            ),
            "ladder_slot": "targetwise_cayley_rank2",
            "candidate_scope": (
                "task-gradient-oracle-selected rank-2-pair Cayley side per "
                "target: output/left orbit for shared QK and input/right "
                "orbit for V and c_proj; no MLP replacement, learned dense "
                "basis, or additive low-rank residual"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "Exact rank-4 skew recovery is 0.2770 left versus "
                    "0.1173 right for QK, while V favors right "
                    "0.3967 versus left 0.2837 and c_proj favors right "
                    "0.3307 versus left 0.2826. Replace only the "
                    "misallocated QK input chart."
                ),
                "recorded_at": "2026-07-30",
                "scope": (
                    "124M/0.5TPP targetwise left/right Cayley allocation"
                ),
            },
            "screen_only_resolution": (
                "poll this 238-update run directly; compare with uniform-right "
                "Cayley 5.4822, attention control 5.4918, and dense 5.4890"
            ),
            "practical_equivalence_policy": (
                "require MFU >=20%, stability, and terminal validation CE "
                "better than uniform-right Cayley 5.4822 before any 5TPP "
                "promotion"
            ),
        }
    )
    return config


def make_targetwise_promotion_config() -> dict[str, object]:
    config = make_targetwise_config()
    stem = (
        "y400_mai_v3_124m_fullattn_cayley_targetwise_rank2_"
        "5tpp_lr24e4"
    )
    config.update(
        {
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "eval_interval": 594,
            "max_iters": 2373,
            "lr_decay_iters": 2373,
            "warmup_iters": 23,
            "planned_tokens": 621868800,
            "planned_tpp": 5.0,
            "scheduled_tokens": 622067712,
            "scheduled_tpp": 5.001599308407175,
            "hpo_stage": (
                "attention_direction_repair_targetwise_cayley_124m_5tpp"
            ),
            "ladder_role": "attention_direction_repair_confirmation",
            "ladder_slot": "targetwise_cayley_rank2_top1",
            "screen_only": False,
            "screen_only_resolution": None,
            "confirmation_slot": "top1",
            "confirmation_source": (
                "registered 124M/0.5TPP targetwise screen: val 5.4738, "
                "uniform-right Cayley 5.4822, attention control 5.4918, "
                "dense 5.4890"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "The targetwise left-QK/right-V/right-cproj screen "
                    "finished at val 5.4738, improving uniform-right Cayley "
                    "by 0.0084, attention control by 0.0180, and dense by "
                    "0.0152. Promote to the informative 5TPP budget."
                ),
                "recorded_at": "2026-07-30",
                "scope": (
                    "124M/5TPP targetwise Cayley attention-only confirmation"
                ),
            },
            "practical_equivalence_policy": (
                "compare terminal fixed validation CE against dense 3.5401, "
                "attention control 3.6744, output gain 3.6712, and "
                "uniform-right Cayley 3.6748; closure requires val <=3.6401"
            ),
        }
    )
    return config


def make_horizon_capacity_chartseed2_config() -> dict[str, object]:
    """Repeat the admitted capacity chart with a new Cayley frame only."""
    config = json.loads(HORIZON_CAPACITY_PATH.read_text(encoding="utf-8"))
    stem = (
        "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_"
        "qk32_v16_cproj8_chartseed271828_5tpp_lr24e4"
    )
    config.update(
        {
            "block_fht_attn_cayley_seed": 271828,
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "hpo_stage": (
                "attention_direction_repair_horizon_capacity_chartseed2_"
                "124m_5tpp"
            ),
            "ladder_slot": (
                "horizon_capacity_qk32_v16_cproj8_chartseed271828"
            ),
            "confirmation_slot": "horizon_capacity_chartseed2",
            "confirmation_source": (
                "the primary direction-correct pair-rank 32/16/8 chart "
                "reduced the matched step-594 gap from +0.1384 to +0.0833; "
                "change only its identity-initialized fixed Cayley frame"
            ),
            "candidate_scope": (
                "exact chart-seed replication of the terminal-oracle-selected "
                "QK-output pair-rank 32, V-input pair-rank 16, and c-proj-"
                "output pair-rank 8 candidate; same model/data seeds, "
                "BlockFHT decoder, optimizer, schedule, and initial function"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "The primary high-capacity chart crossed the requested "
                    "+0.10 gap threshold at step 594. A second fixed Cayley "
                    "frame tests whether that gain is structural rather than "
                    "specific to one random identity-initialized orbit basis."
                ),
                "recorded_at": "2026-07-30",
                "scope": (
                    "124M/5TPP direction-correct high-capacity attention "
                    "chart-seed replication"
                ),
            },
            "practical_equivalence_policy": (
                "compare every fixed validation checkpoint with the primary "
                "pair-rank 32/16/8 run, dense 3.5401, and attention control "
                "3.6744; terminal closure requires val <=3.6401"
            ),
        }
    )
    return config


def main() -> None:
    configs = {
        "y400_mai_v3_124m_fullattn_cayley_rank2_0p5tpp_lr24e4.json":
            make_config(),
        "y400_mai_v3_124m_fullattn_cayley_rank2_5tpp_lr24e4.json":
            make_promotion_config(),
        "y400_mai_v3_124m_fullattn_cayley_targetwise_rank2_0p5tpp_lr24e4.json":
            make_targetwise_config(),
        "y400_mai_v3_124m_fullattn_cayley_targetwise_rank2_5tpp_lr24e4.json":
            make_targetwise_promotion_config(),
        "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_cproj8_chartseed271828_5tpp_lr24e4.json":
            make_horizon_capacity_chartseed2_config(),
    }
    for filename, config in configs.items():
        path = CONFIG_DIR / filename
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
