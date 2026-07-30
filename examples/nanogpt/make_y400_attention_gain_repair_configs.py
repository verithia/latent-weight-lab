from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
BASE_PATH = (
    CONFIG_DIR
    / "y400_mai_v2_124m_fullattn_blockfht_0p5tpp_mult1p00.json"
)
BASE_5TPP_PATH = (
    CONFIG_DIR / "y400_mai_v2_124m_fullattn_blockfht_5tpp_top1.json"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_attention_gain_repairs"
)
TARGETS = [
    "attn.c_attn.qk_headwise",
    "attn.c_attn.v",
    "attn.c_proj",
]


SPECS = {
    "outputgain": {
        "block_fht_output_gain_targets": TARGETS,
        "block_fht_input_gain_targets": [],
        "candidate_scope": (
            "selected attention-only BlockFHT chart plus per-output-channel "
            "multiplicative gain on QK-headwise, V, and c_proj"
        ),
    },
    "inputgain": {
        "block_fht_output_gain_targets": [],
        "block_fht_input_gain_targets": TARGETS,
        "candidate_scope": (
            "selected attention-only BlockFHT chart plus per-input-channel "
            "multiplicative gain on QK-headwise, V, and c_proj"
        ),
    },
    "dualgain": {
        "block_fht_output_gain_targets": TARGETS,
        "block_fht_input_gain_targets": TARGETS,
        "candidate_scope": (
            "selected attention-only BlockFHT chart plus bilateral "
            "per-output/per-input-channel multiplicative gains on "
            "QK-headwise, V, and c_proj"
        ),
    },
}


def make_config(slot: str, updates: dict[str, object]) -> dict[str, object]:
    config = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    config.update(updates)
    stem = (
        "y400_mai_v3_124m_fullattn_"
        f"{slot}_0p5tpp_lr24e4"
    )
    config.update(
        {
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "hpo_stage": (
                "attention_direction_repair_gain_screen_124m_0p5tpp"
            ),
            "ladder_role": "attention_direction_repair_screen",
            "ladder_slot": slot,
            "screen_only": True,
            "screen_only_resolution": (
                "poll this 238-update run directly; compare terminal fixed "
                "validation CE with the matched attention-only control; "
                "promote at most one stable gain geometry to 124M/5TPP"
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "2026-07-30 exact trajectory analysis found that the "
                    "deployed one-percent fixed BlockFHT tangent captures only "
                    "equal-rank-Haar chord energy; test the smallest "
                    "state-dependent diagonal chart extensions without a "
                    "learned dense basis"
                ),
                "recorded_at": "2026-07-30",
                "scope": (
                    "124M/0.5TPP output/input/bilateral channel-gain screen"
                ),
            },
            "candidate_learning_rate_resolution": (
                "hold the selected attention-only learning rate at 2.4e-3; "
                "change only channel-gain geometry"
            ),
            "practical_equivalence_policy": (
                "reject instability or terminal validation CE worse than the "
                "matched attention-only control by more than 0.02; among "
                "stable equivalent candidates prefer the smallest gain chart; "
                "promote at most one to the informative 124M/5TPP budget"
            ),
            "prelaunch_provenance_requirements": (
                "record commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed "
                "evaluation digest, and MFU certificate"
            ),
        }
    )
    return config


def make_outputgain_5tpp_config() -> dict[str, object]:
    config = json.loads(BASE_5TPP_PATH.read_text(encoding="utf-8"))
    stem = "y400_mai_v3_124m_fullattn_outputgain_5tpp_lr24e4"
    config.update(
        {
            "block_fht_output_gain_targets": TARGETS,
            "block_fht_input_gain_targets": [],
            "out_dir": f"{OUTPUT_ROOT}/{stem}",
            "candidate_scope": (
                "selected attention-only BlockFHT chart plus "
                "per-output-channel multiplicative gain on QK-headwise, V, "
                "and c_proj; no MLP replacement"
            ),
            "candidate_learning_rate_resolution": (
                "hold the selected attention-only 5TPP recipe at 2.4e-3; "
                "the 124M/0.5TPP gain screen selected output-only by the "
                "preregistered simplicity rule"
            ),
            "confirmation_source": (
                "2026-07-30 registered attention-gain screen: output val "
                "5.4890, input 5.4891, bilateral 5.4863, base attention "
                "5.4918; output selected because all are inside 0.01"
            ),
            "hpo_stage": (
                "attention_direction_repair_outputgain_confirmation_5tpp"
            ),
            "ladder_role": "attention_direction_repair_confirmation",
            "ladder_slot": "outputgain",
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "The exact production tangent captured only equal-rank "
                    "Haar energy. The registered 0.5TPP diagonal-gain screen "
                    "selected output-only for the informative 5TPP test."
                ),
                "recorded_at": "2026-07-30",
                "scope": (
                    "124M/5TPP output-channel gain attention-only "
                    "confirmation"
                ),
            },
            "practical_equivalence_policy": (
                "compare terminal fixed validation CE against dense 3.5401 "
                "and base attention-only 3.6744; target gap <=0.10 to dense "
                "and require a material improvement over base attention "
                "before any 20TPP promotion"
            ),
            "prelaunch_provenance_requirements": (
                "record commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed "
                "evaluation digest, and MFU certificate"
            ),
        }
    )
    return config


def main() -> None:
    for slot, updates in SPECS.items():
        path = CONFIG_DIR / (
            "y400_mai_v3_124m_fullattn_"
            f"{slot}_0p5tpp_lr24e4.json"
        )
        path.write_text(
            json.dumps(make_config(slot, updates), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(path)
    selected = (
        CONFIG_DIR
        / "y400_mai_v3_124m_fullattn_outputgain_5tpp_lr24e4.json"
    )
    selected.write_text(
        json.dumps(make_outputgain_5tpp_config(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(selected)


if __name__ == "__main__":
    main()
