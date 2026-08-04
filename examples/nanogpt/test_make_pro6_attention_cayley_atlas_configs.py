from __future__ import annotations

import json

from examples.nanogpt.make_pro6_attention_cayley_atlas_configs import (
    BASE_NAME,
    CONFIG_DIR,
    VARIANTS,
    destination_name,
    make_config,
)


def test_atlas_candidates_change_only_registered_fields() -> None:
    base = json.loads((CONFIG_DIR / BASE_NAME).read_text(encoding="utf-8"))
    allowed = {
        "atlas_performance_selection",
        "block_fht_attn_cayley_atlas_start_steps",
        "candidate_scope",
        "compute_equivalence_sop",
        "confirmation_slot",
        "confirmation_source",
        "hpo_stage",
        "ladder_role",
        "ladder_slot",
        "operator_override",
        "out_dir",
        "practical_equivalence_nll",
        "practical_equivalence_policy",
        "screen_only_resolution",
    }
    for slot, start_steps in VARIANTS:
        candidate = make_config(slot, start_steps)
        changed = {
            key
            for key in base.keys() | candidate.keys()
            if base.get(key) != candidate.get(key)
        }
        assert changed <= allowed
        assert candidate["block_fht_attn_cayley_atlas_start_steps"] == list(
            start_steps
        )
        assert candidate["block_fht_attn_cayley_ranks"] == {
            "attn.c_attn.qk_headwise": 32,
            "attn.c_attn.v": 16,
            "attn.c_proj": 8,
        }
        assert candidate["block_fht_output_gain_targets"] == []
        assert candidate["max_iters"] == 2373
        assert candidate["mfu_min_fraction"] >= 0.20


def test_atlas_fallback_order_is_mfu_only() -> None:
    assert VARIANTS == (
        ("phase4", (0, 594, 1188, 1782)),
        ("phase3", (0, 594, 1188)),
        ("phase2", (0, 594)),
    )
    for slot, start_steps in VARIANTS:
        policy = make_config(slot, start_steps)["atlas_performance_selection"]
        assert "MFU >=20%" in policy
        assert "never use validation CE" in policy


def test_checked_in_configs_match_generator() -> None:
    for slot, start_steps in VARIANTS:
        actual = json.loads(
            (CONFIG_DIR / destination_name(slot)).read_text(encoding="utf-8")
        )
        assert actual == make_config(slot, start_steps)
