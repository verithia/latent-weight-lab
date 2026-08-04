from __future__ import annotations

import json

from examples.nanogpt.make_pro6_attention_cayley_hybrid_left_muon_config import (
    CONFIG_DIR,
    DESTINATION_NAME,
    SOURCE_NAME,
    make_config,
)


def test_hybrid_screen_is_optimizer_isolated() -> None:
    source = json.loads((CONFIG_DIR / SOURCE_NAME).read_text(encoding="utf-8"))
    candidate = make_config(source)
    allowed = {
        "block_fht_attn_cayley_factor_optimizer",
        "block_fht_attn_cayley_muon_lr_scale",
        "candidate_scope",
        "confirmation_slot",
        "confirmation_source",
        "data_dir",
        "execution_host",
        "factor_optimizer_policy",
        "host_transfer_policy",
        "host_transfer_source_config",
        "hpo_stage",
        "ladder_role",
        "ladder_slot",
        "operator_override",
        "out_dir",
        "practical_equivalence_policy",
        "screen_only_resolution",
    }
    changed = {
        key
        for key in source.keys() | candidate.keys()
        if source.get(key) != candidate.get(key)
    }
    assert changed <= allowed
    assert candidate["block_fht_attn_cayley_factor_optimizer"] == (
        "hybrid_left_muon"
    )
    assert candidate["block_fht_attn_cayley_ranks"] == {
        "attn.c_attn.qk_headwise": 32,
        "attn.c_attn.v": 16,
        "attn.c_proj": 8,
    }
    assert candidate["max_iters"] == 238
    assert candidate["mfu_min_fraction"] == 0.20
    assert "dense learned basis" in candidate["candidate_scope"]
    assert DESTINATION_NAME.endswith("_0p5tpp_lr24e4.json")
