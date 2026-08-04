from __future__ import annotations

import json

from examples.nanogpt.make_y400_attention_capacity_closure_configs import (
    BASE_NAME,
    CONFIG_DIR,
    SOURCE_DATA_DIR,
    STAGED_DATA_DIR,
    VARIANTS,
    destination_name,
    make_config,
)


def test_capacity_factorial_changes_only_registered_fields() -> None:
    base = json.loads((CONFIG_DIR / BASE_NAME).read_text(encoding="utf-8"))
    allowed = {
        "block_fht_attn_cayley_ranks",
        "block_fht_output_gain_targets",
        "candidate_scope",
        "compute_equivalence_sop",
        "confirmation_slot",
        "confirmation_source",
        "dense_fixed_validation_curve",
        "data_dir",
        "data_staging_policy",
        "data_staging_source",
        "execution_host",
        "hpo_stage",
        "ladder_role",
        "ladder_slot",
        "operator_override",
        "out_dir",
        "parent_dense_token_equivalent_penalty",
        "parent_fixed_validation_curve",
        "practical_equivalence_nll",
        "practical_equivalence_policy",
        "screen_only_resolution",
    }
    for slot, specification in VARIANTS.items():
        candidate = make_config(slot, specification)
        changed = {
            key
            for key in base.keys() | candidate.keys()
            if base.get(key) != candidate.get(key)
        }
        assert changed <= allowed
        assert {
            "candidate_scope",
            "compute_equivalence_sop",
            "confirmation_slot",
            "dense_fixed_validation_curve",
            "hpo_stage",
            "ladder_role",
            "ladder_slot",
            "operator_override",
            "out_dir",
            "parent_dense_token_equivalent_penalty",
            "parent_fixed_validation_curve",
            "practical_equivalence_nll",
            "practical_equivalence_policy",
        } <= changed
        if specification["output_gain"]:
            assert "block_fht_output_gain_targets" in changed
        else:
            assert "block_fht_output_gain_targets" not in changed
        assert candidate["max_iters"] == 2373
        assert candidate["data_dir"] == STAGED_DATA_DIR
        assert candidate["data_staging_source"] == SOURCE_DATA_DIR
        assert candidate["data_manifest_sha256"] == base[
            "data_manifest_sha256"
        ]
        assert candidate["fixed_eval_indices"] is True
        assert candidate["eval_iters"] == 400
        assert candidate["block_fht_attn_cayley_lr_scale"] == 10.0 / 3.0
        assert candidate["block_fht_attn_cayley_bilateral_targets"] == [
            "attn.c_attn.qk_headwise",
            "attn.c_attn.v",
        ]
        assert candidate["mfu_min_fraction"] >= 0.20


def test_factorial_is_qk_capacity_crossed_with_radial_gain() -> None:
    qk64 = make_config("qk64", VARIANTS["qk64"])
    outputgain = make_config("outputgain", VARIANTS["outputgain"])
    combined = make_config("qk64_outputgain", VARIANTS["qk64_outputgain"])
    assert qk64["block_fht_attn_cayley_ranks"] == {
        "attn.c_attn.qk_headwise": 64,
        "attn.c_attn.v": 16,
        "attn.c_proj": 8,
    }
    assert qk64["block_fht_output_gain_targets"] == []
    assert outputgain["block_fht_attn_cayley_ranks"]["attn.c_attn.qk_headwise"] == 32
    assert outputgain["block_fht_output_gain_targets"]
    assert combined["block_fht_attn_cayley_ranks"] == qk64[
        "block_fht_attn_cayley_ranks"
    ]
    assert combined["block_fht_output_gain_targets"] == outputgain[
        "block_fht_output_gain_targets"
    ]


def test_checked_in_configs_match_generator() -> None:
    for slot, specification in VARIANTS.items():
        actual = json.loads(
            (CONFIG_DIR / destination_name(slot)).read_text(encoding="utf-8")
        )
        assert actual == make_config(slot, specification)
