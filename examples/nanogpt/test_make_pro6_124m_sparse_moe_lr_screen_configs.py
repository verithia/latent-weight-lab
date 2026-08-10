from __future__ import annotations

import json

from examples.nanogpt.make_pro6_124m_sparse_moe_lr_screen_configs import (
    BASE,
    VARIANTS,
    make_variant,
)


def test_lr_screen_changes_only_registered_lr_identity_fields() -> None:
    source = json.loads(BASE.read_text())
    mutable = {
        "learning_rate",
        "min_lr",
        "out_dir",
        "mfu_preflight_certificate",
    }
    for label, learning_rate in VARIANTS.items():
        candidate = make_variant(source, label, learning_rate)
        assert candidate["learning_rate"] == learning_rate
        assert candidate["min_lr"] == learning_rate / 10.0
        assert f"lr{label}" in candidate["out_dir"]
        assert f"lr{label}" in candidate["mfu_preflight_certificate"]
        assert {
            key: value for key, value in candidate.items() if key not in mutable
        } == {key: value for key, value in source.items() if key not in mutable}


def test_lr_screen_retains_complete_expert_and_promotion_contract() -> None:
    source = json.loads(BASE.read_text())
    assert source["moe_num_experts"] == 8
    assert source["moe_top_k"] == 2
    assert source["moe_expert_hidden_multiplier"] == 2
    assert source["estimated_active_params"] == 124447488
    assert source["estimated_stored_params"] == 294316800
    assert source["max_iters"] == 238
    assert source["scheduled_tpp_active"] > 0.5
