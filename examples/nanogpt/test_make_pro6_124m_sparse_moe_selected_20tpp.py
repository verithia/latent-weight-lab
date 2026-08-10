from __future__ import annotations

import json
import math

from examples.nanogpt.make_pro6_124m_sparse_moe_selected_20tpp import (
    BASE,
    make_selected_20tpp,
)


def test_selected_20tpp_changes_only_horizon_identity_fields() -> None:
    source = json.loads(BASE.read_text())
    candidate = make_selected_20tpp(source)
    mutable = {
        "confirmation_slot",
        "eval_interval",
        "experiment_role",
        "hpo_stage",
        "lr_decay_iters",
        "max_iters",
        "mfu_preflight_certificate",
        "monitoring_policy",
        "out_dir",
        "planned_tokens",
        "planned_tpp_active",
        "promotion_source",
        "scheduled_tokens",
        "scheduled_tpp_active",
        "warmup_iters",
    }
    assert {key: value for key, value in candidate.items() if key not in mutable} == {
        key: value for key, value in source.items() if key not in mutable
    }


def test_selected_20tpp_exact_active_parameter_schedule_and_contract() -> None:
    candidate = make_selected_20tpp(json.loads(BASE.read_text()))
    expected_iters = math.ceil(
        20 * candidate["estimated_active_params"] / candidate["tokens_per_iter"]
    )
    assert candidate["max_iters"] == expected_iters == 9495
    assert candidate["scheduled_tokens"] == expected_iters * candidate["tokens_per_iter"]
    assert candidate["planned_tokens"] == 20 * candidate["estimated_active_params"]
    assert candidate["moe_num_experts"] == 8
    assert candidate["moe_top_k"] == 2
    assert candidate["moe_expert_hidden_multiplier"] == 2
    assert candidate["learning_rate"] == 0.0024
    assert candidate["eval_interval"] == 2374
    assert candidate["warmup_iters"] == 94
    assert "124m_sparse_moe_5tpp_ranking.json rank 1" in candidate["promotion_source"]
    assert candidate["mfu_preflight_required"] is True
