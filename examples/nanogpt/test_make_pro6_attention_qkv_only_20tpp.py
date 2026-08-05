import json

from examples.nanogpt.make_pro6_attention_qkv_only_20tpp import (
    PARENT,
    PLAN,
    build,
    validate_evidence,
)


def test_evidence_is_hash_pinned_and_terminal() -> None:
    plan = json.loads(PLAN.read_text())
    validate_evidence(plan)


def test_transform_changes_only_registered_fields() -> None:
    parent = json.loads(PARENT.read_text())
    plan = json.loads(PLAN.read_text())
    candidate = build(parent, plan)
    allowed = set(plan["registered_config_transform"]["allowed_changes"])
    changed = {
        key
        for key in set(parent) | set(candidate)
        if parent.get(key) != candidate.get(key)
    }
    assert changed == allowed
    assert candidate["max_iters"] == 9489
    assert candidate["eval_interval"] == 2373
    assert candidate["warmup_iters"] == 94
    assert candidate["planned_tpp"] == 20.0
    assert candidate["learning_rate"] == 0.0024
    assert candidate["min_lr"] == 0.00024
    assert candidate["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
    ]
    assert candidate["init_from"] == parent["init_from"] if "init_from" in parent else True
    assert "20tpp" in candidate["out_dir"]


def test_transform_preserves_scientific_invariants() -> None:
    parent = json.loads(PARENT.read_text())
    plan = json.loads(PLAN.read_text())
    candidate = build(parent, plan)
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "block_size",
        "optimizer",
        "muon_momentum",
        "muon_ns_steps",
        "muon_adamw_lr_scale",
        "weight_decay",
        "model_seed",
        "train_data_seed",
        "data_manifest_sha256",
        "block_fht_attn_cayley_ranks",
        "block_fht_output_gain_targets",
    ):
        assert candidate[key] == parent[key]
    assert candidate["mfu_preflight_required"] is True
    assert candidate["mfu_min_fraction"] >= 0.20
