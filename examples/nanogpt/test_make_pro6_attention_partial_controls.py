import json
from pathlib import Path

from examples.nanogpt.make_pro6_attention_partial_controls import (
    PROJECTION,
    QK,
    VALUE,
    build,
)


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_qk64_outputgain_5tpp_lr24e4.json"


def test_qk_only_control_changes_only_registered_attention_scope() -> None:
    parent = json.loads(PARENT.read_text())
    candidate = build(parent)
    assert candidate["block_fht_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_output_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_bilateral_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_ranks"] == {QK: 64}
    assert candidate["block_fht_output_gain_targets"] == [QK]
    for key in (
        "learning_rate",
        "max_iters",
        "warmup_iters",
        "batch_size",
        "gradient_accumulation_steps",
        "train_data_seed",
        "model_seed",
        "data_manifest_sha256",
        "optimizer",
        "muon_momentum",
        "muon_adamw_lr_scale",
    ):
        assert candidate[key] == parent[key]
    assert candidate["mfu_preflight_required"] is True
    assert candidate["mfu_min_fraction"] >= 0.2
    assert "partial_controls" in candidate["out_dir"]


def test_qkv_control_keeps_only_output_projection_dense() -> None:
    parent = json.loads(PARENT.read_text())
    candidate = build(parent, "qkv_only")
    assert candidate["block_fht_targets"] == [QK, VALUE]
    assert candidate["block_fht_attn_cayley_targets"] == [QK, VALUE]
    assert candidate["block_fht_attn_cayley_output_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_bilateral_targets"] == [QK, VALUE]
    assert candidate["block_fht_attn_cayley_ranks"] == {QK: 64, VALUE: 16}
    assert candidate["block_fht_output_gain_targets"] == [QK, VALUE]
    assert "qkv_only" in candidate["out_dir"]


def test_qk_cproj_control_keeps_only_value_dense() -> None:
    parent = json.loads(PARENT.read_text())
    candidate = build(parent, "qk_cproj_only")
    assert candidate["block_fht_targets"] == [QK, PROJECTION]
    assert candidate["block_fht_attn_cayley_targets"] == [QK, PROJECTION]
    assert candidate["block_fht_attn_cayley_output_targets"] == [QK, PROJECTION]
    assert candidate["block_fht_attn_cayley_bilateral_targets"] == [QK]
    assert candidate["block_fht_attn_cayley_ranks"] == {QK: 64, PROJECTION: 8}
    assert candidate["block_fht_output_gain_targets"] == [QK, PROJECTION]
    assert "qk_cproj_only" in candidate["out_dir"]


def test_unknown_partial_scope_is_rejected() -> None:
    parent = json.loads(PARENT.read_text())
    try:
        build(parent, "not_registered")
    except ValueError as error:
        assert "unknown partial-attention scope" in str(error)
    else:
        raise AssertionError("unknown scope was accepted")
