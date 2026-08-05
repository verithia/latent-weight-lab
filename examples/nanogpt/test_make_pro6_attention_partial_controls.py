import json
from pathlib import Path

from examples.nanogpt.make_pro6_attention_partial_controls import QK, build


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
