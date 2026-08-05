import json
from pathlib import Path

from examples.nanogpt.make_pro6_attention_muon_geometry_control import build


ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_muon_5tpp_attention_trajectory_replay_lr24e4.json"
)


def test_control_changes_only_optimizer_geometry_and_diagnostics() -> None:
    parent = json.loads(PARENT.read_text())
    candidate = build(parent)
    assert candidate["method"] == "baseline"
    assert candidate["optimizer"] == "muon"
    assert candidate["muon_split_attention_qkv_rows"] is True
    assert candidate["mfu_preflight_required"] is True
    assert candidate["mfu_min_fraction"] >= 0.2
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "max_iters",
        "warmup_iters",
        "learning_rate",
        "min_lr",
        "lr_decay_iters",
        "weight_decay",
        "beta1",
        "beta2",
        "muon_momentum",
        "muon_ns_steps",
        "muon_adamw_lr_scale",
        "model_seed",
        "train_data_seed",
        "eval_seed",
        "eval_interval",
        "eval_iters",
        "eval_batch_size",
        "data_manifest_sha256",
    ):
        assert candidate[key] == parent[key]
    assert "optimizer_probe_steps" not in candidate
    assert "trajectory_snapshot_interval" not in candidate
    assert "partial_controls" in candidate["out_dir"]
