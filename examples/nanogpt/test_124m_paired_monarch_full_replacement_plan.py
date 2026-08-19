import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_fullmlp_paired_monarch16_0p5tpp.json"
)


def load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_candidate_replaces_every_attention_and_mlp_matrix() -> None:
    config = load_config()
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    ]
    assert config["block_fht_mlp_paired_monarch_block_width"] == 16
    assert config["block_fht_cache_weights"] is True
    assert config["block_fht_residual_base_scale"] == 0.0


def test_registered_parameter_and_flop_accounting_is_exact() -> None:
    config = load_config()
    layers = int(config["n_layer"])
    model_width = int(config["n_embd"])
    hidden_width = 4 * model_width
    block_width = int(config["block_fht_mlp_paired_monarch_block_width"])
    dense = 2 * layers * model_width * hidden_width
    monarch = 2 * layers * hidden_width * block_width
    block_fht = math.ceil(dense * float(config["block_fht_latent_ratio"]))
    compact = monarch + block_fht

    conventional_active = (
        int(config["vocab_size"]) * model_width
        + int(config["block_size"]) * model_width
        + layers * (12 * model_width * model_width + 13 * model_width)
        + 2 * model_width
    )

    assert config["estimated_active_params"] == conventional_active
    assert config["estimated_dense_mlp_parameters"] == dense
    assert config["estimated_monarch_coordinates"] == monarch
    assert config["estimated_mlp_blockfht_coordinates"] == block_fht
    assert config["estimated_total_compact_mlp_coordinates"] == compact
    assert math.isclose(
        float(config["estimated_mlp_parameter_compression_ratio"]),
        dense / compact,
        rel_tol=1e-6,
    )
    assert math.isclose(
        float(config["estimated_mlp_parameter_saving_fraction"]),
        1.0 - compact / dense,
        rel_tol=1e-6,
    )
    assert config["estimated_monarch_forward_materialization_flops_per_update"] == (
        8 * layers * model_width * hidden_width * block_width
    )
    assert config["estimated_monarch_fused_activation_flops_per_token_if_not_folded"] == (
        8 * layers * hidden_width * block_width
    )


def test_loss_and_performance_endpoints_are_preregistered() -> None:
    config = load_config()
    decision = config["preregistered_decision_rule"]
    assert decision["structural_pass_validation_ce_maximum"] == 5.5918
    assert config["matched_attention_control_terminal_validation_loss"] == 5.4918
    assert math.isclose(
        config["matched_full_mlp_error_feedback_terminal_validation_loss"],
        5.529316425323486,
    )
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] >= 0.20
    assert config["terminal_eval_required"] is True
    assert config["fixed_eval_indices"] is True
    assert config["monitoring_policy"].endswith(
        "one idempotent terminal-or-error @Codex callback"
    )
