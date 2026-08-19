import json
import math
from pathlib import Path


CONFIG = (
    Path(__file__).parent
    / "configs/pro6_mai_v3_124m_fullattn_shared_dense_tri_monarch16_0p5tpp.json"
)


def load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_registered_scope_is_shared_core_with_three_private_transports() -> None:
    config = load_config()
    assert config["mlp_shared_dense_trunk"] is True
    assert config["mlp_shared_dense_tri_monarch_block_width"] == 16
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
        "attn.c_proj",
    ]
    assert config["block_fht_muon_latent_targets"] == []
    assert config["block_fht_cache_weights"] is True


def test_registered_parameter_and_flop_accounting_is_exact() -> None:
    config = load_config()
    layers = int(config["n_layer"])
    width = int(config["n_embd"])
    hidden = 4 * width
    block = int(config["mlp_shared_dense_tri_monarch_block_width"])
    dense = 2 * layers * width * hidden
    shared = 2 * width * hidden
    transports = layers * 2 * block * (hidden + 2 * width)
    diagonals = layers * (hidden + width)
    compact = shared + transports + diagonals

    assert config["estimated_dense_mlp_parameters"] == dense
    assert config["estimated_shared_dense_mlp_parameters"] == shared
    assert config["estimated_layer_private_tri_monarch_parameters"] == transports
    assert config["estimated_layer_private_diagonal_parameters"] == diagonals
    assert config["estimated_total_learned_mlp_parameters"] == compact
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
    assert config["estimated_tri_monarch_forward_materialization_flops_per_update"] == (
        16 * layers * width * hidden * block
    )
    assert config["estimated_tri_monarch_fused_flops_per_token_if_not_folded"] == (
        8 * layers * block * (hidden + width)
    )


def test_loss_and_system_gates_are_frozen() -> None:
    config = load_config()
    decision = config["preregistered_decision_rule"]
    assert decision["structural_pass_validation_ce_maximum"] == 5.5918
    assert config["matched_attention_control_terminal_validation_loss"] == 5.4918
    assert config["matched_private_random_residual_terminal_validation_loss"] == 5.6824
    assert config["matched_shared_dense_trunk_terminal_validation_loss"] == 5.6971
    assert config["max_iters"] == 238
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] >= 0.20
    assert config["launch_ready"] is True
    assert config["monitoring_policy"].endswith(
        "one idempotent terminal-or-error @Codex callback"
    )
