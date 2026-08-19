import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_shared_dense_mlp_trunk_0p5tpp.json"
)


def load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_candidate_restores_shared_full_rank_mlp_directions() -> None:
    config = load_config()
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
        "attn.c_proj",
    ]
    assert config["mlp_shared_dense_trunk"] is True
    assert config["block_fht_ffn_pregelu_gain"] is True
    assert config["block_fht_mlp_residual_output_gain"] is True
    assert config["block_fht_residual_base_scale"] == 0.0


def test_parameter_and_compute_accounting_is_exact() -> None:
    config = load_config()
    layers = int(config["n_layer"])
    width = int(config["n_embd"])
    hidden = 4 * width
    dense = 2 * layers * width * hidden
    shared = 2 * width * hidden
    private = layers * (hidden + width)
    compact = shared + private

    assert config["estimated_dense_mlp_parameters"] == dense
    assert config["estimated_shared_dense_mlp_parameters"] == shared
    assert config["estimated_layer_private_diagonal_parameters"] == private
    assert config["estimated_total_learned_mlp_parameters"] == compact
    assert math.isclose(
        float(config["estimated_mlp_parameter_compression_ratio"]),
        dense / compact,
        rel_tol=1e-7,
    )
    assert math.isclose(
        float(config["estimated_mlp_parameter_saving_fraction"]),
        1.0 - compact / dense,
        rel_tol=1e-7,
    )
    assert config["estimated_extra_activation_multiplies_per_token"] == (
        layers * hidden
    )
    assert config["estimated_extra_cproj_weight_multiplies_per_update"] == (
        layers * width * hidden
    )


def test_loss_and_performance_endpoints_are_preregistered() -> None:
    config = load_config()
    decision = config["preregistered_decision_rule"]
    assert decision["structural_pass_validation_ce_maximum"] == 5.5918
    assert config["matched_attention_control_terminal_validation_loss"] == 5.4918
    assert config["mfu_preflight_required"] is True
    assert config["mfu_min_fraction"] >= 0.20
    assert config["terminal_eval_required"] is True
    assert config["fixed_eval_indices"] is True
    assert config["monitoring_policy"].endswith(
        "one idempotent terminal-or-error @Codex callback"
    )
