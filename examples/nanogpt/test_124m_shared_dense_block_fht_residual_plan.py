import json
import math
from pathlib import Path


CONFIG = Path(__file__).parent / "configs/pro6_mai_v3_124m_fullattn_shared_dense_blockfht_residual_0p5tpp.json"


def test_registered_hybrid_scope_and_controls() -> None:
    config = json.loads(CONFIG.read_text())
    assert config["mlp_shared_dense_block_fht_residual"] is True
    assert config["mlp_shared_dense_trunk"] is False
    assert math.isclose(
        config["mlp_shared_dense_block_fht_residual_scale"],
        math.sqrt(0.5),
    )
    assert {"mlp.c_fc", "mlp.c_proj"}.issubset(
        config["block_fht_targets"]
    )
    assert set(config["block_fht_muon_latent_targets"]) == {
        "mlp.c_fc",
        "mlp.c_proj",
    }
    assert config["matched_attention_control_terminal_validation_loss"] == 5.4918
    assert config["matched_shared_dense_trunk_terminal_validation_loss"] == 5.6971


def test_registered_hybrid_accounting_and_gate() -> None:
    config = json.loads(CONFIG.read_text())
    total = (
        config["estimated_shared_dense_mlp_parameters"]
        + config["estimated_layer_private_block_fht_parameters"]
        + config["estimated_layer_private_diagonal_parameters"]
    )
    assert total == config["estimated_total_learned_mlp_parameters"]
    assert math.isclose(
        config["estimated_mlp_parameter_compression_ratio"],
        config["estimated_dense_mlp_parameters"] / total,
        rel_tol=1e-6,
    )
    assert config["estimated_mlp_parameter_saving_fraction"] > 0.90
    assert config["mfu_min_fraction"] == 0.2
    assert config["mfu_preflight_required"] is True
    assert config["max_iters"] == 238
    assert config["launch_ready"] is True
