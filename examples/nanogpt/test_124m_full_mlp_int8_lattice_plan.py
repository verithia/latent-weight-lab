import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_full_mlp_int8_lattice_0p5tpp_plan.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_pins_ambient_mlp_codec_and_strict_parent_gate() -> None:
    plan = json.loads(PLAN.read_text())
    config_path = ROOT / plan["candidate"]["config"]
    attention_result = ROOT / plan["causal_basis"]["attention_result"]["path"]
    mfu_result = ROOT / plan["mfu_result"]["path"]
    assert sha256(config_path) == plan["candidate"]["config_sha256"]
    assert sha256(attention_result) == (
        plan["causal_basis"]["attention_result"]["sha256"]
    )
    assert sha256(mfu_result) == plan["mfu_result"]["sha256"]
    config = json.loads(config_path.read_text())
    assert config["block_fht_mlp_int8_lattice_targets"] == [
        "mlp.c_fc",
        "mlp.c_proj",
    ]
    assert config["block_fht_mlp_int8_lattice_block_size"] == 4096
    assert plan["representation"]["ambient_direction_support"] is True
    assert plan["representation"]["mlp_persistent_weight_reduction"] > 3.99
    assert plan["terminal_gate"]["maximum_validation_ce_delta_to_parent"] == 0.02
    assert plan["terminal_gate"]["maximum_terminal_validation_ce"] == 5.3117
    assert plan["decision_rules"]["automatic_5tpp"] is False
    assert plan["decision_rules"]["mapping_network_200x_claim"] is False
    assert plan["mfu_result"]["mfu_fraction"] >= 0.20
    assert plan["authorization"]["exact_config_mfu_passed"] is True
    assert plan["authorization"]["authorized_training_consumed"] is True
