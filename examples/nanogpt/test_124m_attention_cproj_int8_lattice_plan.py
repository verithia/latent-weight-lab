import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_cproj_int8_lattice_0p5tpp_plan.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_pins_the_selected_codec_and_blocks_training_before_mfu() -> None:
    plan = json.loads(PLAN.read_text())
    config_path = ROOT / plan["candidate"]["config"]
    config = json.loads(config_path.read_text())
    assert sha256(config_path) == plan["candidate"]["config_sha256"]
    assert config["block_fht_attn_cproj_int8_lattice"] is True
    assert config["block_fht_attn_cproj_int8_lattice_block_size"] == 4096
    assert config["checkpoint_wall_clock_seconds"] == 7200
    assert plan["candidate"]["storage_ratio"] == 0.2501220703125
    assert plan["authorization"]["training_before_mfu_pass"] is False
    assert plan["authorization"]["larger_model_or_token_rung"] is False
    oracle = ROOT / plan["identity"]["integer_lattice_gate_result"]
    assert sha256(oracle) == plan["identity"]["integer_lattice_gate_result_sha256"]
    rejected_launch = ROOT / plan["identity"]["scientific_launch_reject_result"]
    assert sha256(rejected_launch) == plan["identity"]["scientific_launch_reject_result_sha256"]
