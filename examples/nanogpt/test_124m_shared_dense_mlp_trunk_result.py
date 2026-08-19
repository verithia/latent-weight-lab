import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_shared_dense_mlp_trunk_0p5tpp.json"
RESULT = REPO / "examples/nanogpt/configs/selection_artifacts/124m_shared_dense_mlp_trunk_0p5tpp_result.json"


def test_terminal_result_is_exactly_bound_and_rejected() -> None:
    result = json.loads(RESULT.read_text())
    assert result["identity"]["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert result["classification"] == "REJECT_SHARED_DENSE_MLP_TRUNK_RANK1_LAYER_AXIS_LOSS"
    assert result["execution"]["terminal_status"] == "clean"
    assert result["execution"]["exit_code"] == 0
    assert result["decision"]["mfu_gate_passed"] is True
    assert result["decision"]["loss_gate_passed"] is False
    assert result["decision"]["larger_rung_authorized"] is False


def test_loss_and_perplexity_gaps_are_explicit() -> None:
    loss = json.loads(RESULT.read_text())["fixed_window_loss"]
    assert math.isclose(loss["ce_gap_to_attention_only"], loss["terminal_validation_ce"] - loss["attention_only_control_validation_ce"])
    assert math.isclose(loss["perplexity_ratio_to_attention_only"], math.exp(loss["ce_gap_to_attention_only"]), rel_tol=1e-9)
    assert math.isclose(loss["ce_improvement_over_paired_monarch"], loss["paired_monarch_validation_ce"] - loss["terminal_validation_ce"])
    assert loss["threshold_miss_ce"] > 0.1


def test_parameter_and_system_tradeoff_are_complete() -> None:
    result = json.loads(RESULT.read_text())
    accounting = result["parameter_accounting"]
    benchmark = result["exact_config_benchmark"]
    assert accounting["mlp_parameter_compression_ratio"] > 11
    assert accounting["mlp_parameter_saving_fraction"] > 0.91
    assert accounting["generated_weight_expansion_required"] is False
    assert benchmark["mfu_fraction"] >= 0.20
    assert benchmark["tokens_per_second"] > 200_000
    assert benchmark["peak_allocated_mib"] > 0
    assert benchmark["relative_to_full_mlp_error_feedback"]["throughput_fraction_change"] > 0.3
