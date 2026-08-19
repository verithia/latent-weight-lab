import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_fullmlp_paired_monarch16_0p5tpp.json"
RESULT = REPO / "examples/nanogpt/configs/selection_artifacts/124m_paired_monarch_full_replacement_0p5tpp_result.json"


def test_terminal_result_is_exactly_bound_and_rejected() -> None:
    result = json.loads(RESULT.read_text())
    assert result["identity"]["config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    assert result["classification"] == (
        "REJECT_PAIRED_MONARCH_FULL_REPLACEMENT_LM_LOSS"
    )
    assert result["execution"]["terminal_status"] == "clean"
    assert result["execution"]["exit_code"] == 0
    assert result["decision"]["mfu_gate_passed"] is True
    assert result["decision"]["loss_gate_passed"] is False
    assert result["decision"]["larger_rung_authorized"] is False


def test_loss_and_perplexity_gaps_are_not_hidden() -> None:
    result = json.loads(RESULT.read_text())
    loss = result["fixed_window_loss"]
    assert math.isclose(
        loss["ce_gap_to_attention_only"],
        loss["terminal_validation_ce"]
        - loss["attention_only_control_validation_ce"],
    )
    assert math.isclose(
        loss["perplexity_ratio_to_attention_only"],
        math.exp(loss["ce_gap_to_attention_only"]),
        rel_tol=1e-9,
    )
    assert math.isclose(
        loss["perplexity_ratio_to_full_mlp_error_feedback"],
        math.exp(loss["ce_gap_to_full_mlp_error_feedback"]),
        rel_tol=1e-9,
    )
    assert loss["threshold_miss_ce"] > 0.3


def test_benchmark_and_parameter_tradeoff_are_complete() -> None:
    result = json.loads(RESULT.read_text())
    benchmark = result["exact_config_benchmark"]
    accounting = result["parameter_accounting"]
    assert benchmark["mfu_fraction"] >= 0.20
    assert benchmark["tokens_per_second"] > 0
    assert benchmark["peak_allocated_mib"] > 0
    assert benchmark["relative_to_full_mlp_error_feedback"][
        "throughput_fraction_change"
    ] > 0.4
    assert accounting["mlp_parameter_compression_ratio"] > 32
    assert accounting["dense_temporal_error_feedback_state"] is False
