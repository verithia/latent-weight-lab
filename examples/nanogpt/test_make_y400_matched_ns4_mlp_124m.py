from __future__ import annotations

from examples.nanogpt.make_y400_matched_ns4_mlp_124m import (
    compact,
    dense,
    native,
)


def test_native_control_inherits_the_default_ns5_path() -> None:
    config = native()
    assert config["muon_ns_steps"] == 5
    assert config["muon_mlp_ns_steps"] == 0
    assert config["muon_mlp_lr_scale"] == 1.0
    assert config["launch_ready"] is False


def test_dense_gate_passed_under_the_selected_optimizer() -> None:
    config = dense()
    assert config["muon_ns_steps"] == 5
    assert config["muon_mlp_ns_steps"] == 4
    assert config["muon_mlp_lr_scale"] == 1.225
    assert config["launch_ready"] is True
    assert config["endpoint_gate"]["terminal_validation_ce_max"] == 5.371


def test_compact_gate_is_authorized_by_the_sealed_dense_result() -> None:
    config = compact()
    assert config["muon_mlp_ns_steps"] == 4
    assert config["muon_mlp_lr_scale"] == 1.225
    assert config["launch_ready"] is True
    assert config["launch_block_reason"] is None
    assert config["authorization_result"].endswith(
        "124m_pair_vq_matched_ns4_dense_result.json"
    )
    assert len(config["authorization_result_sha256"]) == 64
    assert config["endpoint_gate"]["matched_dense_terminal_validation_ce"] == 5.3663
    assert config["endpoint_gate"]["candidate_minus_matched_dense_validation_ce_max"] == 0.01
    assert config["endpoint_gate"]["terminal_candidate_validation_ce_max"] == 5.381
