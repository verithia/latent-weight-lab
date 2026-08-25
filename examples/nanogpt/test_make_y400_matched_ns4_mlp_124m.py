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


def test_dense_gate_is_the_only_launch_ready_scientific_config() -> None:
    config = dense()
    assert config["muon_ns_steps"] == 5
    assert config["muon_mlp_ns_steps"] == 4
    assert config["muon_mlp_lr_scale"] == 1.225
    assert config["launch_ready"] is True
    assert config["endpoint_gate"]["terminal_validation_ce_max"] == 5.371


def test_compact_gate_is_blocked_on_the_dense_result() -> None:
    config = compact()
    assert config["muon_mlp_ns_steps"] == 4
    assert config["muon_mlp_lr_scale"] == 1.225
    assert config["launch_ready"] is False
    assert config["endpoint_gate"]["candidate_minus_matched_dense_validation_ce_max"] == 0.01
    assert config["endpoint_gate"]["terminal_candidate_validation_ce_max"] == 5.381
