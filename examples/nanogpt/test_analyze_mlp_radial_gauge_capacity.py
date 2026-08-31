import torch

from examples.nanogpt.analyze_mlp_radial_gauge_capacity import (
    deployment_accounting,
    radial_activation,
    self_test,
)


def test_h56_accounting() -> None:
    accounting = deployment_accounting()
    assert accounting["total_checkpoint_bytes"] == 1_106_496
    assert accounting["checkpoint_byte_fraction"] == 0.009770711263020834
    assert accounting["persistent_pca_or_per_node_basis_values"] == 0


def test_h56_radial_equivariance_and_procrustes() -> None:
    result = self_test("cuda" if torch.cuda.is_available() else "cpu")
    assert result["status"] == "passed"
    assert result["procrustes_recovery_relative_error_sum"] <= 1e-4
    assert result["radial_equivariance_relative_error"] <= 1e-5
    assert result["packed_roundtrip"]


def test_h56_radial_shape() -> None:
    value = torch.randn(3, 5, 32)
    assert radial_activation(value).shape == value.shape
