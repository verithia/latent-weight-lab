import torch

from examples.nanogpt.analyze_mlp_global_chord_binary_schur_capacity import (
    deployment_accounting,
    self_test,
)


def test_h53_accounting() -> None:
    accounting = deployment_accounting()
    assert accounting["total_checkpoint_bytes"] == 1_104_720
    assert accounting["checkpoint_byte_fraction"] == 0.009755028618706597
    assert accounting["persistent_pca_or_dense_basis_values"] == 0


def test_h53_schur_rank_and_own_family() -> None:
    result = self_test("cuda" if torch.cuda.is_available() else "cpu")
    assert result["status"] == "passed"
    assert result["synthetic_minimum_capture"] >= 0.999999
    assert result["factor_rank"] < result["observed_schur_rank"]
    assert result["observed_schur_rank"] <= result["schur_rank_bound"]
