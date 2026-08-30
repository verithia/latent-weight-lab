import torch

from examples.nanogpt.analyze_mlp_shared_int4_row_dictionary_capacity import (
    deployment_accounting,
    self_test,
)


def test_h54_accounting() -> None:
    accounting = deployment_accounting()
    assert accounting["total_checkpoint_bytes"] == 1_114_112
    assert accounting["checkpoint_byte_fraction"] == 0.009837962962962963
    assert accounting["persistent_pca_or_per_node_basis_values"] == 0


def test_h54_sparse_codec_and_own_family() -> None:
    result = self_test("cuda" if torch.cuda.is_available() else "cpu")
    assert result["status"] == "passed"
    assert result["synthetic_own_family_capture"] >= 0.999999
    assert result["synthetic_hard_reencode_capture"] > 0.0
