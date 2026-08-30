import torch

from examples.nanogpt.analyze_mlp_lowbit_global_frame_dct_carrier_capacity import (
    deployment_accounting,
    self_test,
)


def test_h51_accounting() -> None:
    accounting = deployment_accounting()
    assert accounting["total_checkpoint_bytes"] == 1_108_992
    assert accounting["dense_replaced_mlp_fp16_bytes"] == 113_246_208
    assert accounting["checkpoint_byte_fraction"] < 0.01
    assert accounting["persistent_pca_or_carrier_values"] == 0


def test_h51_dct_carrier_and_int4_self_test() -> None:
    result = self_test("cuda" if torch.cuda.is_available() else "cpu")
    assert result["status"] == "passed"
    assert result["dct_orthogonality_max_error"] < 2e-5
    assert abs(result["four_block_norm_ratio"] - 4.0) < 2e-4
