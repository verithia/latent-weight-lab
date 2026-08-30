import torch

from examples.nanogpt.analyze_mlp_eight_binary_global_frame_dct_capacity import (
    deployment_accounting,
    self_test,
)


def test_h52_accounting() -> None:
    accounting = deployment_accounting()
    assert accounting["total_checkpoint_bytes"] == 1_032_192
    assert accounting["checkpoint_byte_fraction"] == 0.009114583333333334
    assert accounting["continuous_coordinate_fraction"] == 0.00390625
    assert accounting["persistent_pca_or_carrier_values"] == 0


def test_h52_binary_codec_and_ste() -> None:
    result = self_test("cuda" if torch.cuda.is_available() else "cpu")
    assert result["status"] == "passed"
