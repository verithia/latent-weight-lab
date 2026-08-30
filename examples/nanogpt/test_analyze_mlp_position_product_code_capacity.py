import torch

from examples.nanogpt.analyze_mlp_position_product_code_capacity import (
    deployment_accounting,
    pack_unsigned_codes,
    self_test,
    unpack_unsigned_codes,
)


def test_h55_accounting() -> None:
    accounting = deployment_accounting()
    assert accounting["total_checkpoint_bytes"] == 1_119_744
    assert accounting["checkpoint_byte_fraction"] == 0.0098876953125
    assert accounting["persistent_pca_or_per_node_basis_values"] == 0


def test_h55_five_bit_packing() -> None:
    codes = torch.arange(0, 32, dtype=torch.int64).repeat(7)
    packed = pack_unsigned_codes(codes, bits=5)
    recovered = unpack_unsigned_codes(packed, values=codes.numel(), bits=5)
    assert torch.equal(recovered, codes)
    assert packed.numel() == codes.numel() * 5 // 8


def test_h55_product_codec_own_family() -> None:
    result = self_test("cuda" if torch.cuda.is_available() else "cpu")
    assert result["status"] == "passed"
    assert result["synthetic_own_family_capture"] >= 0.999999
    assert result["packed_roundtrip"]
