from examples.nanogpt.analyze_mlp_sign_private_microcore_union import (
    checkpoint_accounting,
    self_test,
)


def test_sign_private_union_accounting() -> None:
    accounting = checkpoint_accounting()
    assert accounting["total_compact_checkpoint_bytes"] == 1_132_032
    assert accounting["global_bitplane_bytes"] == 294_912
    assert accounting["private_core_scalars"] == 98_304
    assert accounting["checkpoint_byte_fraction"] < 0.01


def test_sign_private_union_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
    assert result["finite_nonzero_private_core_gradient"] is True
