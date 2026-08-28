from examples.nanogpt.analyze_mlp_role_sign_carrier_loo import (
    checkpoint_accounting,
    self_test,
)


def test_role_sign_carrier_accounting() -> None:
    accounting = checkpoint_accounting()
    assert accounting["total_compact_checkpoint_bytes"] == 1_132_080
    assert accounting["fp16_equivalent_scalars"] == 566_040
    assert accounting["checkpoint_byte_fraction"] < 0.01


def test_role_sign_carrier_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
