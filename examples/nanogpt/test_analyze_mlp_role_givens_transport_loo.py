from examples.nanogpt.analyze_mlp_role_givens_transport_loo import (
    checkpoint_accounting,
    self_test,
)


def test_role_givens_accounting() -> None:
    accounting = checkpoint_accounting()
    assert accounting["total_state_scalars"] == 566_040
    assert accounting["fp16_checkpoint_bytes"] == 1_132_080
    assert accounting["state_fraction"] < 0.01


def test_role_givens_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
