from examples.nanogpt.analyze_mlp_dense_shared_relation_control import (
    control_accounting,
    self_test,
)


def test_dense_shared_relation_control_accounting() -> None:
    accounting = control_accounting()
    assert accounting["full_control_state_scalars"] == 10_315_032
    assert accounting["fp16_checkpoint_bytes"] == 20_630_064
    assert accounting["state_fraction"] > 0.18


def test_dense_shared_relation_control_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
