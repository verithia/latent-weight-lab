from examples.nanogpt.analyze_mlp_shared_bilateral_transport_capacity import (
    self_test,
    transport_accounting,
)


def test_transport_accounting() -> None:
    accounting = transport_accounting()
    assert accounting["total_state_scalars"] == 564_528
    assert accounting["fp16_checkpoint_bytes"] == 1_129_056
    assert accounting["state_fraction"] < 0.01


def test_bilateral_fit_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
