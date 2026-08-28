from examples.nanogpt.analyze_mlp_shared_fullcoverage_gl2_capacity import (
    gl2_accounting,
    self_test,
)


def test_gl2_accounting() -> None:
    accounting = gl2_accounting()
    assert accounting["total_state_scalars"] == 564_504
    assert accounting["fp16_checkpoint_bytes"] == 1_129_008
    assert accounting["state_fraction"] < 0.01


def test_fullcoverage_gl2_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
