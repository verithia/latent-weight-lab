from examples.nanogpt.analyze_mlp_shared_fullcoverage_givens_capacity import (
    givens_accounting,
    self_test,
)


def test_givens_accounting() -> None:
    accounting = givens_accounting()
    assert accounting["total_state_scalars"] == 565_272
    assert accounting["fp16_checkpoint_bytes"] == 1_130_544
    assert accounting["state_fraction"] < 0.01


def test_fullcoverage_givens_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
