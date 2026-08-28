from examples.nanogpt.analyze_mlp_role_wide_givens_atlas import (
    checkpoint_accounting,
    self_test,
)


def test_wide_atlas_accounting() -> None:
    accounting = checkpoint_accounting()
    assert accounting["total_state_scalars"] == 566_016
    assert accounting["fp16_checkpoint_bytes"] == 1_132_032
    assert accounting["state_fraction"] < 0.01


def test_wide_atlas_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
