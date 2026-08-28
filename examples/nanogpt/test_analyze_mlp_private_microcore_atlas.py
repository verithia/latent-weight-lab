from examples.nanogpt.analyze_mlp_private_microcore_atlas_helpers import (
    checkpoint_accounting,
    self_test,
)


def test_private_microcore_accounting() -> None:
    accounting = checkpoint_accounting()
    assert accounting["total_state_scalars"] == 541_440
    assert accounting["fp16_checkpoint_bytes"] == 1_082_880
    assert accounting["state_fraction"] < 0.01


def test_private_microcore_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
    assert result["finite_nonzero_core_gradient"] is True
