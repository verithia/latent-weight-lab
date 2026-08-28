from examples.nanogpt.analyze_mlp_task_modulated_microcore_atlas import (
    checkpoint_accounting,
    self_test,
)


def test_microcore_atlas_accounting() -> None:
    accounting = checkpoint_accounting()
    assert accounting["total_state_scalars"] == 525_056
    assert accounting["fp16_checkpoint_bytes"] == 1_050_112
    assert accounting["state_fraction"] < 0.01


def test_microcore_atlas_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
