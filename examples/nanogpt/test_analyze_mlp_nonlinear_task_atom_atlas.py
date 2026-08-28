from examples.nanogpt.analyze_mlp_nonlinear_task_atom_atlas import (
    checkpoint_accounting,
    feature_parameters,
    self_test,
)


def test_nonlinear_atlas_schedule() -> None:
    assert feature_parameters(1) == (0.5, -1.0)
    assert feature_parameters(4) == (4.0, -1.0)
    assert feature_parameters(5) == (0.5, -0.5)
    assert feature_parameters(21) == (0.5, -1.0)


def test_nonlinear_atlas_accounting() -> None:
    accounting = checkpoint_accounting()
    assert accounting["total_state_scalars"] == 566_016
    assert accounting["fp16_checkpoint_bytes"] == 1_132_032
    assert accounting["state_fraction"] < 0.01


def test_nonlinear_atlas_self_test() -> None:
    result = self_test("cpu")
    assert result["status"] == "passed"
