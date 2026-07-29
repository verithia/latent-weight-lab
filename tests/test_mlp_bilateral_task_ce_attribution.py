import torch

from examples.nanogpt.analyze_mlp_bilateral_task_ce_attribution import (
    attribution_decision,
    combine_states,
    variant_specs,
)


def test_combine_states_selects_exact_group_and_layer() -> None:
    identity = {
        "layer.0.hidden_gain": torch.tensor([0.0]),
        "layer.0.output_gain": torch.tensor([0.0]),
        "layer.1.hidden_gain": torch.tensor([0.0]),
        "layer.1.output_gain": torch.tensor([0.0]),
    }
    accepted = {key: value + 1.0 for key, value in identity.items()}
    combined = combine_states(
        identity,
        accepted,
        lambda layer, group: layer == 1 and group == "output_gain",
    )
    assert float(combined["layer.1.output_gain"]) == 1.0
    assert sum(float(value) for value in combined.values()) == 1.0


def test_variant_specs_are_unique_and_complete() -> None:
    names = [name for name, _ in variant_specs()]
    assert len(names) == len(set(names))
    assert len(names) == 27
    assert "accepted_all" in names
    assert "only_output_rotation" in names
    assert "without_output_gain" in names
    assert "only_depth_late" in names
    assert "only_layer_11" in names


def make_rows() -> list[dict[str, object]]:
    names = [name for name, _ in variant_specs()]
    return [
        {
            "variant": name,
            "primary_gain": 0.0,
            "confirmation_gain": 0.0,
        }
        for name in names
    ]


def test_attribution_decision_selects_output_frame() -> None:
    rows = make_rows()
    by_name = {str(row["variant"]): row for row in rows}
    by_name["accepted_all"].update(
        primary_gain=0.01, confirmation_gain=0.01
    )
    by_name["only_output_rotation"].update(
        primary_gain=0.006, confirmation_gain=0.007
    )
    for group in ("hidden_rotation", "hidden_gain"):
        by_name[f"without_{group}"].update(
            primary_gain=0.009, confirmation_gain=0.009
        )
    decision = attribution_decision(rows, 0.005)
    assert decision["output_frame_supported"] is True
    assert decision["hidden_frame_supported"] is False
    assert decision["next_structure"] == (
        "TEST_NARROW_RESIDUAL_MLP_OUTPUT_FRAME"
    )


def test_attribution_decision_identifies_distributed_chart() -> None:
    rows = make_rows()
    by_name = {str(row["variant"]): row for row in rows}
    by_name["accepted_all"].update(
        primary_gain=0.01, confirmation_gain=0.01
    )
    for group in (
        "hidden_rotation",
        "hidden_gain",
        "output_rotation",
        "output_gain",
    ):
        by_name[f"without_{group}"].update(
            primary_gain=0.007, confirmation_gain=0.007
        )
    decision = attribution_decision(rows, 0.005)
    assert decision["distributed_chart"] is True
    assert decision["output_frame_supported"] is False
    assert decision["hidden_frame_supported"] is False
    assert decision["next_structure"] == (
        "RETAIN_DISTRIBUTED_LATE_CHART_CONTROL"
    )
