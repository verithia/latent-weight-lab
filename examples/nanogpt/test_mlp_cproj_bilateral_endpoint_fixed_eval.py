from __future__ import annotations

from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    select_variant,
)
from examples.nanogpt import analyze_mlp_cproj_teacher_forced_bilateral_full_carry as replay


def _rows(output32_train: float, output32_val: float) -> list[dict[str, object]]:
    return [
        {"variant": "dense_endpoint", "train_ce": 5.40, "val_ce": 5.41},
        {"variant": "hidden88_full_carry", "train_ce": 5.43, "val_ce": 5.44},
        {
            "variant": "hidden88_output32_full_carry",
            "train_ce": output32_train,
            "val_ce": output32_val,
        },
        {
            "variant": "hidden88_output64_full_carry",
            "train_ce": 5.42,
            "val_ce": 5.43,
        },
    ]


def test_selects_smallest_candidate_meeting_fixed_ce_gate() -> None:
    result = select_variant(
        _rows(5.429, 5.437),
        minimum_val_gain=0.002,
        minimum_train_gain=0.0,
    )
    assert result["selected_variant"] == "hidden88_output32_full_carry"
    assert result["production_implementation_authorized"] is True
    assert result["language_model_training_authorized"] is False


def test_train_regression_rejects_smaller_candidate_and_falls_through() -> None:
    result = select_variant(
        _rows(5.431, 5.437),
        minimum_val_gain=0.002,
        minimum_train_gain=0.0,
    )
    assert result["comparisons"]["hidden88_output32_full_carry"]["passed"] is False
    assert result["selected_variant"] == "hidden88_output64_full_carry"


def test_no_candidate_defaults_to_right_only_without_training_authority() -> None:
    rows = _rows(5.431, 5.439)
    rows[-1]["train_ce"] = 5.431
    rows[-1]["val_ce"] = 5.439
    result = select_variant(
        rows,
        minimum_val_gain=0.002,
        minimum_train_gain=0.0,
    )
    assert result["selected_variant"] == "hidden88_full_carry"
    assert result["production_implementation_authorized"] is False
    assert result["language_model_training_authorized"] is False


def test_production_parent_selector_does_not_change_later_angle_targets(
    monkeypatch,
) -> None:
    import torch

    calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def fake_fit_right_pass(
        weight,
        target_update,
        *,
        matching_target_update=None,
        stages,
        neighbors,
        seed,
    ):
        del stages, neighbors, seed
        calls.append(
            (
                target_update.clone(),
                None
                if matching_target_update is None
                else matching_target_update.clone(),
            )
        )
        return weight.clone()

    monkeypatch.setattr(replay, "fit_right_pass", fake_fit_right_pass)
    weight = torch.zeros(4, 4)
    requested = torch.ones_like(weight)
    feedback = torch.full_like(weight, 2.0)
    parent_connectivity = torch.full_like(weight, 5.0)
    replay.structured_step(
        weight,
        requested,
        feedback,
        parent_connectivity_update=parent_connectivity,
        output_stages=2,
        learning_rate=0.1,
        weight_decay=0.0,
        neighbors=2,
        seed=7,
    )
    assert len(calls) == 3
    assert torch.equal(calls[0][0], requested + feedback)
    assert torch.equal(calls[0][1], parent_connectivity)
    assert torch.equal(calls[1][0], requested + feedback)
    assert calls[1][1] is None
    assert torch.equal(calls[2][0], (requested + feedback).T)
    assert calls[2][1] is None
