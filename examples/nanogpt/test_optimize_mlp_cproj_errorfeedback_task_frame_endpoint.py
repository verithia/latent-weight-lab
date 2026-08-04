from __future__ import annotations

import torch

from examples.nanogpt.optimize_mlp_cproj_errorfeedback_task_frame_endpoint import (
    capture_frame_state,
    evaluate_without_frames,
    frame_parameters,
    restore_frame_state,
    select_decision,
    set_variant,
)
from examples.nanogpt.test_materialized_muon_cproj_chart_cache import make_model


def synthetic_rows(full: float, post: float, pre: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, offset in (("primary", 0.0), ("confirmation", 0.01)):
        for update in (0, 120):
            values = (
                {
                    "full_coupled": 5.0,
                    "post_only": 5.0,
                    "pre_only": 5.0,
                    "identity": 5.0,
                }
                if update == 0
                else {
                    "full_coupled": 5.0 - full,
                    "post_only": 5.0 - post,
                    "pre_only": 5.0 - pre,
                    "identity": 5.0,
                }
            )
            for variant, ce in values.items():
                rows.append(
                    {
                        "split": split,
                        "variant": variant,
                        "update": update,
                        "ce": ce + offset,
                    }
                )
    return rows


def test_selection_distinguishes_coupled_and_post_only_capacity() -> None:
    coupled = select_decision(synthetic_rows(0.010, 0.006, 0.001), 0.0075, 0.005, 0.0025)
    assert coupled["decision"] == "COUPLED_ACTIVATION_FRAME_CAPACITY"
    post_only = select_decision(synthetic_rows(0.006, 0.006, 0.001), 0.0075, 0.005, 0.0025)
    assert post_only["decision"] == "POST_CPROJ_FRAME_ONLY_CAPACITY"
    rejected = select_decision(synthetic_rows(0.004, 0.003, 0.001), 0.0075, 0.005, 0.0025)
    assert rejected["decision"] == "LOCAL_FRAME_CAPACITY_INSUFFICIENT"


def test_variant_state_round_trip_is_exact() -> None:
    torch.manual_seed(23)
    model = make_model()
    mlp = model.transformer.h[0].mlp
    assert mlp.pregelu_block_rotation is not None
    parameters = frame_parameters(model)
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.normal_(std=0.02)
    state = capture_frame_state(model)
    set_variant(model, state, "identity")
    assert all(torch.count_nonzero(parameter) == 0 for parameter in parameters.values())
    restore_frame_state(model, state)
    assert all(
        torch.equal(parameter.detach().cpu(), state[key])
        for key, parameter in parameters.items()
    )


def test_frame_disabled_evaluation_restores_every_module() -> None:
    model = make_model().eval()
    mlp = model.transformer.h[0].mlp
    before = (
        mlp.pregelu_block_rotation,
        mlp.hidden_block_rotation,
        mlp.output_block_rotation,
    )
    tokens = torch.randint(0, 32, (2, 9))
    ce = evaluate_without_frames(model, [tokens], "cpu")
    assert torch.isfinite(torch.tensor(ce))
    assert (
        mlp.pregelu_block_rotation,
        mlp.hidden_block_rotation,
        mlp.output_block_rotation,
    ) == before
