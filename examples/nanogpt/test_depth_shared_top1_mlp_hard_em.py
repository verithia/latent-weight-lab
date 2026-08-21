from __future__ import annotations

import torch

from examples.nanogpt.analyze_depth_shared_top1_mlp_hard_em import (
    assignment_costs,
    assignment_summary,
    expert_output,
    oracle_assignments,
    selected_prediction,
)
from examples.nanogpt.test_depth_shared_top1_mlp_teacher_fit import family


def toy_data() -> dict[str, torch.Tensor]:
    module = family()
    clean = torch.randn(module.layers, 2, 5, 4)
    delta = torch.randn_like(clean) * 0.01
    variants = torch.stack((clean, clean + delta, clean - delta))
    targets = torch.empty(3, module.layers, 2, 5, 4)
    with torch.no_grad():
        for layer in range(module.layers):
            targets[:, layer] = expert_output(module, layer, layer, variants[:, layer])
    return {"clean": clean, "variants": variants, "targets": targets}


def test_oracle_recovers_exact_expert_assignment() -> None:
    module = family()
    data = toy_data()
    assignments = oracle_assignments(module, data)
    assert torch.equal(assignments[0], torch.zeros_like(assignments[0]))
    assert torch.equal(assignments[1], torch.ones_like(assignments[1]))
    assert torch.all(assignment_costs(module, data).amin(dim=-1) < 1e-8)


def test_selected_prediction_matches_direct_expert() -> None:
    module = family()
    data = toy_data()
    assignments = oracle_assignments(module, data)
    layers = torch.arange(module.layers)
    rows = torch.arange(data["clean"].shape[2])
    prediction = selected_prediction(
        module, data["variants"], layers, assignments, rows
    )
    torch.testing.assert_close(prediction, data["targets"])


def test_assignment_summary_tracks_changes() -> None:
    first = torch.tensor([[[0, 1], [2, 3]]])
    second = torch.tensor([[[0, 2], [2, 1]]])
    row = assignment_summary(second, previous=first)
    assert row["counts"] == [1, 1, 2, 0]
    assert row["change_fraction"] == 0.5
