import torch

from examples.nanogpt.analyze_mlp_cfc_functional_shear_pareto import (
    replay_blended_recipes,
    symmetric_recipe_to_pair_recipe,
)
from examples.nanogpt.analyze_mlp_cfc_task_shear_fit import apply_pair_stage


def test_blended_recipe_replays_exact_endpoints() -> None:
    source = torch.eye(2)
    pairs = torch.tensor([[0, 1]])
    weight_coordinates = torch.tensor([[0.01, 0.0]], dtype=torch.float64)
    functional_coordinates = torch.tensor([[0.03, 0.0]], dtype=torch.float64)
    weight_recipe = [(pairs, weight_coordinates)]
    functional_recipe = [(pairs, functional_coordinates)]
    weight_expected, _ = apply_pair_stage(source, pairs, weight_coordinates)
    function_expected, _ = apply_pair_stage(source, pairs, functional_coordinates)
    weight_update, _ = replay_blended_recipes(
        source, weight_recipe, functional_recipe, beta=0.0
    )
    function_update, _ = replay_blended_recipes(
        source, weight_recipe, functional_recipe, beta=1.0
    )
    torch.testing.assert_close(weight_update, weight_expected - source)
    torch.testing.assert_close(function_update, function_expected - source)


def test_blended_recipe_interpolates_coordinates_not_updates() -> None:
    source = torch.eye(2)
    pairs = torch.tensor([[0, 1]])
    left = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    right = torch.tensor([[0.04, 0.0]], dtype=torch.float64)
    expected, _ = apply_pair_stage(
        source,
        pairs,
        torch.tensor([[0.01, 0.0]], dtype=torch.float64),
    )
    actual, finite = replay_blended_recipes(
        source, [(pairs, left)], [(pairs, right)], beta=0.25
    )
    torch.testing.assert_close(actual, expected - source)
    assert finite["maximum_condition_number"] < 1.1


def test_production_symmetric_recipe_adapts_to_pair_coordinates() -> None:
    pairs = torch.tensor([[0, 1], [2, 3]])
    coordinates = torch.tensor([0.01, -0.02], dtype=torch.float64)
    adapted = symmetric_recipe_to_pair_recipe([(pairs, coordinates)])
    assert torch.equal(adapted[0][0], pairs)
    assert adapted[0][1].shape == (2, 2)
    assert torch.equal(adapted[0][1][:, 0], coordinates)
    assert torch.equal(adapted[0][1][:, 1], torch.zeros_like(coordinates))
