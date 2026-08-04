from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_hybrid_endpoint_interpolation import (
    INTERIOR_ALPHAS,
    classify_endpoint_result,
    family_tensor_names,
    install_variant,
    interpolation_variant,
    validate_state_topology,
)


def plan() -> dict:
    return {
        "evaluation": {"independent_window_seeds": [11, 22]},
        "interpolation": {"layers": [0]},
    }


def rows(
    *,
    parent: tuple[float, float],
    hybrid: tuple[float, float],
    transplanted: tuple[float, float],
    interiors: dict[float, tuple[float, float]] | None = None,
    parent_endpoint_in_parent_context: tuple[float, float] | None = None,
) -> list[dict]:
    values = []
    interiors = interiors or {
        0.25: hybrid,
        0.50: hybrid,
        0.75: hybrid,
    }
    parent_endpoint_in_parent_context = (
        transplanted
        if parent_endpoint_in_parent_context is None
        else parent_endpoint_in_parent_context
    )
    for index, seed in enumerate((11, 22)):
        h_values = {
            0.0: hybrid[index],
            0.25: interiors[0.25][index],
            0.50: interiors[0.50][index],
            0.75: interiors[0.75][index],
            1.0: transplanted[index],
        }
        p_values = {
            0.0: parent[index],
            0.25: parent[index],
            0.50: parent[index],
            0.75: parent[index],
            1.0: parent_endpoint_in_parent_context[index],
        }
        for context, mapping in (("hybrid", h_values), ("parent", p_values)):
            for alpha, ce in mapping.items():
                values.append(
                    {
                        "window_seed": seed,
                        "variant": interpolation_variant(context, alpha),
                        "val_ce": ce,
                    }
                )
    return values


def test_direction_classification() -> None:
    result = classify_endpoint_result(
        rows(
            parent=(5.50, 5.51),
            hybrid=(5.60, 5.61),
            transplanted=(5.54, 5.55),
            interiors={
                0.25: (5.58, 5.59),
                0.50: (5.56, 5.57),
                0.75: (5.55, 5.56),
            },
        ),
        plan(),
    )
    assert result["primary_classification"] == "HYBRID_CPROJ_ENDPOINT_DIRECTION_DOMINATES"


def test_wider_coadaptation_classification() -> None:
    result = classify_endpoint_result(
        rows(
            parent=(5.50, 5.51),
            hybrid=(5.60, 5.61),
            transplanted=(5.59, 5.60),
            interiors={
                0.25: (5.595, 5.605),
                0.50: (5.594, 5.604),
                0.75: (5.593, 5.603),
            },
        ),
        plan(),
    )
    assert result["primary_classification"] == "WIDER_BLOCK_COADAPTATION_DOMINATES"


def test_mixed_classification() -> None:
    result = classify_endpoint_result(
        rows(
            parent=(5.50, 5.51),
            hybrid=(5.60, 5.61),
            transplanted=(5.57, 5.58),
            interiors={
                0.25: (5.59, 5.60),
                0.50: (5.58, 5.59),
                0.75: (5.575, 5.585),
            },
        ),
        plan(),
    )
    assert result["primary_classification"] == "MIXED_ENDPOINT_DIRECTION_AND_COADAPTATION"


def test_common_interior_barrier_requires_both_windows() -> None:
    data = rows(
        parent=(5.50, 5.50),
        hybrid=(5.60, 5.60),
        transplanted=(5.50, 5.50),
        interiors={
            0.25: (5.585, 5.585),
            0.50: (5.57, 5.57),
            0.75: (5.54, 5.54),
        },
    )
    result = classify_endpoint_result(data, plan())
    assert result["functional_barrier_classification"] == "INTERIOR_FUNCTIONAL_BARRIER"


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_2 = torch.nn.LayerNorm(2)
        self.mlp = torch.nn.Module()
        self.mlp.c_fc = torch.nn.Linear(2, 3, bias=False)
        self.mlp.c_proj = torch.nn.Linear(3, 2, bias=False)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([TinyBlock()])


def test_install_variant_restores_context_and_interpolates_only_cproj() -> None:
    model = TinyModel()
    base = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    other = {name: tensor.detach().clone() + 2.0 for name, tensor in base.items()}
    model.transformer.h[0].mlp.c_fc.weight.data.fill_(99.0)
    install_variant(
        model,
        base_state=base,
        other_state=other,
        spec={
            "variant": "hybrid_context_all_cproj_alpha0p50",
            "context": "hybrid",
            "kind": "all_cproj_interpolation",
            "alpha": 0.5,
            "layers": [0],
        },
        all_layers=[0],
    )
    state = model.state_dict()
    cproj = "transformer.h.0.mlp.c_proj.weight"
    cfc = "transformer.h.0.mlp.c_fc.weight"
    assert torch.allclose(state[cproj], base[cproj] + 1.0)
    assert torch.equal(state[cfc], base[cfc])


def test_wider_ln2_variant_transplants_all_registered_families() -> None:
    model = TinyModel()
    base = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    other = {name: tensor.detach().clone() + 3.0 for name, tensor in base.items()}
    install_variant(
        model,
        base_state=base,
        other_state=other,
        spec={
            "variant": "parent_all_cproj_plus_all_cfc_plus_all_ln2",
            "context": "hybrid",
            "kind": "wider_context_transplant",
            "alpha": 1.0,
            "layers": [0],
        },
        all_layers=[0],
    )
    state = model.state_dict()
    for family in ("cproj", "cfc", "ln2"):
        for name in family_tensor_names(base, family, [0]):
            assert torch.equal(state[name], other[name])


def test_state_topology_fails_closed() -> None:
    parent = {"x": torch.zeros(2, 2)}
    validate_state_topology(parent, copy.deepcopy(parent))
    with pytest.raises(ValueError, match="topology differs"):
        validate_state_topology(parent, {"y": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="shape differs"):
        validate_state_topology(parent, {"x": torch.zeros(3, 2)})


@pytest.mark.parametrize("alpha", INTERIOR_ALPHAS)
def test_variant_names_are_stable(alpha: float) -> None:
    assert interpolation_variant("hybrid", alpha).endswith(
        f"alpha{alpha:.2f}".replace(".", "p")
    )
