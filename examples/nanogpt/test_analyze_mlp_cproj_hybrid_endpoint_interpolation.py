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
    validate_model_configs,
    validate_state_topology,
)
from examples.nanogpt.model import GPTConfig
from examples.nanogpt.muon_matched_givens import MuonMatchedGivensLinear


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
    hybrid = copy.deepcopy(parent)
    for layer in range(12):
        prefix = f"transformer.h.{layer}.mlp.c_proj."
        hybrid[prefix + "output_last_angles"] = torch.zeros(1)
        hybrid[prefix + "output_selected_permutations"] = torch.zeros(1)
        hybrid[prefix + "output_selected_inverse_permutations"] = torch.zeros(1)
    assert len(validate_state_topology(parent, hybrid)) == 36
    with pytest.raises(ValueError, match="topology differs"):
        validate_state_topology(parent, {"y": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="shape differs"):
        malformed = copy.deepcopy(hybrid)
        malformed["x"] = torch.zeros(3, 2)
        validate_state_topology(parent, malformed)


def test_only_hybrid_optimizer_config_difference_is_allowed() -> None:
    parent = vars(GPTConfig()).copy()
    hybrid = parent.copy()
    hybrid["block_fht_mlp_cproj_hybrid_output"] = True
    differences = validate_model_configs(parent, hybrid)
    assert differences == {"block_fht_mlp_cproj_hybrid_output": [False, True]}
    malformed = hybrid.copy()
    malformed["n_embd"] = parent["n_embd"] * 2
    with pytest.raises(ValueError, match="forward-relevant"):
        validate_model_configs(parent, malformed)


def test_selector_history_does_not_change_eval_forward() -> None:
    common = {
        "in_features": 6,
        "out_features": 4,
        "bias": False,
        "stages": 2,
        "residual_stages": 0,
        "neighbors": 2,
        "refresh_interval": 1,
        "fast_fresh_matching": True,
        "matching_seed": 17,
        "weight_std": 0.02,
    }
    parent = MuonMatchedGivensLinear(**common, output_stages=0)
    hybrid = MuonMatchedGivensLinear(
        **common,
        output_stages=2,
        hybrid_output=True,
        hybrid_directed_incoming=1,
        hybrid_control_output_stages=2,
    )
    hybrid.weight.copy_(parent.weight)
    hybrid.output_last_angles.fill_(0.7)
    hybrid.output_selected_permutations.copy_(
        torch.flip(hybrid.output_selected_permutations, dims=(1,))
    )
    parent.eval()
    hybrid.eval()
    values = torch.randn(3, 6)
    assert torch.equal(parent(values), hybrid(values))


@pytest.mark.parametrize("alpha", INTERIOR_ALPHAS)
def test_variant_names_are_stable(alpha: float) -> None:
    assert interpolation_variant("hybrid", alpha).endswith(
        f"alpha{alpha:.2f}".replace(".", "p")
    )
