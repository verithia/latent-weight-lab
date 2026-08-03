from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_joint_prospective_step_by_depth import (
    classify_depth_interactions,
    make_band_variants,
)


def test_make_band_variants_slices_both_families_exactly() -> None:
    cfc = {layer: torch.tensor(float(layer)) for layer in range(4)}
    cproj = {layer: torch.tensor(float(layer + 10)) for layer in range(4)}
    variants = make_band_variants(
        cfc, cproj, {"early": [0, 1], "late": [2, 3], "all": [0, 1, 2, 3]}
    )
    assert set(variants["early"]["cfc_only"]["c_fc"]) == {0, 1}
    assert set(variants["late"]["cproj_only"]["c_proj"]) == {2, 3}
    assert set(variants["all"]["joint"]) == {"c_fc", "c_proj"}
    assert set(variants["all"]["joint"]["c_fc"]) == {0, 1, 2, 3}


def _rows(interactions: dict[str, list[float]]):
    rows = []
    for band, values in interactions.items():
        for index, interaction in enumerate(values):
            rows.append(
                {
                    "window": f"validation_{index // 4 + 1}",
                    "band": band,
                    "cfc_loss_change": -0.02,
                    "cproj_loss_change": -0.03,
                    "joint_loss_change": -0.05 + interaction,
                    "finite_interaction": interaction,
                }
            )
    return rows


def test_depth_classifier_requires_ci_and_window_sign_agreement() -> None:
    interactions = {
        "destructive": [0.010, 0.011, 0.009, 0.010] * 4,
        "cooperative": [-0.010, -0.011, -0.009, -0.010] * 4,
        "additive": [-0.001, 0.001, -0.001, 0.001] * 4,
        "mixed": [0.020, 0.018, 0.019, 0.021] * 2
        + [-0.020, -0.018, -0.019, -0.021] * 2,
    }
    result = classify_depth_interactions(
        _rows(interactions),
        bands=list(interactions),
        confidence_z=2.576,
        additive_tolerance=0.10,
    )
    assert result["bands"]["destructive"]["classification"] == (
        "DESTRUCTIVE_CFC_CPROJ_UPDATE_INTERACTION"
    )
    assert result["bands"]["cooperative"]["classification"] == (
        "COOPERATIVE_CFC_CPROJ_UPDATE_INTERACTION"
    )
    assert result["bands"]["additive"]["classification"] == (
        "CFC_CPROJ_UPDATES_ARE_FINITE_CE_ADDITIVE"
    )
    assert result["bands"]["mixed"]["classification"] == (
        "MIXED_CFC_CPROJ_UPDATE_INTERACTION"
    )
    assert result["classification"] == "DEPTH_LOCALIZED_DESTRUCTIVE_INTERACTION"
    assert result["destructive_bands"] == ["destructive"]
