from __future__ import annotations

from examples.nanogpt.analyze_mlp_pair_checkpoint_splice import (
    component_keys,
    depth_bands,
    recovery_decision,
    variant_specs,
)


def test_registered_variants_cover_pair_ln2_depth_and_converse() -> None:
    specs = {str(item["name"]): item for item in variant_specs(12)}
    assert depth_bands(12) == {
        "early": list(range(4)),
        "middle": list(range(4, 8)),
        "late": list(range(8, 12)),
    }
    assert specs["candidate_parent_mlp_pair_all"]["components"] == ["c_fc", "c_proj"]
    assert specs["candidate_parent_mlp_pair_ln2_all"]["components"] == [
        "c_fc",
        "c_proj",
        "ln_2",
    ]
    assert specs["parent_candidate_mlp_pair_all"]["source"] == "candidate"


def test_component_keys_are_endpoint_only() -> None:
    state = {
        "transformer.h.0.mlp.c_fc.weight": object(),
        "transformer.h.0.mlp.c_fc.latent": object(),
        "transformer.h.0.mlp.c_proj.weight": object(),
        "transformer.h.0.ln_2.weight": object(),
    }
    assert component_keys(state, 0, "c_fc") == ["transformer.h.0.mlp.c_fc.weight"]
    assert component_keys(state, 0, "c_proj") == ["transformer.h.0.mlp.c_proj.weight"]
    assert component_keys(state, 0, "ln_2") == ["transformer.h.0.ln_2.weight"]


def synthetic_rows(cfc: float, cproj: float, pair: float, pair_ln: float):
    rows = []
    for scale in ("124m", "350m"):
        for window in ("a", "b"):
            values = {
                "parent": 4.4,
                "candidate": 4.6,
                "candidate_parent_cfc_all": cfc,
                "candidate_parent_cproj_all": cproj,
                "candidate_parent_mlp_pair_all": pair,
                "candidate_parent_mlp_pair_ln2_all": pair_ln,
            }
            rows.extend(
                {"scale": scale, "window": window, "variant": name, "ce": ce}
                for name, ce in values.items()
            )
    return rows


def test_decision_separates_pair_ln_and_wider_context() -> None:
    assert recovery_decision(synthetic_rows(4.59, 4.58, 4.48, 4.47))[
        "classification"
    ] == "INTRA_MLP_PAIR_COADAPTATION_DOMINATES"
    assert recovery_decision(synthetic_rows(4.59, 4.58, 4.57, 4.48))[
        "classification"
    ] == "PRE_MLP_NORMALIZATION_COADAPTATION_DOMINATES"
    assert recovery_decision(synthetic_rows(4.59, 4.58, 4.57, 4.56))[
        "classification"
    ] == "WIDER_RESIDUAL_BLOCK_CONTEXT_DOMINATES"
