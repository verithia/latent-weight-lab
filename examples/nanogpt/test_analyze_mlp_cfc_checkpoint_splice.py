from __future__ import annotations

from examples.nanogpt.analyze_mlp_cfc_checkpoint_splice import (
    splice_decision,
    variant_specs,
)


def rows(parent: float, candidate: float, splice_a: float, splice_b: float):
    output = []
    for window, splice in (("a", splice_a), ("b", splice_b)):
        output.extend(
            [
                {"window": window, "variant": "parent", "ce": parent},
                {"window": window, "variant": "candidate", "ce": candidate},
                {"window": window, "variant": "candidate_parent_cfc_all", "ce": splice},
            ]
        )
    return output


def test_registered_variants_cover_global_bands_and_converse() -> None:
    specs = {str(item["name"]): item for item in variant_specs(24)}
    assert specs["candidate_parent_cfc_early"]["parent_cfc_layers"] == list(range(8))
    assert specs["candidate_parent_cfc_middle"]["parent_cfc_layers"] == list(range(8, 16))
    assert specs["candidate_parent_cfc_late"]["parent_cfc_layers"] == list(range(16, 24))
    assert specs["candidate_parent_cfc_all"]["parent_cfc_layers"] == list(range(24))
    assert specs["parent_candidate_cfc_all"]["candidate_cfc_layers"] == list(range(24))


def test_splice_decision_uses_both_fixed_windows() -> None:
    assert splice_decision(rows(4.4, 4.6, 4.49, 4.48))["classification"] == (
        "MISSING_CFC_ENDPOINT_DIRECTION_DOMINATES"
    )
    assert splice_decision(rows(4.4, 4.6, 4.57, 4.56))["classification"] == (
        "DOWNSTREAM_COADAPTATION_DOMINATES"
    )
    assert splice_decision(rows(4.4, 4.6, 4.48, 4.56))["classification"] == (
        "MIXED_CFC_DIRECTION_AND_COADAPTATION"
    )
