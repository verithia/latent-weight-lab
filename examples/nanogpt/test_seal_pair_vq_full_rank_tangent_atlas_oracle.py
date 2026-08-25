from examples.nanogpt.seal_pair_vq_full_rank_tangent_atlas_oracle import (
    PASSED_S1,
    PASSED_S2,
    REJECTED,
    candidate_summary,
    classification,
)


def test_classification_is_fail_closed_and_ordered() -> None:
    assert classification(False, True, True).endswith("INVALID")
    assert classification(True, True, True) == PASSED_S1
    assert classification(True, False, True) == PASSED_S2
    assert classification(True, False, False) == REJECTED


def test_candidate_summary_preserves_side_specific_minima() -> None:
    matrices = []
    for side, offset in (("c_fc", 0.0), ("c_proj", 0.1)):
        for index in range(12):
            matrices.append(
                {
                    "side": side,
                    "value": {"cosine": 0.99 - offset - index * 1e-4},
                    "tangent": {"cosine": 0.50 - offset - index * 1e-3},
                    "functional": {"cosine": 0.40 - offset - index * 1e-3},
                    "task_line_retention": 0.30 - offset - index * 1e-3,
                }
            )
    candidate = {
        "passed": False,
        "protocol": {"atoms": 1, "coordinates_per_panel": 8448, "stages": 10},
        "checks": {},
        "measurements": {},
        "tangent": {
            "cosine": 0.1,
            "positive_line_recovery": 0.01,
            "candidate_energy": 1.0,
            "reference_energy": 100.0,
        },
        "functional": {
            "cosine": 0.2,
            "positive_line_recovery": 0.04,
            "candidate_energy": 2.0,
            "reference_energy": 100.0,
        },
        "matrices": matrices,
    }
    summary = candidate_summary(candidate)
    assert summary["c_fc"]["minimum_tangent_cosine"] == 0.489
    assert summary["c_proj"]["minimum_tangent_cosine"] == 0.389
    assert summary["aggregate_tangent"]["candidate_to_reference_energy"] == 0.01
