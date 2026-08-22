from examples.nanogpt.fixed_model_compute_equivalence import EvalPoint
from examples.nanogpt.seal_qkonly_pairvq_mlp_5tpp_localization import (
    candidate_points,
    classification,
    reference_points,
)


def test_reference_and_candidate_points() -> None:
    reference = {
        "run": {
            "fixed_evaluations": [
                {"step": 594, "validation_ce": 4.0173},
                {"step": 2373, "validation_ce": 3.4858},
            ]
        }
    }
    losses = {
        0: {"train": 11.0, "val": 10.9},
        594: {"train": 4.1, "val": 4.0},
        2373: {"train": 3.5, "val": 3.49},
    }
    assert reference_points(reference)[-1] == EvalPoint(2373.0, 3.4858)
    assert candidate_points(losses, [0, 594, 2373])[-1] == EvalPoint(
        2373.0, 3.49
    )


def test_classification_is_semantic_only_after_valid_seal() -> None:
    assert classification(False, 0.0, 0.01).startswith("INVALID_")
    assert classification(True, 0.0099, 0.01).startswith("MLP_NEGLIGIBLE_")
    assert classification(True, 0.0101, 0.01).startswith("MLP_MATERIAL_")
