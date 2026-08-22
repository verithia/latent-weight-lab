from examples.nanogpt.fixed_model_compute_equivalence import EvalPoint
from examples.nanogpt.seal_attention_pair_vq_vo_5tpp import (
    candidate_points,
    dense_points,
)


def test_dense_result_and_candidate_log_points() -> None:
    dense = {
        "run": {
            "fixed_evaluations": [
                {"step": 594, "validation_ce": 4.1742},
                {"step": 1188, "validation_ce": 3.7631},
                {"step": 1782, "validation_ce": 3.6039},
                {"step": 2373, "validation_ce": 3.5402},
            ]
        }
    }
    losses = {
        0: {"train": 11.0, "val": 10.9},
        594: {"train": 4.1, "val": 4.15},
        1188: {"train": 3.7, "val": 3.75},
        1782: {"train": 3.6, "val": 3.59},
        2373: {"train": 3.5, "val": 3.5499},
    }
    assert dense_points(dense)[-1] == EvalPoint(2373.0, 3.5402)
    assert candidate_points(losses, [0, 594, 1188, 1782, 2373])[-1] == EvalPoint(
        2373.0, 3.5499
    )
