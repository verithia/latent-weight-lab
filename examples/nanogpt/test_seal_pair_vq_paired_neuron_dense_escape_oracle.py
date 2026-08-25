from examples.nanogpt.seal_pair_vq_paired_neuron_dense_escape_oracle import (
    DIFFUSE,
    INVALID,
    PASSED,
    _candidate_pass,
    classify,
)


def test_classification_is_fail_closed() -> None:
    assert classify(False, 0.25, 0.25) == INVALID
    assert classify(True, None, 0.25) == INVALID
    assert classify(True, 0.25, 0.25) == PASSED
    assert classify(True, 0.5, 0.25) == DIFFUSE


def test_candidate_pass_recomputes_all_functional_gates() -> None:
    thresholds = {
        "aggregate_heldout_functional_cosine": 0.99,
        "minimum_layer_heldout_functional_cosine": 0.98,
        "aggregate_heldout_positive_line_recovery": 0.98,
        "aggregate_task_line_retention": 0.98,
        "minimum_layer_task_line_retention": 0.95,
        "full_fraction_sanity_cosine": 0.999999,
    }
    candidate = {
        "measurements": {
            "aggregate_heldout_functional_cosine": 1.0,
            "minimum_layer_heldout_functional_cosine": 1.0,
            "aggregate_heldout_positive_line_recovery": 1.0,
            "aggregate_task_line_retention": 1.0,
            "minimum_layer_task_line_retention": 1.0,
        }
    }
    checks, passed = _candidate_pass(candidate, thresholds)
    assert passed
    assert all(checks.values())
    candidate["measurements"]["aggregate_task_line_retention"] = 0.5
    checks, passed = _candidate_pass(candidate, thresholds)
    assert not passed
    assert not checks["aggregate_task_line_retention"]
