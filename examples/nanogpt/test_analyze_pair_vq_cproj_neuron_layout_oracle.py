import torch

from examples.nanogpt.analyze_pair_vq_cproj_neuron_layout_oracle import (
    action_metrics,
    classify,
    matrix_to_pairs,
    pairs_to_matrix,
)


def test_layout_roundtrip_is_exact() -> None:
    weight = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    for layout in ("row_major", "neuron_major"):
        pairs = matrix_to_pairs(weight, layout=layout)
        restored = pairs_to_matrix(pairs, shape=(6, 8), layout=layout)
        torch.testing.assert_close(restored, weight)
    assert not torch.equal(
        matrix_to_pairs(weight, layout="row_major"),
        matrix_to_pairs(weight, layout="neuron_major"),
    )


def test_action_metric_is_exact_at_dense_weight() -> None:
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(13, 8, generator=generator)
    weight = torch.randn(6, 8, generator=generator)
    metrics = action_metrics(hidden, weight, weight.clone())
    assert metrics["error_energy"] == 0.0
    assert metrics["energy_recovery"] == 1.0
    assert abs(metrics["cosine"] - 1.0) < 1e-6


def test_classifier_requires_function_and_fixed_ce() -> None:
    thresholds = {
        "minimum_every_bank_action_error_closure": 0.10,
        "minimum_every_layer_action_error_closure": -0.05,
        "minimum_fixed_ce_improvement": 0.002,
        "minimum_fixed_ce_gap_closure": 0.10,
    }
    classification, gates = classify(
        bank_error_closures=[0.20, 0.15],
        layer_error_closures=[0.10, 0.00],
        dense_ce=3.40,
        row_major_ce=3.50,
        neuron_major_ce=3.48,
        equal_state_bytes=True,
        thresholds=thresholds,
    )
    assert classification == "CPROJ_NEURON_MAJOR_LAYOUT_AUTHORIZED"
    assert all(gates.values())

    classification, gates = classify(
        bank_error_closures=[0.20, 0.15],
        layer_error_closures=[0.10, 0.00],
        dense_ce=3.40,
        row_major_ce=3.50,
        neuron_major_ce=3.501,
        equal_state_bytes=True,
        thresholds=thresholds,
    )
    assert classification == "CPROJ_NEURON_MAJOR_LAYOUT_REJECTED"
    assert gates["fixed_ce_improves"] is False
