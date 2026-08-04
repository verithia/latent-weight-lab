from examples.nanogpt.analyze_attention_refresh_cadence import horizon_passes


def test_horizon_gate_requires_every_registered_condition() -> None:
    summary = {
        "energy_recovery": 0.21,
        "normalized_enrichment": 3.1,
        "maximum_orthogonality_error": 1e-8,
        "maximum_relative_normal_residual": 1e-6,
    }
    chord = {
        "energy_recovery": 0.051,
        "maximum_orthogonality_error": 1e-8,
        "maximum_relative_normal_residual": 1e-6,
    }
    by_target = {
        target: {"energy_recovery": 0.026}
        for target in ("qk", "v", "cproj")
    }
    thresholds = {
        "current_dense_recovery_minimum": 0.20,
        "current_dense_enrichment_minimum": 3.0,
        "future_chord_recovery_minimum": 0.05,
        "future_chord_over_random_minimum": 2.0,
        "per_target_chord_recovery_minimum": 0.025,
        "maximum_projection_error": 1e-4,
        "maximum_normal_residual": 1e-4,
    }
    assert horizon_passes(
        current=summary,
        chord=chord,
        chord_over_random=2.1,
        by_target=by_target,
        thresholds=thresholds,
    )
    assert not horizon_passes(
        current=summary,
        chord=chord,
        chord_over_random=1.9,
        by_target=by_target,
        thresholds=thresholds,
    )
