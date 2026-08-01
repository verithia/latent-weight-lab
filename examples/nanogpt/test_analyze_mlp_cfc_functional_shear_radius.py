from examples.nanogpt.analyze_mlp_cfc_functional_shear_radius import (
    CONTROL,
    WEIGHT_SHEAR,
    aggregate_radius,
    scale_name,
)


def _rows(scale_value: float) -> list[dict]:
    rows = []
    for window in ("fit", "holdout"):
        for candidate, value in (
            (CONTROL, 0.2),
            (WEIGHT_SHEAR, 0.2),
            (scale_name(0.25), scale_value),
        ):
            rows.append(
                {
                    "window": window,
                    "candidate": candidate,
                    "weight_target_energy": 1.0,
                    "weight_fixed_scale_recovery": value,
                    "post_gelu_target_energy": 1.0,
                    "post_gelu_fixed_scale_recovery": value,
                    "mlp_output_target_energy": 1.0,
                    "mlp_output_fixed_scale_recovery": value,
                    "predicted_ce_decrease": value,
                }
            )
    return rows


def _fit_rows() -> list[dict]:
    return [
        {
            "candidate": scale_name(0.25),
            "finite": {
                "maximum_determinant_error": 1e-9,
                "maximum_condition_number": 1.001,
            },
        }
    ]


def test_radius_gate_promotes_balanced_gain() -> None:
    result = aggregate_radius(
        _rows(0.24),
        _fit_rows(),
        scales=[0.25],
        minimum_mlp_output_ratio=1.1,
        minimum_post_gelu_ratio=0.9,
        minimum_ce_descent_ratio=1.0,
        minimum_weight_ratio=0.9,
        maximum_determinant_error=1e-6,
        maximum_condition_number=1.01,
    )
    assert result["decision"] == "PROMOTE_FUNCTIONAL_COORDINATE_RADIUS_TO_FINITE_CE"
    assert result["selected_scale"] == 0.25


def test_radius_gate_rejects_hidden_geometry_loss() -> None:
    rows = _rows(0.24)
    for row in rows:
        if row["candidate"] == scale_name(0.25):
            row["post_gelu_fixed_scale_recovery"] = 0.1
    result = aggregate_radius(
        rows,
        _fit_rows(),
        scales=[0.25],
        minimum_mlp_output_ratio=1.1,
        minimum_post_gelu_ratio=0.9,
        minimum_ce_descent_ratio=1.0,
        minimum_weight_ratio=0.9,
        maximum_determinant_error=1e-6,
        maximum_condition_number=1.01,
    )
    assert result["decision"] == "REJECT_FUNCTIONAL_COORDINATE_TRUST_RADIUS"
