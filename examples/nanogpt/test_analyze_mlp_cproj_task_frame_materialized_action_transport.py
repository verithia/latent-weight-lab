from __future__ import annotations

from examples.nanogpt.analyze_mlp_cproj_task_frame_materialized_action_transport import (
    select_decision,
)


def table(
    *,
    raw: tuple[float, float] = (-0.02, -0.02),
    full: tuple[float, float],
    cfc: tuple[float, float],
    cproj: tuple[float, float],
) -> dict[str, dict[str, float]]:
    native = {"primary": 5.5, "confirmation": 5.6}

    def ce(gain: tuple[float, float]) -> dict[str, float]:
        return {
            "primary": native["primary"] - gain[0],
            "confirmation": native["confirmation"] - gain[1],
        }

    return {
        "native_delayed": native,
        "identity": ce((-0.04, -0.04)),
        "raw_endpoint_coordinates": ce(raw),
        "additive_materialized_full": ce(full),
        "additive_materialized_cfc_only": ce(cfc),
        "additive_materialized_cproj_only": ce(cproj),
    }


def test_portable_action_and_supported_component() -> None:
    result = select_decision(
        table(full=(0.02, 0.018), cfc=(0.012, 0.011), cproj=(0.004, 0.003)),
        minimum_gain=0.005,
        component_fraction=0.5,
    )
    assert result["decision"] == "MATERIALIZED_ACTION_PORTABLE"
    assert result["raw_coordinate_control_passed"] is True
    assert result["component_supported"]["additive_materialized_cfc_only"] is True
    assert result["component_supported"]["additive_materialized_cproj_only"] is False
    assert result["automatic_training_run_authorized"] is False


def test_nonportable_when_one_window_is_negative() -> None:
    result = select_decision(
        table(full=(0.02, -0.001), cfc=(0.01, 0.0), cproj=(0.01, 0.0)),
        minimum_gain=0.005,
        component_fraction=0.5,
    )
    assert result["decision"] == "MATERIALIZED_ACTION_NONPORTABLE"


def test_raw_control_sign_drift_fails_closed() -> None:
    result = select_decision(
        table(
            raw=(0.001, -0.02),
            full=(0.02, 0.02),
            cfc=(0.01, 0.01),
            cproj=(0.01, 0.01),
        ),
        minimum_gain=0.005,
        component_fraction=0.5,
    )
    assert result["raw_coordinate_control_passed"] is False
    assert result["decision"] == "MATERIALIZED_ACTION_NONPORTABLE"
