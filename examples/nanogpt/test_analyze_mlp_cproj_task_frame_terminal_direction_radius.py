from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cproj_task_frame_terminal_direction_radius import (
    GROUP_SUFFIXES,
    make_variants,
    select_decision,
    state_group_norm,
)


def synthetic_state(scales: tuple[float, float, float]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer in range(12):
        for suffix, scale in zip(GROUP_SUFFIXES, scales, strict=True):
            state[f"layer.{layer}.{suffix}"] = torch.tensor(
                [scale * (layer + 1), scale * (layer + 2)],
                dtype=torch.float32,
            )
    return state


def test_variants_preserve_direction_and_match_registered_group_radius() -> None:
    native = synthetic_state((1.0, 2.0, 3.0))
    endpoint = synthetic_state((4.0, 6.0, 8.0))
    variants, scales = make_variants(native, endpoint)
    assert torch.count_nonzero(
        torch.cat(list(variants["identity"].values()))
    ).item() == 0
    for suffix in GROUP_SUFFIXES:
        native_norm = state_group_norm(native, suffix)
        endpoint_norm = state_group_norm(endpoint, suffix)
        assert torch.isclose(
            torch.tensor(
                state_group_norm(
                    variants["endpoint_direction_native_radius"], suffix
                )
            ),
            torch.tensor(native_norm),
        )
        assert torch.isclose(
            torch.tensor(
                state_group_norm(
                    variants["native_direction_endpoint_radius"], suffix
                )
            ),
            torch.tensor(endpoint_norm),
        )
        assert scales["endpoint_direction_to_native_radius"][suffix] < 1.0
        assert scales["native_direction_to_endpoint_radius"][suffix] > 1.0


def ce_table(
    *, full: float, endpoint_small: float, native_large: float
) -> dict[str, dict[str, float]]:
    native = {"primary": 5.5, "confirmation": 5.6}
    return {
        "native_delayed": native,
        "identity": {"primary": 5.501, "confirmation": 5.601},
        "endpoint_full_radius": {
            "primary": native["primary"] - full,
            "confirmation": native["confirmation"] - full,
        },
        "endpoint_direction_native_radius": {
            "primary": native["primary"] - endpoint_small,
            "confirmation": native["confirmation"] - endpoint_small,
        },
        "native_direction_endpoint_radius": {
            "primary": native["primary"] - native_large,
            "confirmation": native["confirmation"] - native_large,
        },
    }


def test_decision_identifies_amplitude_direction_both_and_nonportability() -> None:
    amplitude = select_decision(
        ce_table(full=0.02, endpoint_small=0.004, native_large=0.012),
        minimum_gain=0.005,
        minimum_fraction=0.5,
    )
    assert amplitude["decision"] == "AMPLITUDE_DOMINATES"
    direction = select_decision(
        ce_table(full=0.02, endpoint_small=0.012, native_large=0.004),
        minimum_gain=0.005,
        minimum_fraction=0.5,
    )
    assert direction["decision"] == "DIRECTION_DOMINATES"
    both = select_decision(
        ce_table(full=0.02, endpoint_small=0.006, native_large=0.006),
        minimum_gain=0.005,
        minimum_fraction=0.5,
    )
    assert both["decision"] == "BOTH_DIRECTION_AND_RADIUS_MATTER"
    not_portable = select_decision(
        ce_table(full=0.004, endpoint_small=0.003, native_large=0.003),
        minimum_gain=0.005,
        minimum_fraction=0.5,
    )
    assert not_portable["decision"] == "ENDPOINT_NOT_PORTABLE"


def test_negative_confirmation_window_fails_sufficiency() -> None:
    table = ce_table(full=0.02, endpoint_small=0.012, native_large=0.012)
    table["native_direction_endpoint_radius"]["confirmation"] = 5.61
    result = select_decision(
        table,
        minimum_gain=0.005,
        minimum_fraction=0.5,
    )
    assert result["decision"] == "DIRECTION_DOMINATES"
    assert result["automatic_training_run_authorized"] is False
