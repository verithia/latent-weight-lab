from __future__ import annotations

import json
from pathlib import Path
import math


ROOT = Path(__file__).resolve().parents[2]
DESIGN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "moe_active_parameter_ladder_0p1b_to_1b_design.json"
)


def load_design() -> dict:
    return json.loads(DESIGN.read_text())


def parameter_counts(rung: dict, design: dict) -> tuple[int, int, int]:
    architecture = design["architecture"]
    vocab = architecture["vocab_size"]
    block = architecture["block_size"]
    layers = rung["n_layer"]
    width = rung["n_embd"]
    experts = architecture["num_experts"]
    top_k = architecture["top_k"]
    hidden_multiplier = architecture["expert_hidden_multiplier"]

    shared = (
        vocab * width
        + block * width
        + layers * (4 * width**2 + 2 * width)
        + width
    )
    single_expert = 2 * hidden_multiplier * width**2
    router = layers * experts * width
    dense = shared + layers * 8 * width**2
    active = shared + router + layers * top_k * single_expert
    stored = shared + router + layers * experts * single_expert
    return dense, active, stored


def test_ladder_spans_requested_active_parameter_range_and_exact_counts() -> None:
    design = load_design()
    rungs = design["rungs"]
    assert [rung["name"] for rung in rungs] == [
        "moe_124m_active",
        "moe_350m_active",
        "moe_690m_active",
        "moe_985m_active",
    ]
    for rung in rungs:
        dense, active, stored = parameter_counts(rung, design)
        assert rung["dense_reference_parameters"] == dense
        assert rung["active_parameters"] == active
        assert rung["stored_parameters"] == stored
        assert rung["active_billions"] == active / 1e9
        assert rung["stored_billions"] == stored / 1e9
        assert stored > active
    assert 0.1e9 <= rungs[0]["active_parameters"] < 0.2e9
    assert 0.9e9 <= rungs[-1]["active_parameters"] <= 1.0e9


def test_each_expert_owns_projection_and_active_mlp_matches_dense() -> None:
    design = load_design()
    architecture = design["architecture"]
    assert architecture["num_experts"] == 8
    assert architecture["top_k"] == 2
    assert architecture["expert_hidden_multiplier"] == 2
    assert architecture["projection_ownership"].startswith(
        "Every expert owns both W_fc,e and W_proj,e"
    )
    for rung in design["rungs"]:
        width = rung["n_embd"]
        active_expert_matrices = (
            architecture["top_k"]
            * 2
            * architecture["expert_hidden_multiplier"]
            * width**2
        )
        assert active_expert_matrices == 8 * width**2


def test_dense_control_is_distinct_from_generated_candidate() -> None:
    design = load_design()
    arms = {arm["name"]: arm for arm in design["scientific_arms"]}
    assert arms["dense_complete_experts_control"]["launch_ready"] is True
    assert arms["dense_complete_experts_control"]["authorization_scope"] == (
        "124M-active selected 20TPP after its exact-config MFU gate"
    )
    assert (
        arms["adaptive_paired_neuron_generated_experts"]["launch_ready"]
        is False
    )
    assert "negative control" in arms[
        "shared_post_mixture_projection_negative_control"
    ]["blocker"]
    assert design["launch_authorization"]["current"].startswith(
        "124m_dense_complete_expert_selected_20tpp_"
    )


def test_performance_and_scaling_gates_are_binding() -> None:
    design = load_design()
    gates = design["gates"]
    assert gates["performance"]["minimum_mfu"] == 0.2
    assert "foreground" in gates["performance"]["preflight"]
    assert gates["scientific"]["generated_arm_requires_offline_direction_oracle"]
    assert gates["scientific"]["larger_rungs_authorized"] is False
    assert all(
        rung["authorization"]
        != "scientific_training_authorized"
        for rung in design["rungs"]
    )


def test_monitoring_matches_project_callback_policy() -> None:
    monitoring = load_design()["monitoring"]
    assert monitoring["heartbeat_interval_minutes"] == 90
    assert monitoring["progress_callback_resets_heartbeat"] is True
    assert "completion-or-error" in monitoring[
        "estimated_five_minutes_to_two_hours"
    ]


def test_every_rung_has_exact_active_tpp_iteration_schedules() -> None:
    design = load_design()
    tokens_per_iter = 32 * 8 * design["architecture"]["block_size"]
    for rung in design["rungs"]:
        active = rung["active_parameters"]
        for label, target_tpp in (("0.5TPP", 0.5), ("5TPP", 5.0), ("20TPP", 20.0)):
            schedule = rung["token_schedules"][label]
            expected_iters = math.ceil(target_tpp * active / tokens_per_iter)
            assert schedule["max_iters"] == expected_iters
            assert schedule["scheduled_tokens"] == expected_iters * tokens_per_iter
            assert schedule["scheduled_tpp"] == expected_iters * tokens_per_iter / active
