import torch

from examples.nanogpt.analyze_qk_cfc_additive_normal_budget import (
    classify,
    minimum_material_rank_allocation,
    recovery_from_spectrum,
)


def test_recovery_from_spectrum_counts_residual_tail() -> None:
    spectrum = torch.tensor([4.0, 1.0])
    assert recovery_from_spectrum(10.0, spectrum, 0) == 0.5
    assert recovery_from_spectrum(10.0, spectrum, 1) == 0.9
    assert recovery_from_spectrum(10.0, spectrum, 2) == 1.0


def test_allocation_enforces_late_then_global_gate() -> None:
    rows = [
        {"target_energy": 10.0, "residual_spectrum": torch.tensor([4.0, 1.0])}
        for _ in range(12)
    ]
    result = minimum_material_rank_allocation(
        rows, aggregate_threshold=0.8, late_layer_threshold=0.7
    )
    assert result["all_energy_weighted_recovery"] >= 0.8
    assert result["late_minimum_layer_recovery"] >= 0.7
    assert all(result["ranks_by_layer"][layer] >= 1 for layer in (8, 9, 10, 11))


def test_classification_uses_joint_material_bytes() -> None:
    rule = {
        "maximum_compact_joint_byte_ratio": 0.7,
        "maximum_dense_equivalent_joint_byte_ratio": 1.0,
    }
    accounting = {
        "output6_then_input": {"joint_byte_ratio_to_dense": 0.8},
        "input_then_output6": {"joint_byte_ratio_to_dense": 0.65},
    }
    result = classify(accounting, rule)
    assert result["classification"] == "LOW_RANK_NORMAL_COMPLETION_PLAUSIBLE"
    assert result["selected_oracle_family"] == "input_then_output6"
    accounting["input_then_output6"]["joint_byte_ratio_to_dense"] = 0.9
    assert classify(accounting, rule)["classification"] == "DENSE_SCALE_NORMAL_COMPLETION"
    accounting["output6_then_input"]["joint_byte_ratio_to_dense"] = 1.1
    accounting["input_then_output6"]["joint_byte_ratio_to_dense"] = 1.2
    assert classify(accounting, rule)["classification"] == "NORMAL_COMPLETION_EXCEEDS_DENSE_STATE"
