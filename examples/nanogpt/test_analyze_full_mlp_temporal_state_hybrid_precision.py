import torch

from examples.nanogpt.analyze_full_mlp_temporal_state_hybrid_precision import (
    classify,
    floating_reconstruct,
)


def test_floating_reconstruct_accounts_two_bytes_per_element() -> None:
    source = torch.tensor([1.0, 0.1, -0.2])
    reconstructed, storage = floating_reconstruct(source, "float16")
    assert reconstructed.dtype == torch.float32
    assert storage["persistent_bytes"] == 6
    assert storage["dense_fp32_bytes"] == 12


def _summary(name: str, *, passing: bool, ratio: float) -> dict:
    recovery = 0.99999 if passing else 0.99
    cosine = 0.99999 if passing else 0.99
    return {
        "name": name,
        "raw": {
            state: {"energy_recovery": recovery, "minimum_layer_cosine": cosine}
            for state in ("momentum_buffer", "compression_residual")
        },
        "momentum_state_only_polar_proxy": {
            "energy_recovery": recovery,
            "minimum_layer_cosine": cosine,
        },
        "persistent_storage_bytes": round(ratio * 1000),
        "persistent_storage_ratio_to_dense_fp32": ratio,
    }


def test_classify_selects_smallest_passing_hybrid() -> None:
    rule = {
        "maximum_persistent_storage_ratio": 0.38,
        "minimum_global_raw_energy_recovery": 0.9999,
        "minimum_layer_raw_cosine": 0.9999,
        "minimum_global_polar_energy_recovery": 0.999,
        "minimum_layer_polar_cosine": 0.999,
        "threshold_changes_after_measurement": False,
    }
    summaries = [
        _summary("rejected", passing=False, ratio=0.25),
        _summary("larger", passing=True, ratio=0.50),
        _summary("selected", passing=True, ratio=0.375),
    ]
    decision = classify(summaries, rule)
    assert decision["selected_candidate"] == "selected"
    assert decision["passing_candidate_count"] == 1
