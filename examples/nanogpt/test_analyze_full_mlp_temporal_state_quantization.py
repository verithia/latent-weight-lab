import torch

from examples.nanogpt.analyze_full_mlp_temporal_state_quantization import (
    blockwise_symmetric_quantize,
    classify,
    tensor_metrics,
)


def test_blockwise_quantization_preserves_zero_and_accounts_storage() -> None:
    source = torch.zeros(17)
    reconstructed, storage = blockwise_symmetric_quantize(
        source, bits=8, block_size=8
    )
    torch.testing.assert_close(reconstructed, source)
    assert storage["blocks"] == 3
    assert storage["packed_value_bytes"] == 17
    assert storage["scale_bytes"] == 6
    assert storage["persistent_bytes"] == 23


def test_blockwise_quantization_uses_fp16_maxabs_scale() -> None:
    source = torch.tensor([-2.0, -1.0, 0.0, 2.0, 0.25])
    reconstructed, storage = blockwise_symmetric_quantize(
        source, bits=4, block_size=4
    )
    assert storage["qmax"] == 7
    assert reconstructed[0].item() == -reconstructed[3].item()
    assert abs(reconstructed[3].item() - 2.0) < 0.002
    assert abs(reconstructed[4].item() - 0.25) < 0.002


def test_tensor_metrics_identity() -> None:
    source = torch.arange(1, 5, dtype=torch.float32)
    metrics = tensor_metrics(source, source)
    assert metrics["relative_fro_error"] == 0.0
    assert metrics["energy_recovery"] == 1.0
    assert abs(metrics["cosine"] - 1.0) < 1e-12


def _summary(*, passing: bool, storage: int = 25) -> dict:
    recovery = 0.99999 if passing else 0.9
    cosine = 0.99999 if passing else 0.9
    raw = {
        name: {
            "energy_recovery": recovery,
            "minimum_layer_cosine": cosine,
        }
        for name in ("momentum_buffer", "compression_residual")
    }
    return {
        "bits": 8,
        "block_size": 1024,
        "raw": raw,
        "momentum_state_only_polar_proxy": {
            "energy_recovery": recovery,
            "minimum_layer_cosine": cosine,
        },
        "persistent_storage_bytes": storage,
        "persistent_storage_ratio_to_dense_fp32": 0.251,
    }


def test_classify_requires_raw_and_polar_direction_gates() -> None:
    rule = {
        "maximum_persistent_storage_ratio": 0.26,
        "minimum_global_raw_energy_recovery": 0.9999,
        "minimum_layer_raw_cosine": 0.9999,
        "minimum_global_polar_energy_recovery": 0.999,
        "minimum_layer_polar_cosine": 0.999,
        "threshold_changes_after_measurement": False,
    }
    rejected = classify([_summary(passing=False)], rule)
    assert rejected["selected_candidate"] is None
    accepted = classify([_summary(passing=True)], rule)
    assert accepted["selected_candidate"] == {"bits": 8, "block_size": 1024}
