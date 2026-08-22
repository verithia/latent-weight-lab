import math

from examples.nanogpt.seal_attention_pair_vq_vo_endpoint import (
    compute_equivalent_penalty,
    expected_eval_steps,
    parse_fixed_losses,
    parse_json_lines,
    parse_key_value_line,
    parse_perf_tps,
)


def test_parse_endpoint_log_contract() -> None:
    text = "\n".join(
        [
            'rng_eval_metadata {"fixed_eval_indices_sha256":"abc"}',
            "mlp_pair_vq: modules=48 elements=70,778,880 codec_bytes=88,627,584 persistent_training_bytes=319,996,032 dense_master_weight=disabled",
            "step 0: train loss 11.0, val loss 10.9",
            "perf iter=10 tokens_per_s=140151.78 iter_ms=1870.60",
            'pair_vq_reserved_escape {"step":10,"momentum_bytes":112960984}',
            "step 238: train loss 5.3, val loss 5.36",
        ]
    )
    losses = parse_fixed_losses(text)
    assert losses[238] == {"train": 5.3, "val": 5.36}
    assert parse_perf_tps(text) == [140151.78]
    stats = parse_key_value_line(text, "mlp_pair_vq: ")
    assert stats["elements"] == "70,778,880"
    assert stats["dense_master_weight"] == "disabled"
    assert parse_json_lines(text, "pair_vq_reserved_escape ")[0]["step"] == 10


def test_eval_inventory_and_compute_penalty() -> None:
    assert expected_eval_steps(238, 60) == [0, 60, 120, 180, 238]
    observed = compute_equivalent_penalty(0.0028, 5.3602, 0.07)
    assert math.isclose(observed, 1.007490321276494, rel_tol=1e-12)
