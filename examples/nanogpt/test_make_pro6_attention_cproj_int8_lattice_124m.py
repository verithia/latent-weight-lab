from examples.nanogpt.make_pro6_attention_cproj_int8_lattice_124m import (
    build_config,
)


def test_generated_config_is_the_frozen_smallest_rung() -> None:
    config = build_config()
    assert config["block_fht_attn_cproj_int8_lattice"] is True
    assert config["block_fht_attn_cproj_int8_lattice_block_size"] == 4096
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
    ]
    assert config["max_iters"] == config["lr_decay_iters"] == 238
    assert config["checkpoint_wall_clock_seconds"] == 7200
    assert config["planned_tpp"] == 0.5
    assert config["mfu_min_fraction"] == 0.2
    assert config["int8_lattice_representation"] == {
        "base": "reproducible frozen Gaussian initialization",
        "blocks": 1728,
        "code_bytes": 7077888,
        "elements": 7077888,
        "fp16_scale_bytes": 3456,
        "fp32_weight_bytes": 28311552,
        "optimizer_momentum": "dense_fp32_not_in_codec_count",
        "persistent_codec_bytes": 7081344,
        "runtime_base": "transient_dense_fp32",
        "runtime_weight": "transient_dense_fp32",
        "storage_ratio": 0.2501220703125,
        "storage_reduction": 3.998047828208882,
    }
