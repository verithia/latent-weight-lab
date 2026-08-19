from examples.nanogpt.make_pro6_full_mlp_int8_lattice_124m import build_config


def test_generated_config_is_the_frozen_full_mlp_lattice_rung() -> None:
    config = build_config()
    assert config["block_fht_mlp_int8_lattice_targets"] == [
        "mlp.c_fc",
        "mlp.c_proj",
    ]
    assert config["block_fht_mlp_int8_lattice_block_size"] == 4096
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
    ]
    assert config["block_fht_attn_cproj_int8_lattice"] is True
    assert config["max_iters"] == config["lr_decay_iters"] == 238
    assert config["planned_tpp"] == 0.5
    assert config["mfu_min_fraction"] == 0.2
    assert config["launch_ready"] is True
    assert config["mlp_int8_lattice_representation"] == {
        "base": "independent reproducible frozen Gaussian initialization",
        "blocks": 13824,
        "code_bytes": 56623104,
        "elements": 56623104,
        "fp16_scale_bytes": 27648,
        "fp32_weight_bytes": 226492416,
        "optimizer_momentum": "dense_fp32_not_in_codec_count",
        "persistent_codec_bytes": 56650752,
        "runtime_base": "transient_dense_fp32",
        "runtime_weight": "transient_dense_fp32",
        "storage_ratio": 0.2501220703125,
        "storage_reduction": 3.998047828208882,
    }
