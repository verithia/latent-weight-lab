from examples.nanogpt.make_pro6_attention_cproj_int8_lattice_124m_5tpp import (
    OUTPUT,
    build_config,
)

import json


def test_generated_config_is_the_frozen_same_size_horizon_transfer() -> None:
    config = build_config()
    assert json.loads(OUTPUT.read_text()) == config
    assert config["block_fht_attn_cproj_int8_lattice"] is True
    assert config["block_fht_attn_cproj_int8_lattice_block_size"] == 4096
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
    ]
    assert config["max_iters"] == config["lr_decay_iters"] == 2373
    assert config["warmup_iters"] == 23
    assert config["eval_interval"] == 594
    assert config["planned_tpp"] == 5.0
    assert config["scheduled_tokens"] == 622067712
    assert config["learning_rate"] == 0.0024
    assert config["min_lr"] == 0.00024
    assert config["mfu_min_fraction"] == 0.2
    assert config["launch_ready"] is True
    assert config["practical_equivalence_nll"] == 0.02
    assert "3.5602" in config["practical_equivalence_policy"]
    assert config["checkpoint_wall_clock_seconds"] == 7200
