from __future__ import annotations

from examples.nanogpt.make_y400_matched_ns4_batchshape_124m import config_for


def test_batchshape_preflights_preserve_the_scientific_representation() -> None:
    for batch_size, gradient_accumulation_steps in ((64, 4), (128, 2)):
        config = config_for(batch_size, gradient_accumulation_steps)
        assert config["batch_size"] * config["gradient_accumulation_steps"] == 256
        assert config["tokens_per_update"] == 262144
        assert config["block_fht_mlp_pair_vq_feedback_codec"] == (
            "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16"
        )
        assert config["block_fht_mlp_pair_vq_fp16_ambient_momentum"] is False
        assert config["block_fht_mlp_pair_vq_lazy_retraction_interval"] == 1
        assert config["persistent_training_bytes_exact"] == 157500864
        assert config["mfu_preflight_pair_vq_persistent_training_bytes_exact"] == 157500864
        assert config["muon_ns_steps"] == 5
        assert config["muon_mlp_ns_steps"] == 4
        assert config["muon_mlp_lr_scale"] == 1.225
        assert config["compile"] is False
        assert config["preflight_only"] is True
        assert config["launch_ready"] is False
