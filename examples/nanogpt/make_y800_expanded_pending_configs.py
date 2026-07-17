from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"


BASE = {
    "method": "block_fht",
    "data_dir": "/root/userdata/MappingNetworks/data/finewebedu_2b",
    "init_from": "scratch",
    "max_iters": 763,
    "lr_decay_iters": 763,
    "eval_interval": 125,
    "eval_iters": 5,
    "batch_size": 4,
    "gradient_accumulation_steps": 32,
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "vocab_size": 50304,
    "learning_rate": 0.0015,
    "min_lr": 0.00015,
    "warmup_iters": 50,
    "weight_decay": 0.1,
    "dtype": "bfloat16",
    "compile": False,
    "block_fht_targets": ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    "block_fht_latent_ratio": 0.05,
    "block_fht_latent_init_std": 0.02,
    "block_fht_modulation_alpha": 0.0,
    "block_fht_match_gpt_init": True,
    "block_fht_layers": 2,
    "block_fht_seed": 1000,
    "block_fht_cache_weights": True,
    "freeze_non_block_fht": False,
}


def write(name: str, **updates) -> str:
    cfg = dict(BASE)
    cfg.update(updates)
    cfg["out_dir"] = f"{REMOTE_RUN_DIR}/{name}"
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return f"{REMOTE_CONFIG_DIR}/{path.name}"


def main() -> None:
    tasks: list[tuple[str, str]] = []
    specs = [
        ("y800_exp_100m_01_cfc_lowrank16", {"block_fht_ffn_lowrank_rank": 16}),
        ("y800_exp_100m_02_cfc_lowrank32", {"block_fht_ffn_lowrank_rank": 32}),
        ("y800_exp_100m_03_cfc_lowrank128", {"block_fht_ffn_lowrank_rank": 128}),
        ("y800_exp_100m_04_cfc_lowrank64_s03", {"block_fht_ffn_lowrank_rank": 64, "block_fht_ffn_lowrank_scale": 0.3}),
        ("y800_exp_100m_05_gain_cfc", {"block_fht_output_gain_targets": ["mlp.c_fc"], "block_fht_cache_weights": False}),
        ("y800_exp_100m_06_gain_ffn", {"block_fht_output_gain_targets": ["mlp.c_fc", "mlp.c_proj"], "block_fht_cache_weights": False}),
        ("y800_exp_100m_07_gain_attn", {"block_fht_output_gain_targets": ["attn.c_attn", "attn.c_proj"], "block_fht_cache_weights": False}),
        ("y800_exp_100m_08_gain_cfc_lowrank64", {"block_fht_output_gain_targets": ["mlp.c_fc"], "block_fht_cache_weights": False, "block_fht_ffn_lowrank_rank": 64}),
        ("y800_exp_100m_09_postgelu_t010_l001", {"block_fht_ffn_postgelu_std_target": 0.10, "block_fht_ffn_postgelu_std_lambda": 0.01, "block_fht_cache_weights": False}),
        ("y800_exp_100m_10_postgelu_t020_l001", {"block_fht_ffn_postgelu_std_target": 0.20, "block_fht_ffn_postgelu_std_lambda": 0.01, "block_fht_cache_weights": False}),
        ("y800_exp_100m_11_postgelu_t015_l003", {"block_fht_ffn_postgelu_std_target": 0.15, "block_fht_ffn_postgelu_std_lambda": 0.003, "block_fht_cache_weights": False}),
        ("y800_exp_100m_12_maploss_stab", {"mapping_stability_lambda": 0.001, "mapping_stability_sigma": 0.001, "block_fht_cache_weights": False}),
        ("y800_exp_100m_13_maploss_norm003", {"mapping_norm_lambda": 1.0, "mapping_norm_target_rms": 0.03}),
        ("y800_exp_100m_14_maploss_norm004_stab", {"mapping_norm_lambda": 1.0, "mapping_norm_target_rms": 0.04, "mapping_stability_lambda": 0.001, "mapping_stability_sigma": 0.001, "block_fht_cache_weights": False}),
        ("y800_exp_100m_15_fht_r1_scaled", {"block_fht_layers": 1}),
        ("y800_exp_100m_16_fht_r3_scaled", {"block_fht_layers": 3}),
        ("y800_exp_100m_17_gradnorm_003", {"block_fht_latent_grad_normalize": True, "block_fht_latent_grad_target_rms": 0.003}),
        ("y800_exp_100m_18_gradnorm_010", {"block_fht_latent_grad_normalize": True, "block_fht_latent_grad_target_rms": 0.01}),
        ("y800_exp_100m_19_gradnorm_030", {"block_fht_latent_grad_normalize": True, "block_fht_latent_grad_target_rms": 0.03}),
        ("y800_exp_100m_20_alloc_cfc10", {"block_fht_latent_ratios": {"attn.c_attn": 0.05, "attn.c_proj": 0.05, "mlp.c_fc": 0.10, "mlp.c_proj": 0.05}}),
        ("y800_exp_100m_21_alloc_cfc10_attn8", {"block_fht_latent_ratios": {"attn.c_attn": 0.08, "attn.c_proj": 0.08, "mlp.c_fc": 0.10, "mlp.c_proj": 0.05}}),
        ("y800_exp_100m_22_alloc_proj10", {"block_fht_latent_ratios": {"attn.c_attn": 0.05, "attn.c_proj": 0.10, "mlp.c_fc": 0.05, "mlp.c_proj": 0.10}}),
        ("y800_exp_100m_23_small_6l384", {"batch_size": 8, "gradient_accumulation_steps": 16, "n_layer": 6, "n_head": 6, "n_embd": 384}),
        ("y800_exp_100m_24_small_8l512", {"batch_size": 8, "gradient_accumulation_steps": 16, "n_layer": 8, "n_head": 8, "n_embd": 512}),
        ("y800_exp_100m_25_untied_head", {"tie_word_embeddings": False}),
        ("y800_exp_100m_26_untied_head_lowrank64", {"tie_word_embeddings": False, "block_fht_ffn_lowrank_rank": 64}),
        ("y800_exp_100m_27_cfc_only_lat10", {"block_fht_targets": ["mlp.c_fc"], "block_fht_latent_ratio": 0.10}),
        ("y800_exp_100m_28_ffn_only_lat8", {"block_fht_targets": ["mlp.c_fc", "mlp.c_proj"], "block_fht_latent_ratio": 0.08}),
        ("y800_exp_100m_29_attn_only_lat8", {"block_fht_targets": ["attn.c_attn", "attn.c_proj"], "block_fht_latent_ratio": 0.08}),
        ("y800_exp_100m_30_combo_gain_lowrank_post", {"block_fht_output_gain_targets": ["mlp.c_fc"], "block_fht_cache_weights": False, "block_fht_ffn_lowrank_rank": 64, "block_fht_ffn_postgelu_std_target": 0.15, "block_fht_ffn_postgelu_std_lambda": 0.003}),
    ]
    for name, updates in specs:
        tasks.append((name, write(name, **updates)))
    queue_path = CONFIG_DIR / "y800_expanded_pending_queue.tsv"
    queue_path.write_text("".join(f"{name}\t{path}\t763\t131072\n" for name, path in tasks), encoding="utf-8")
    print(queue_path)


if __name__ == "__main__":
    main()
