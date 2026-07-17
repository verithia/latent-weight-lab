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
    "save_checkpoint": False,
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
    "block_fht_targets": ["attn.c_proj", "mlp.c_fc"],
    "block_fht_latent_ratio": 0.08,
    "block_fht_latent_ratios": {"attn.c_proj": 0.08, "mlp.c_fc": 0.10},
    "block_fht_latent_init_std": 0.02,
    "block_fht_modulation_alpha": 0.0,
    "block_fht_match_gpt_init": True,
    "block_fht_layers": 2,
    "block_fht_seed": 1000,
    "block_fht_cache_weights": True,
    "block_fht_ffn_lowrank_rank": 32,
    "freeze_non_block_fht": False,
}


def write(name: str, **updates) -> str:
    cfg = dict(BASE)
    cfg.update(updates)
    if "learning_rate" in updates and "min_lr" not in updates:
        cfg["min_lr"] = float(updates["learning_rate"]) * 0.1
    cfg["out_dir"] = f"{REMOTE_RUN_DIR}/{name}"
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return f"{REMOTE_CONFIG_DIR}/{path.name}"


def main() -> None:
    specs = [
        (
            "y800_stage2c_100m_01_rank32_cfc_l8",
            {
                "block_fht_ffn_lowrank_rank": 32,
                "block_fht_latent_ratios": {"attn.c_proj": 0.08, "mlp.c_fc": 0.08},
            },
        ),
        (
            "y800_stage2c_100m_02_rank32_cfc_l12",
            {
                "block_fht_ffn_lowrank_rank": 32,
                "block_fht_latent_ratios": {"attn.c_proj": 0.08, "mlp.c_fc": 0.12},
            },
        ),
        (
            "y800_stage2c_100m_03_rank48_repeat",
            {"block_fht_ffn_lowrank_rank": 48},
        ),
        (
            "y800_stage2c_100m_04_rank32_lr12e4",
            {"block_fht_ffn_lowrank_rank": 32, "learning_rate": 0.0012},
        ),
        (
            "y800_stage2c_100m_05_rank32_lr18e4",
            {"block_fht_ffn_lowrank_rank": 32, "learning_rate": 0.0018},
        ),
        (
            "y800_stage2c_100m_06_rank48_lr18e4",
            {"block_fht_ffn_lowrank_rank": 48, "learning_rate": 0.0018},
        ),
    ]
    tasks = [(name, write(name, **updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y800_stage_cproj_cfc_tune_queue.tsv"
    queue_path.write_text("".join(f"{name}\t{path}\t763\t131072\n" for name, path in tasks), encoding="utf-8")
    print(queue_path)


if __name__ == "__main__":
    main()
