from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"
TOKENS_PER_ITER = 131072

FULL_ATTN_QK = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj"]
FULL_ATTN_MIX = ["attn.c_attn.qk_mix25_headwise", "attn.c_attn.v", "attn.c_proj"]
FULL_ATTN_CPROJ = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj", "mlp.c_proj"]

BASE: dict[str, Any] = {
    "data_dir": "/root/userdata/MappingNetworks/data/finewebedu_2b",
    "init_from": "scratch",
    "eval_iters": 5,
    "save_checkpoint": False,
    "batch_size": 4,
    "gradient_accumulation_steps": 32,
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "vocab_size": 50304,
    "warmup_iters": 100,
    "weight_decay": 0.1,
    "optimizer": "muon",
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "dtype": "bfloat16",
    "compile": False,
    "method": "block_fht",
    "block_fht_latent_ratio": 0.01,
    "block_fht_latent_init_std": 0.02,
    "block_fht_modulation_alpha": 0.0,
    "block_fht_match_gpt_init": True,
    "block_fht_layers": 2,
    "block_fht_seed": 1000,
    "block_fht_cache_weights": True,
    "freeze_non_block_fht": False,
}


def write(name: str, updates: dict[str, Any]) -> str:
    config = dict(BASE)
    config.update(updates)
    if "learning_rate" in updates and "min_lr" not in updates:
        config["min_lr"] = float(updates["learning_rate"]) / 10
    config["out_dir"] = f"{REMOTE_RUN_DIR}/{name}"
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return f"{REMOTE_CONFIG_DIR}/{path.name}"


def main() -> None:
    specs = [
        (
            "y400_next_idle_01_fhtlora_r16_lr26_decay6500",
            {
                "max_iters": 7630,
                "lr_decay_iters": 6500,
                "eval_interval": 500,
                "learning_rate": 0.0026,
                "min_lr": 0.000052,
                "block_fht_targets": FULL_ATTN_CPROJ,
                "block_fht_cproj_lowrank_mode": "block_fht",
                "block_fht_cproj_lowrank_rank": 16,
                "block_fht_cproj_lowrank_scale": 1.0,
                "block_fht_cproj_lowrank_latent_ratio": 0.01,
            },
        ),
        (
            "y400_next_idle_02_fhtlora_r16_lr24_decay6500",
            {
                "max_iters": 7630,
                "lr_decay_iters": 6500,
                "eval_interval": 500,
                "learning_rate": 0.0024,
                "min_lr": 0.000048,
                "block_fht_targets": FULL_ATTN_CPROJ,
                "block_fht_cproj_lowrank_mode": "block_fht",
                "block_fht_cproj_lowrank_rank": 16,
                "block_fht_cproj_lowrank_scale": 1.0,
                "block_fht_cproj_lowrank_latent_ratio": 0.01,
            },
        ),
        (
            "y400_next_idle_03_fhtlora_liveB_lr24_decay6500",
            {
                "max_iters": 7630,
                "lr_decay_iters": 6500,
                "eval_interval": 500,
                "learning_rate": 0.0024,
                "min_lr": 0.000048,
                "block_fht_targets": FULL_ATTN_CPROJ,
                "block_fht_cproj_lowrank_mode": "block_fht",
                "block_fht_cproj_lowrank_rank": 16,
                "block_fht_cproj_lowrank_scale": 0.1,
                "block_fht_cproj_lowrank_b_zero_init": False,
                "block_fht_cproj_lowrank_latent_ratio": 0.01,
            },
        ),
        (
            "y400_next_idle_04_fullattn_qk_lr21_4b",
            {
                "max_iters": 30520,
                "lr_decay_iters": 30520,
                "eval_interval": 1000,
                "learning_rate": 0.0021,
                "block_fht_targets": FULL_ATTN_QK,
            },
        ),
        (
            "y400_next_idle_05_fullattn_mix25_lr19_4b",
            {
                "max_iters": 30520,
                "lr_decay_iters": 30520,
                "eval_interval": 1000,
                "learning_rate": 0.0019,
                "block_fht_targets": FULL_ATTN_MIX,
            },
        ),
    ]
    tasks = [(name, write(name, updates), int(updates["max_iters"])) for name, updates in specs]
    queue_path = CONFIG_DIR / "y400_next_idle_queue.tsv"
    queue_path.write_text(
        "".join(f"{name}\t{path}\t{iters}\t{TOKENS_PER_ITER}\n" for name, path, iters in tasks),
        encoding="utf-8",
    )
    print(queue_path)


if __name__ == "__main__":
    main()
