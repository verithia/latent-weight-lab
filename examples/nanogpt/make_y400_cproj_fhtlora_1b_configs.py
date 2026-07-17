from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"
MAX_ITERS = 7630
TOKENS_PER_ITER = 131072

FULL_ATTN_CPROJ = [
    "attn.c_attn.qk_headwise",
    "attn.c_attn.v",
    "attn.c_proj",
    "mlp.c_proj",
]

BASE: dict[str, Any] = {
    "data_dir": "/root/userdata/MappingNetworks/data/finewebedu_2b",
    "init_from": "scratch",
    "max_iters": MAX_ITERS,
    "lr_decay_iters": MAX_ITERS,
    "eval_interval": 500,
    "eval_iters": 5,
    "save_checkpoint": False,
    "batch_size": 4,
    "gradient_accumulation_steps": 32,
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "vocab_size": 50304,
    "learning_rate": 0.0022,
    "min_lr": 0.00022,
    "warmup_iters": 100,
    "weight_decay": 0.1,
    "optimizer": "muon",
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "dtype": "bfloat16",
    "compile": False,
    "method": "block_fht",
    "block_fht_targets": FULL_ATTN_CPROJ,
    "block_fht_latent_ratio": 0.01,
    "block_fht_latent_init_std": 0.02,
    "block_fht_modulation_alpha": 0.0,
    "block_fht_match_gpt_init": True,
    "block_fht_layers": 2,
    "block_fht_seed": 1000,
    "block_fht_cache_weights": True,
    "freeze_non_block_fht": False,
    "block_fht_cproj_lowrank_mode": "block_fht",
    "block_fht_cproj_lowrank_latent_ratio": 0.01,
    "block_fht_cproj_lowrank_init_std": 0.02,
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
        ("y400_cproj_fhtlora_1b_01_r16_zeroB_lr22e4", {"block_fht_cproj_lowrank_rank": 16, "block_fht_cproj_lowrank_scale": 1.0, "learning_rate": 0.0022}),
        ("y400_cproj_fhtlora_1b_02_r16_zeroB_lr26e4", {"block_fht_cproj_lowrank_rank": 16, "block_fht_cproj_lowrank_scale": 1.0, "learning_rate": 0.0026}),
        ("y400_cproj_fhtlora_1b_03_r32_zeroB_lr22e4", {"block_fht_cproj_lowrank_rank": 32, "block_fht_cproj_lowrank_scale": 1.0, "learning_rate": 0.0022}),
        ("y400_cproj_fhtlora_1b_04_r32_zeroB_lr18e4", {"block_fht_cproj_lowrank_rank": 32, "block_fht_cproj_lowrank_scale": 1.0, "learning_rate": 0.0018}),
        ("y400_cproj_fhtlora_1b_05_r16_liveB_s01_lr22e4", {"block_fht_cproj_lowrank_rank": 16, "block_fht_cproj_lowrank_scale": 0.1, "block_fht_cproj_lowrank_b_zero_init": False, "learning_rate": 0.0022}),
        ("y400_cproj_fhtlora_1b_06_r16_bias_zeroB_lr22e4", {"block_fht_cproj_lowrank_rank": 16, "block_fht_cproj_lowrank_scale": 1.0, "block_fht_cproj_lowrank_bias": True, "learning_rate": 0.0022}),
    ]
    tasks = [(name, write(name, updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y400_cproj_fhtlora_1b_queue.tsv"
    queue_path.write_text(
        "".join(f"{name}\t{path}\t{MAX_ITERS}\t{TOKENS_PER_ITER}\n" for name, path in tasks),
        encoding="utf-8",
    )
    print(queue_path)


if __name__ == "__main__":
    main()
