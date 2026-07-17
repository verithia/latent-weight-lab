from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"

ATTN = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj"]
CPROJ = "mlp.c_proj"

BASE = {
    "data_dir": "/root/userdata/MappingNetworks/data/finewebedu_2b",
    "init_from": "scratch",
    "max_iters": 763,
    "lr_decay_iters": 763,
    "eval_interval": 250,
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
    "block_fht_latent_ratio": 0.01,
    "block_fht_latent_init_std": 0.02,
    "block_fht_modulation_alpha": 0.0,
    "block_fht_match_gpt_init": True,
    "block_fht_layers": 2,
    "block_fht_seed": 1000,
    "block_fht_cache_weights": True,
    "freeze_non_block_fht": False,
}


def with_lr(config: dict, learning_rate: float) -> dict:
    updated = dict(config)
    updated["learning_rate"] = learning_rate
    updated["min_lr"] = learning_rate / 10
    return updated


def write(name: str, updates: dict) -> str:
    config = dict(BASE)
    config.update(updates)
    config["out_dir"] = f"{REMOTE_RUN_DIR}/{name}"
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return f"{REMOTE_CONFIG_DIR}/{path.name}"


def cproj_rank16(learning_rate: float) -> dict:
    return with_lr(
        {
            "block_fht_targets": ATTN + [CPROJ],
            "block_fht_latent_ratios": {CPROJ: 0.005},
            "block_fht_cproj_lowrank_rank": 16,
        },
        learning_rate,
    )


def attention_control() -> dict:
    return {
        "block_fht_targets": ATTN,
    }


def main() -> None:
    specs = [
        ("y400_cproj_lrbracket_100m_01_attn_control_lr22e4", attention_control()),
        ("y400_cproj_lrbracket_100m_02_r0050_lora16_lr18e4", cproj_rank16(0.0018)),
        ("y400_cproj_lrbracket_100m_03_r0050_lora16_lr22e4_repeat", cproj_rank16(0.0022)),
        ("y400_cproj_lrbracket_100m_04_r0050_lora16_lr26e4", cproj_rank16(0.0026)),
        ("y400_cproj_lrbracket_100m_05_r0050_lora16_lr30e4", cproj_rank16(0.0030)),
    ]
    tasks = [(name, write(name, updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y400_cproj_lrbracket_100m_queue.tsv"
    queue_path.write_text(
        "".join(f"{name}\t{path}\t763\t131072\n" for name, path in tasks),
        encoding="utf-8",
    )
    print(queue_path)


if __name__ == "__main__":
    main()
