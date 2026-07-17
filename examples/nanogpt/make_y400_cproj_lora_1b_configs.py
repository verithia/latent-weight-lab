from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"
MAX_ITERS = 7630
TOKENS_PER_ITER = 131072

ATTN = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj"]
CPROJ = "mlp.c_proj"

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
    "block_fht_latent_ratio": 0.01,
    "block_fht_latent_init_std": 0.02,
    "block_fht_modulation_alpha": 0.0,
    "block_fht_match_gpt_init": True,
    "block_fht_layers": 2,
    "block_fht_seed": 1000,
    "block_fht_cache_weights": True,
    "freeze_non_block_fht": False,
}


def with_lr(config: dict[str, Any], learning_rate: float) -> dict[str, Any]:
    updated = dict(config)
    updated["learning_rate"] = learning_rate
    updated["min_lr"] = learning_rate / 10
    return updated


def write(name: str, updates: dict[str, Any]) -> str:
    config = dict(BASE)
    config.update(updates)
    config["out_dir"] = f"{REMOTE_RUN_DIR}/{name}"
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return f"{REMOTE_CONFIG_DIR}/{path.name}"


def cproj_lora(rank: int, learning_rate: float) -> dict[str, Any]:
    return with_lr(
        {
            "block_fht_targets": ATTN + [CPROJ],
            "block_fht_latent_ratios": {CPROJ: 0.005},
            "block_fht_cproj_lowrank_rank": rank,
        },
        learning_rate,
    )


def attention_control() -> dict[str, Any]:
    return {
        "block_fht_targets": ATTN,
    }


def main() -> None:
    specs = [
        ("y400_cproj_lora_1b_01_attn_control_lr22e4", attention_control()),
        ("y400_cproj_lora_1b_02_r16_lr30e4_repeat", cproj_lora(16, 0.0030)),
        ("y400_cproj_lora_1b_03_r16_lr26e4", cproj_lora(16, 0.0026)),
        ("y400_cproj_lora_1b_04_r16_lr34e4", cproj_lora(16, 0.0034)),
        ("y400_cproj_lora_1b_05_r8_lr30e4", cproj_lora(8, 0.0030)),
        ("y400_cproj_lora_1b_06_r24_lr30e4", cproj_lora(24, 0.0030)),
        ("y400_cproj_lora_1b_07_r32_lr26e4", cproj_lora(32, 0.0026)),
        ("y400_cproj_lora_1b_08_r32_lr30e4", cproj_lora(32, 0.0030)),
    ]
    tasks = [(name, write(name, updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y400_cproj_lora_1b_queue.tsv"
    queue_path.write_text(
        "".join(f"{name}\t{path}\t{MAX_ITERS}\t{TOKENS_PER_ITER}\n" for name, path in tasks),
        encoding="utf-8",
    )
    print(queue_path)


if __name__ == "__main__":
    main()
