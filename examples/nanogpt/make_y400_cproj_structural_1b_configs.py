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
CPROJ_OUTMIX = "mlp.c_proj_outmix"
CPROJ_OUTGROUP12_MIX = "mlp.c_proj_outgroup12_mix"
CPROJ_GROUP12_INMIX = "mlp.c_proj_group12_inmix"

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
    "learning_rate": 0.003,
    "min_lr": 0.0003,
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


def attention_control() -> dict[str, Any]:
    return {"block_fht_targets": ATTN}


def cproj_target(target: str, rank: int, learning_rate: float, input_gain: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {
        "block_fht_targets": ATTN + [target],
        "block_fht_latent_ratios": {target: 0.005},
    }
    if rank > 0:
        config["block_fht_cproj_lowrank_rank"] = rank
    if input_gain:
        config["block_fht_input_gain_targets"] = [target]
        config["block_fht_cache_weights"] = False
    return with_lr(config, learning_rate)


def main() -> None:
    specs = [
        ("y400_cproj_struct_1b_01_attn_control_lr22e4", with_lr(attention_control(), 0.0022)),
        ("y400_cproj_struct_1b_02_plain_r16_lr30e4", cproj_target(CPROJ, 16, 0.0030)),
        ("y400_cproj_struct_1b_03_plain_r16_colgain_lr30e4", cproj_target(CPROJ, 16, 0.0030, input_gain=True)),
        ("y400_cproj_struct_1b_04_outmix_r16_lr30e4", cproj_target(CPROJ_OUTMIX, 16, 0.0030)),
        ("y400_cproj_struct_1b_05_outmix_r16_colgain_lr30e4", cproj_target(CPROJ_OUTMIX, 16, 0.0030, input_gain=True)),
        ("y400_cproj_struct_1b_06_outgroup12_mix_r16_lr30e4", cproj_target(CPROJ_OUTGROUP12_MIX, 16, 0.0030)),
        ("y400_cproj_struct_1b_07_group12_inmix_r16_lr30e4", cproj_target(CPROJ_GROUP12_INMIX, 16, 0.0030)),
        ("y400_cproj_struct_1b_08_plain_r32_lr30e4", cproj_target(CPROJ, 32, 0.0030)),
    ]
    tasks = [(name, write(name, updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y400_cproj_structural_1b_queue.tsv"
    queue_path.write_text(
        "".join(f"{name}\t{path}\t{MAX_ITERS}\t{TOKENS_PER_ITER}\n" for name, path in tasks),
        encoding="utf-8",
    )
    print(queue_path)


if __name__ == "__main__":
    main()
