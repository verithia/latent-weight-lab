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


def write(name: str, **updates) -> str:
    cfg = dict(BASE)
    cfg.update(updates)
    cfg["out_dir"] = f"{REMOTE_RUN_DIR}/{name}"
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return f"{REMOTE_CONFIG_DIR}/{path.name}"


def cproj_ratio(ratio: float, rank: int = 0) -> dict:
    cfg = {
        "block_fht_targets": ATTN + [CPROJ],
        "block_fht_latent_ratios": {CPROJ: ratio},
    }
    if rank:
        cfg["block_fht_cproj_lowrank_rank"] = rank
    return cfg


def main() -> None:
    specs = [
        ("y800_cproj_lowratio_100m_01_plain_r0025", cproj_ratio(0.0025)),
        ("y800_cproj_lowratio_100m_02_plain_r0050", cproj_ratio(0.0050)),
        ("y800_cproj_lowratio_100m_03_plain_r0075", cproj_ratio(0.0075)),
        ("y800_cproj_lowratio_100m_04_plain_r0025_lora16", cproj_ratio(0.0025, 16)),
        ("y800_cproj_lowratio_100m_05_plain_r0050_lora16", cproj_ratio(0.0050, 16)),
        ("y800_cproj_lowratio_100m_06_plain_r0075_lora16", cproj_ratio(0.0075, 16)),
    ]
    tasks = [(name, write(name, **updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y800_cproj_lowratio_100m_queue.tsv"
    queue_path.write_text("".join(f"{name}\t{path}\t763\t131072\n" for name, path in tasks), encoding="utf-8")
    print(queue_path)


if __name__ == "__main__":
    main()
