from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"


BASE = {
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
    "learning_rate": 0.0018,
    "min_lr": 0.00018,
    "warmup_iters": 50,
    "weight_decay": 0.1,
    "optimizer": "muon",
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "dtype": "bfloat16",
    "compile": False,
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


def main() -> None:
    specs = [
        ("y800_muon_100m_01_vanilla", {"method": "baseline"}),
        (
            "y800_muon_100m_02_cproj_l1",
            {"method": "block_fht", "block_fht_targets": ["attn.c_proj"]},
        ),
        (
            "y800_muon_100m_03_v_only_l1",
            {"method": "block_fht", "block_fht_targets": ["attn.c_attn.v"]},
        ),
        (
            "y800_muon_100m_04_qk_headwise_cproj_l1",
            {"method": "block_fht", "block_fht_targets": ["attn.c_attn.qk_headwise", "attn.c_proj"]},
        ),
        (
            "y800_muon_100m_05_mix25_cproj_l1",
            {"method": "block_fht", "block_fht_targets": ["attn.c_attn.qk_mix25_headwise", "attn.c_proj"]},
        ),
    ]
    tasks = [(name, write(name, **updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y800_attn_muon_1pct_queue.tsv"
    queue_path.write_text("".join(f"{name}\t{path}\t763\t131072\n" for name, path in tasks), encoding="utf-8")
    print(queue_path)


if __name__ == "__main__":
    main()
