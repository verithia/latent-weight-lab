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
    "learning_rate": 0.0018,
    "min_lr": 0.00018,
    "warmup_iters": 50,
    "weight_decay": 0.1,
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
        ("y800_attn_qksameseed_100m_01_global_l1", {"block_fht_targets": ["attn.c_attn.qk_sameseed"]}),
        (
            "y800_attn_qksameseed_100m_02_headwise_l1",
            {"block_fht_targets": ["attn.c_attn.qk_sameseed_headwise"]},
        ),
        (
            "y800_attn_qksameseed_100m_03_global_cproj_l1",
            {"block_fht_targets": ["attn.c_attn.qk_sameseed", "attn.c_proj"]},
        ),
        (
            "y800_attn_qksameseed_100m_04_headwise_cproj_l1",
            {"block_fht_targets": ["attn.c_attn.qk_sameseed_headwise", "attn.c_proj"]},
        ),
    ]
    tasks = [(name, write(name, **updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y800_attn_qk_sameseed_1pct_queue.tsv"
    queue_path.write_text("".join(f"{name}\t{path}\t763\t131072\n" for name, path in tasks), encoding="utf-8")
    print(queue_path)


if __name__ == "__main__":
    main()
