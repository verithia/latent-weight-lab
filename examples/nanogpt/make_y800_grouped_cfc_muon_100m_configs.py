from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"


ATTN = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj"]

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
    if "learning_rate" in updates and "min_lr" not in updates:
        cfg["min_lr"] = float(updates["learning_rate"]) * 0.1
    cfg["out_dir"] = f"{REMOTE_RUN_DIR}/{name}"
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return f"{REMOTE_CONFIG_DIR}/{path.name}"


def grouped(target: str, bias_init: float | None = None) -> dict[str, object]:
    updates: dict[str, object] = {"block_fht_targets": ATTN + [target]}
    if bias_init is not None:
        updates["block_fht_ffn_pregelu_bias"] = True
        updates["block_fht_ffn_pregelu_bias_init"] = bias_init
    return updates


def main() -> None:
    specs = [
        ("y800_groupcfc_100m_01_attn_only_lr22e4", {"block_fht_targets": ATTN}),
        ("y800_groupcfc_100m_02_g12_lr22e4", grouped("mlp.c_fc_group12")),
        ("y800_groupcfc_100m_03_g16_lr22e4", grouped("mlp.c_fc_group16")),
        ("y800_groupcfc_100m_04_g24_lr22e4", grouped("mlp.c_fc_group24")),
        ("y800_groupcfc_100m_05_g12_bneg025_lr22e4", grouped("mlp.c_fc_group12", -0.25)),
        ("y800_groupcfc_100m_06_g12_bneg050_lr22e4", grouped("mlp.c_fc_group12", -0.50)),
        ("y800_groupcfc_100m_07_g16_bneg025_lr22e4", grouped("mlp.c_fc_group16", -0.25)),
        ("y800_groupcfc_100m_08_g16_bneg050_lr22e4", grouped("mlp.c_fc_group16", -0.50)),
    ]
    tasks = [(name, write(name, **updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y800_grouped_cfc_muon_100m_queue.tsv"
    queue_path.write_text("".join(f"{name}\t{path}\t763\t131072\n" for name, path in tasks), encoding="utf-8")
    print(queue_path)


if __name__ == "__main__":
    main()
