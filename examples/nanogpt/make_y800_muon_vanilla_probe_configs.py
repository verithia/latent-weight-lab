from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"


def main() -> None:
    name = "y800_probe_muon_vanilla_1b_lr18e4_ckpt"
    cfg = {
        "data_dir": "/root/userdata/MappingNetworks/data/finewebedu_2b",
        "out_dir": f"{REMOTE_RUN_DIR}/{name}",
        "init_from": "scratch",
        "method": "baseline",
        "max_iters": 7630,
        "lr_decay_iters": 7630,
        "eval_interval": 500,
        "eval_iters": 5,
        "save_checkpoint": True,
        "batch_size": 4,
        "gradient_accumulation_steps": 32,
        "block_size": 1024,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "vocab_size": 50304,
        "learning_rate": 0.0018,
        "min_lr": 0.00018,
        "warmup_iters": 100,
        "weight_decay": 0.1,
        "optimizer": "muon",
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "dtype": "bfloat16",
        "compile": False,
    }
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    queue = CONFIG_DIR / "y800_muon_vanilla_probe_queue.tsv"
    queue.write_text(f"{name}\t{REMOTE_CONFIG_DIR}/{path.name}\t7630\t131072\n", encoding="utf-8")
    print(queue)


if __name__ == "__main__":
    main()
