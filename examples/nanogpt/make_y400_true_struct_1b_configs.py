from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
REMOTE_RUN_DIR = "/root/userdata/MappingNetworks/runs"
MAX_ITERS = 7630
TOKENS_PER_ITER = 131072

FULL_ATTN_CPROJ = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj", "mlp.c_proj"]

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
            "y400_true_struct_1b_01_tied_cfc_skip_vec",
            {
                "block_fht_cproj_tied_cfc_skip": True,
                "block_fht_cproj_tied_cfc_scale_init": 0.0,
                "block_fht_cproj_tied_cfc_vector": True,
            },
        ),
        (
            "y400_true_struct_1b_02_quarter_diag",
            {
                "block_fht_cproj_quarter_diag": True,
                "block_fht_cproj_quarter_diag_scale_init": 0.0,
                "block_fht_cproj_quarter_diag_init_std": 0.02,
            },
        ),
        (
            "y400_true_struct_1b_03_spectral64_diag",
            {
                "block_fht_cproj_spectral_resid_rank": 64,
                "block_fht_cproj_spectral_resid_scale_init": 0.0,
                "block_fht_cproj_spectral_resid_seed": 17001,
            },
        ),
        (
            "y400_true_struct_1b_04_tied_quarter",
            {
                "block_fht_cproj_tied_cfc_skip": True,
                "block_fht_cproj_tied_cfc_scale_init": 0.0,
                "block_fht_cproj_tied_cfc_vector": True,
                "block_fht_cproj_quarter_diag": True,
                "block_fht_cproj_quarter_diag_scale_init": 0.0,
            },
        ),
        (
            "y400_true_struct_1b_05_spectral128_diag",
            {
                "block_fht_cproj_spectral_resid_rank": 128,
                "block_fht_cproj_spectral_resid_scale_init": 0.0,
                "block_fht_cproj_spectral_resid_seed": 17002,
            },
        ),
        (
            "y400_true_struct_1b_06_tied_spectral64",
            {
                "block_fht_cproj_tied_cfc_skip": True,
                "block_fht_cproj_tied_cfc_scale_init": 0.0,
                "block_fht_cproj_tied_cfc_vector": True,
                "block_fht_cproj_spectral_resid_rank": 64,
                "block_fht_cproj_spectral_resid_scale_init": 0.0,
                "block_fht_cproj_spectral_resid_seed": 17003,
            },
        ),
    ]
    tasks = [(name, write(name, updates)) for name, updates in specs]
    queue_path = CONFIG_DIR / "y400_true_struct_1b_queue.tsv"
    queue_path.write_text(
        "".join(f"{name}\t{path}\t{MAX_ITERS}\t{TOKENS_PER_ITER}\n" for name, path in tasks),
        encoding="utf-8",
    )
    print(queue_path)


if __name__ == "__main__":
    main()
