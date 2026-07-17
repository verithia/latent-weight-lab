from __future__ import annotations

"""Generate the baseline-only, performance-first Y400 ladder; never launch it."""

import json
import math
from pathlib import Path

SPEED_DATA_DIR = "/root/userdata/MappingNetworks/data/finewebedu_2b"
LONG_DATA_DIR = "/root/userdata/MappingNetworks/data/finewebedu_20b"
OUT_DIR = "/root/userdata/MappingNetworks/outputs/y400_scaled_ladder"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
QUEUE_PATH = Path(__file__).resolve().parent / "y400_scaled_ladder_queue.tsv"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"

TIERS = {
    "350m": {"n_layer": 24, "n_embd": 1024, "n_head": 16},
    "690m": {"n_layer": 32, "n_embd": 1280, "n_head": 20},
    "1b": {"n_layer": 32, "n_embd": 1536, "n_head": 24},
}
HPO_SHAPES = {
    "350m": (32, 8),
    "690m": (16, 16),
    "1b": (16, 16),
}


def estimate_active_params(n_layer: int, n_embd: int, vocab_size: int = 50304) -> int:
    return int(vocab_size * n_embd + 1024 * n_embd + n_layer * (12 * n_embd * n_embd + 13 * n_embd) + 2 * n_embd)


def tokens_for_tpp(tier: str, tpp: float) -> int:
    model = TIERS[tier]
    return int(estimate_active_params(model["n_layer"], model["n_embd"]) * tpp)


def base_config(name: str, tier: str, planned_tokens: int, optimizer: str, *, data_dir: str, batch_size: int = 8, grad_accum: int = 32) -> dict:
    model = TIERS[tier]
    active = estimate_active_params(model["n_layer"], model["n_embd"])
    tokens_per_iter = batch_size * 1024 * grad_accum
    max_iters = max(1, math.ceil(planned_tokens / tokens_per_iter))
    lr = 3e-4 if optimizer == "adamw" else 1.8e-3
    return {
        "data_dir": data_dir, "out_dir": f"{OUT_DIR}/{name}", "method": "baseline",
        "n_layer": model["n_layer"], "n_embd": model["n_embd"], "n_head": model["n_head"], "model_tier": tier,
        "estimated_active_params": active, "planned_tokens": planned_tokens, "planned_tpp": planned_tokens / active,
        "tokens_per_iter": tokens_per_iter, "scheduled_tokens": max_iters * tokens_per_iter, "max_iters": max_iters,
        "eval_interval": max(100, max_iters // 20), "eval_iters": 50, "batch_size": batch_size, "block_size": 1024,
        "gradient_accumulation_steps": grad_accum, "optimizer": optimizer, "learning_rate": lr, "min_lr": lr * 0.1,
        "muon_adamw_lr_scale": 0.2, "warmup_iters": max(10, max_iters // 100), "lr_decay_iters": max_iters,
        "save_checkpoint": False, "checkpoint_history": False, "dtype": "bfloat16", "device": "cuda", "compile": False,
    }


def hpo_config(name: str, tier: str, planned_tokens: int, optimizer: str, *, data_dir: str) -> dict:
    """Use the selected, same-global-batch shape for every HPO/confirmation run."""
    batch_size, grad_accum = HPO_SHAPES[tier]
    return base_config(name, tier, planned_tokens, optimizer, data_dir=data_dir, batch_size=batch_size, grad_accum=grad_accum)


def retokenize(cfg: dict, batch_size: int, grad_accum: int) -> dict:
    out = dict(cfg); out["batch_size"] = batch_size; out["gradient_accumulation_steps"] = grad_accum
    out["tokens_per_iter"] = batch_size * out["block_size"] * grad_accum
    out["max_iters"] = max(1, math.ceil(out["planned_tokens"] / out["tokens_per_iter"]))
    out["scheduled_tokens"] = out["max_iters"] * out["tokens_per_iter"]
    out["warmup_iters"] = max(10, out["max_iters"] // 100); out["lr_decay_iters"] = out["max_iters"]
    return out


def write(name: str, cfg: dict) -> None:
    (CONFIG_DIR / f"{name}.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    def add(name: str, phase: str, cfg: dict) -> None:
        write(name, cfg)
        rows.append((name, phase, cfg["model_tier"], cfg["optimizer"], cfg["planned_tokens"], f"{cfg['planned_tpp']:.6f}", f"{REMOTE_CONFIG_DIR}/{name}.json"))

    # CE synchronization is deliberately short but has enough periodic logs/evals
    # to prove a clean baseline patch before timing or HPO.
    canary = base_config("y400_canary_350m_ce_sync_muon", "350m", 400 * 8 * 1024 * 32, "muon", data_dir=SPEED_DATA_DIR)
    canary.update(perf_profile=False, log_interval=10, eval_interval=50, eval_iters=20, muon_adamw_lr_scale=0.2, canary="ce_sync_fixed_baseline_muon")
    add("y400_canary_350m_ce_sync_muon", "phase0_ce_sync_canary", canary)

    speed_tokens = 50_000_000
    for tier in TIERS:
        headline = base_config(f"y400_speed_{tier}_vanilla_muon", tier, speed_tokens, "muon", data_dir=SPEED_DATA_DIR)
        headline.update(perf_profile=False, perf_log_interval=10, log_interval=10, eval_interval=10_000, eval_iters=10, timing_mode="headline_no_profile")
        add(f"y400_speed_{tier}_vanilla_muon", "phase1_headline_speed", headline)
        profiled = dict(headline, out_dir=f"{OUT_DIR}/y400_speed_{tier}_vanilla_muon_profile", perf_profile=True, perf_warmup_iters=5, timing_mode="later_breakdown_profile")
        add(f"y400_speed_{tier}_vanilla_muon_profile", "phase1_profiled_breakdown", profiled)
        for batch, accum in ((8, 32), (16, 16), (32, 8)):
            name = f"y400_speed_{tier}_muon_headline_b{batch}ga{accum}"
            bracket = retokenize(dict(headline, out_dir=f"{OUT_DIR}/{name}"), batch, accum)
            bracket.update(timing_mode="headline_no_profile_same_global_batch")
            add(name, "phase1_microbatch_bracket", bracket)

    half_tpp = tokens_for_tpp("350m", 0.5)
    for lr in (1.2e-3, 1.8e-3, 2.4e-3):
        name = f"y400_hpo_350m_0p5tpp_muon_lr{int(lr * 1e4):02d}e4"
        cfg = hpo_config(name, "350m", half_tpp, "muon", data_dir=SPEED_DATA_DIR)
        cfg.update(learning_rate=lr, min_lr=lr * 0.1, muon_adamw_lr_scale=0.2, hpo_phase="successive_halving_lr")
        add(name, "phase2_350m_muon_lr_0p5tpp", cfg)
    for scale in (0.1, 0.2, 0.3):
        name = f"y400_hpo_350m_0p5tpp_muon_scale{int(scale * 10):01d}"
        cfg = hpo_config(name, "350m", half_tpp, "muon", data_dir=SPEED_DATA_DIR)
        cfg.update(learning_rate=2.4e-3, min_lr=2.4e-4, muon_adamw_lr_scale=scale, hpo_phase="successive_halving_fallback_scale", hpo_selected_lr=0.0024, launch_ready=True)
        add(name, "phase3_350m_scale_selected_lr24e4", cfg)
    for lr in (2e-4, 3e-4, 4e-4):
        name = f"y400_hpo_350m_0p5tpp_adamw_lr{int(lr * 1e5):02d}e5"
        cfg = hpo_config(name, "350m", half_tpp, "adamw", data_dir=SPEED_DATA_DIR)
        cfg.update(learning_rate=lr, min_lr=lr * 0.1, hpo_phase="successive_halving_adamw_lr")
        add(name, "phase2_350m_adamw_lr_0p5tpp", cfg)

    for tpp in (5, 10):
        name = f"y400_350m_muon_{tpp}tpp_winner_template"
        cfg = hpo_config(name, "350m", tokens_for_tpp("350m", tpp), "muon", data_dir=LONG_DATA_DIR)
        cfg.update(hpo_selected_lr="PLACEHOLDER: phase2 winner", hpo_selected_fallback_scale="PLACEHOLDER: phase3 winner", launch_ready=False, hpo_phase="winner_template")
        add(name, f"phase4_{tpp}tpp_winner_template_PENDING", cfg)

    name = "y400_350m_muon_5tpp_winner"
    cfg = hpo_config(name, "350m", tokens_for_tpp("350m", 5), "muon", data_dir=SPEED_DATA_DIR)
    cfg.update(
        learning_rate=2.4e-3,
        min_lr=2.4e-4,
        muon_adamw_lr_scale=0.3,
        eval_interval=250,
        eval_iters=50,
        hpo_phase="selected_recipe_confirmation",
        hpo_selected_lr=0.0024,
        hpo_selected_fallback_scale=0.3,
        launch_ready=True,
        confirmation_data_protocol="2B-shard random-offset-with-replacement confirmation",
    )
    add(name, "phase4_5tpp_selected_recipe_confirmation_2b_replacement", cfg)

    for tier in ("690m", "1b"):
        for lr in (1.6e-3, 1.8e-3, 2.0e-3):
            name = f"y400_hpo_{tier}_0p5tpp_muon_transfer_lr{int(lr * 1e4):02d}e4"
            cfg = hpo_config(name, tier, tokens_for_tpp(tier, 0.5), "muon", data_dir=SPEED_DATA_DIR)
            cfg.update(learning_rate=lr, min_lr=lr * 0.1, muon_adamw_lr_scale=0.2, hpo_phase="narrow_transferred_lr")
            add(name, f"phase5_{tier}_narrow_transfer_lr_0p5tpp", cfg)

    # Selected-shape profile diagnostics are intentionally hand-curated configs;
    # retain their queue entries when this generator refreshes the ladder.
    for name in (
        "y400_speed_350m_muon_selected_profile_b32ga8",
        "y400_speed_690m_muon_selected_profile_b16ga16",
        "y400_speed_1b_muon_selected_profile_b16ga16",
    ):
        cfg = json.loads((CONFIG_DIR / f"{name}.json").read_text())
        rows.append((name, "phase1_selected_profile_diagnostic", cfg["model_tier"], cfg["optimizer"], cfg["planned_tokens"], f"{cfg['planned_tpp']:.6f}", f"{REMOTE_CONFIG_DIR}/{name}.json"))
    QUEUE_PATH.write_text("name\tphase\ttier\toptimizer\tplanned_tokens\tplanned_tpp\tconfig\n" + "".join("\t".join(map(str, row)) + "\n" for row in rows))
    print(f"wrote {len(rows)} baseline configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
