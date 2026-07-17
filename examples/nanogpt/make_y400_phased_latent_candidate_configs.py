from __future__ import annotations

"""Generate gated 350M BlockFHT candidates only; this script never launches."""

import json
import math
from pathlib import Path

DATA_DIR = "/root/userdata/MappingNetworks/data/finewebedu_2b"
OUT_DIR = "/root/userdata/MappingNetworks/outputs/y400_phased_latent_candidates"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
QUEUE_PATH = Path(__file__).resolve().parent / "y400_phased_latent_candidates_queue.tsv"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"
ACTIVE_PARAMS_350M = 354_871_296
TOKENS_PER_ITER = 262_144
BASE_TARGETS = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj"]
ACCEPTANCE = "speed>=85% vanilla (>=125351 tok/s); >=15% VRAM headroom; loss<=vanilla+0.05 CE"


def config(name: str, targets: list[str], planned_tokens: int, gate: str, launch_ready: bool) -> dict:
    max_iters = math.ceil(planned_tokens / TOKENS_PER_ITER)
    return {
        "data_dir": DATA_DIR, "out_dir": f"{OUT_DIR}/{name}", "method": "block_fht",
        "model_tier": "350m", "n_layer": 24, "n_embd": 1024, "n_head": 16,
        "estimated_active_params": ACTIVE_PARAMS_350M, "planned_tokens": planned_tokens,
        "planned_tpp": planned_tokens / ACTIVE_PARAMS_350M, "tokens_per_iter": TOKENS_PER_ITER,
        "scheduled_tokens": max_iters * TOKENS_PER_ITER, "max_iters": max_iters,
        "batch_size": 32, "block_size": 1024, "gradient_accumulation_steps": 8,
        "optimizer": "muon", "learning_rate": 0.0024, "min_lr": 0.00024,
        "muon_adamw_lr_scale": 0.3, "warmup_iters": max(10, max_iters // 100), "lr_decay_iters": max_iters,
        "dtype": "bfloat16", "device": "cuda", "compile": False,
        "block_fht_targets": targets, "block_fht_latent_ratio": 0.01, "block_fht_layers": 2,
        "block_fht_match_gpt_init": True, "block_fht_cache_weights": True,
        "perf_profile": False, "perf_log_interval": 10, "log_interval": 10,
        "eval_interval": 10_000 if gate == "speed" else max_iters,
        "eval_iters": 10 if gate == "speed" else 50,
        "save_checkpoint": False, "checkpoint_history": False,
        "candidate_gate": gate, "acceptance_requirements": ACCEPTANCE,
        "launch_ready": launch_ready,
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    phases = [
        ("attn", BASE_TARGETS, "READY"),
        ("attn_cfc", BASE_TARGETS + ["mlp.c_fc"], "BLOCKED_ON_ATTN_GATE"),
        ("attn_cfc_cproj", BASE_TARGETS + ["mlp.c_fc", "mlp.c_proj"], "BLOCKED_ON_CFC_GATE"),
    ]
    rows = []
    for phase, targets, status in phases:
        for gate, tokens in (("speed", 50_000_000), ("terminal_eval_loss", int(ACTIVE_PARAMS_350M * 0.5))):
            name = f"y400_latent_350m_{phase}_{gate}"
            # 350M configs are complete/runnable recipes; ordering is enforced by queue status.
            cfg = config(name, targets, tokens, gate, launch_ready=True)
            (CONFIG_DIR / f"{name}.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
            rows.append((name, phase, gate, status, "350m", "true", ACCEPTANCE, f"{REMOTE_CONFIG_DIR}/{name}.json"))
            if phase == "attn" and gate == "speed":
                profile_name = "y400_latent_350m_attn_profile_headroom"
                profile = dict(cfg)
                profile.update(
                    out_dir=f"{OUT_DIR}/{profile_name}",
                    perf_profile=True,
                    perf_warmup_iters=5,
                    perf_log_interval=10,
                    candidate_gate="profile_headroom",
                )
                (CONFIG_DIR / f"{profile_name}.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
                rows.append((profile_name, phase, "profile_headroom", "BLOCKED_ON_SPEED_PASS", "350m", "true", ACCEPTANCE, f"{REMOTE_CONFIG_DIR}/{profile_name}.json"))
            if phase == "attn_cfc" and gate == "speed":
                profile_name = "y400_latent_350m_attn_cfc_profile_headroom"
                profile = dict(cfg)
                profile.update(
                    out_dir=f"{OUT_DIR}/{profile_name}",
                    perf_profile=True,
                    perf_warmup_iters=5,
                    perf_log_interval=10,
                    candidate_gate="profile_headroom",
                )
                (CONFIG_DIR / f"{profile_name}.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
                rows.append((profile_name, phase, "profile_headroom", "BLOCKED_ON_CFC_SPEED_PASS", "350m", "true", ACCEPTANCE, f"{REMOTE_CONFIG_DIR}/{profile_name}.json"))
    for tier in ("690m", "1b"):
        rows.append((f"y400_latent_{tier}_roadmap", "roadmap", "none", "BLOCKED_ON_MUON_BASELINE_ACCEPTANCE", tier, "false", "Muon baseline acceptance required; final LR intentionally unset", ""))
    QUEUE_PATH.write_text(
        "name\tphase\tgate\tstatus\ttier\tlaunch_ready\tacceptance_requirements\tconfig\n"
        + "".join("\t".join(row) + "\n" for row in rows)
    )
    print(f"wrote {len(rows)} phased latent-candidate queue rows")


if __name__ == "__main__":
    main()
