"""Generate the preregistered late-band c_proj LWT 124M/5TPP config."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_cfcdecay1_cprojdecay0p5_5tpp_lr24e4.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cfc_late_cproj_lwt_5tpp_plan.json"
OUTPUT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_cfc_latecproj_lwt_5tpp_lr24e4.json"
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/muon_matched_givens.py",
    "examples/nanogpt/train.py",
    "examples/nanogpt/mfu_preflight.py",
    "examples/nanogpt/test_mlp_cproj_layer_allocation.py",
    "latent_weight_lab/block_fht.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    config = json.loads(PARENT.read_text())
    root = (
        "/mnt/ssd-data/orj/MappingNetworks/outputs/"
        "pro6_mai_v3_124m_repairedfullattn_plus_cfc_latecproj_lwt_5tpp"
    )
    config.update(
        {
            "out_dir": f"{root}/scientific",
            "hpo_stage": "repaired_attention_cfc_late_cproj_lwt_124m_5tpp",
            "ladder_slot": "124m_5tpp_cfc_all_cproj_layers8_11",
            "ladder_role": "same_gauge_layerwise_mlp_allocation",
            "candidate_scope": (
                "Repaired attention and accepted c_fc in all layers; "
                "ordinary dense c_proj in layers 0-7 and the accepted "
                "hidden64+residual24 decay-0.5 procedural c_proj only in "
                "layers 8-11, all trained jointly from initialization."
            ),
            "block_fht_mlp_cproj_muon_matched_givens_layers": [8, 9, 10, 11],
            "checkpoint_wall_clock_seconds": 7200,
            "registered_plan": str(PLAN.relative_to(ROOT)),
            "registered_plan_sha256": sha256(PLAN),
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": f"{root}/performance_preflight.json",
            "mfu_measurement_protocol": (
                "foreground exact scientific config, one warmup plus eight "
                "timed real updates on PRO6 GPU0; verify dense c_proj layers "
                "0-7, procedural c_proj layers 8-11, native BlockFHT, and "
                "all finite losses; no watchdog"
            ),
            "monitoring_policy": (
                "one idempotent terminal-only watchdog; completion or "
                "actionable error/stall callbacks only; no milestones, "
                "heartbeats, or duplicate terminal messages"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation CE <=3.6478 and no fixed "
                "curve point more than 0.005 worse than the all-layer "
                "procedural control"
            ),
            "operator_override": (
                "2026-08-06: user requires 1-2h runs to emit only terminal "
                "or error callbacks; this run tests same-gauge LWT allocation. "
                "The inherited checkpoint interval is normalized to the "
                "registered deterministic-resume invariant of 7200 seconds."
            ),
            "launch_ready": True,
            "launch_block_reason": None,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": "same_gauge_late_cproj_lwt",
            "screen_only": False,
            "terminal_eval_required": True,
        }
    )
    OUTPUT.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path": str(OUTPUT.relative_to(ROOT)),
                "sha256": sha256(OUTPUT),
                "implementation_commit": config["implementation_commit"],
                "plan_sha256": config["registered_plan_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
