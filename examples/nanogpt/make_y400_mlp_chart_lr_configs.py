"""Generate 124M chart-specific learning-rate screens."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples" / "nanogpt" / "configs"
PARENT = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "hiddenblock32_s2_c4_g4_outblock32_s4_c4_g4_cachevjp_"
    "muonchart154_0p5tpp_lr24e4.json"
)
SOURCE_PATHS = (
    "examples/nanogpt/train.py",
    "examples/nanogpt/model.py",
    "examples/nanogpt/parameter_trajectory.py",
    "latent_weight_lab/block_fht.py",
)
CHART_LR_SCALES = (4.0, 8.0, 16.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def scale_label(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def make_config(scale: float) -> tuple[Path, dict[str, object]]:
    parent = json.loads(PARENT.read_text())
    label = scale_label(scale)
    slot = (
        "hiddenblock32_s2_c4_g4_outblock32_s4_c4_g4_"
        f"chartlr{label}_cachevjp"
    )
    stem = (
        "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
        f"{slot}_muonchart154_0p5tpp"
    )
    config = dict(parent)
    config.update(
        {
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_chart_lr_screens/"
                f"{stem}"
            ),
            "hpo_stage": f"post_manifold_{slot}_0p5tpp",
            "ladder_slot": slot,
            "ladder_role": (
                "mlp_manifold_chart_optimizer_scale_screen_provisional"
            ),
            "candidate_scope": (
                "unrestricted two-stage hidden/four-stage output fixed-basis "
                f"c_proj chart with {scale:g}x chart-only AdamW learning rate; "
                "Muon matrix latents and non-chart AdamW parameters unchanged"
            ),
            "block_fht_mlp_chart_lr_scale": scale,
            "block_fht_mlp_cproj_chart_start_iter": 0,
            "block_fht_mlp_cproj_chart_freeze_base_at_start": False,
            "block_fht_mlp_hidden_chart_stop_iter": -1,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_chart_lr_screens/"
                f"performance_preflight_{slot}.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "post_staged_chart_coordinate_scale_mismatch"
            ),
            "recipe_resolution_dependency": (
                "fixed-fit held-out oracle coordinates are about 0.10-0.25 "
                "RMS, whereas unrestricted CE training reaches about "
                "0.003-0.005 raw RMS; delayed activation and source freezing "
                "were negative"
            ),
            "selection_endpoint": (
                "terminal fixed-window NLL against unrestricted bilateral "
                "5.7001, output-only 5.6906, output+activation 5.6832, and "
                "attention-only 5.4918"
            ),
            "chart_optimizer_screen": {
                "chart_adamw_lr_scale": scale,
                "base_adamw_lr_scale": parent["muon_adamw_lr_scale"],
                "generated_cproj_optimizer": "Muon",
                "learned_dense_basis": False,
                "lora_adapter": False,
                "causal_question": (
                    "does CE under-traverse an otherwise useful fixed chart?"
                ),
            },
            "staged_chart_terminal_evidence": {
                "start60_movingbase_val": 5.758588790893555,
                "start120_movingbase_val": 5.748360633850098,
                "start120_fixedbase_val": 5.751996994018555,
                "decision": (
                    "moving-source interference is not the primary failure"
                ),
            },
            "parent_unrestricted_bilateral_config": str(
                PARENT.relative_to(ROOT)
            ),
            "parent_unrestricted_bilateral_config_sha256": sha256(PARENT),
            "screen_only_resolution": (
                "124M/0.5TPP chart-only optimizer-scale causal screen on "
                "fixed evaluation windows"
            ),
        }
    )
    path = CONFIG_DIR / f"{stem}_lr24e4.json"
    return path, config


def main() -> None:
    output = []
    for scale in CHART_LR_SCALES:
        path, config = make_config(scale)
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "chart_lr_scale": scale,
                "out_dir": config["out_dir"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ],
            }
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
