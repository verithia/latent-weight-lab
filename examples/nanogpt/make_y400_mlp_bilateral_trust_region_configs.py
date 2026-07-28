"""Generate 124M local-trust-region bilateral MLP screen configs."""

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
STOP_ITERS = (30, 60, 120)


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


def make_config(stop_iter: int) -> tuple[Path, dict[str, object]]:
    parent = json.loads(PARENT.read_text())
    slot = (
        "hiddenblock32_s2_c4_g4_stop"
        f"{stop_iter}_outblock32_s4_c4_g4_cachevjp"
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
                "y400_mai_v3_mlp_bilateral_trust_region_screens/"
                f"{stem}"
            ),
            "hpo_stage": f"post_manifold_{slot}_0p5tpp",
            "ladder_slot": slot,
            "ladder_role": (
                "mlp_manifold_local_trust_region_screen_provisional"
            ),
            "candidate_scope": (
                "the selected two-stage hidden/four-stage output fixed-basis "
                "bilateral c_proj chart, with hidden-side updates stopped at "
                f"iteration {stop_iter}"
            ),
            "block_fht_mlp_hidden_chart_stop_iter": stop_iter,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_bilateral_trust_region_screens/"
                f"performance_preflight_{slot}.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "post_bilateral_terminal_local_trust_region"
            ),
            "recipe_resolution_dependency": (
                "the unrestricted bilateral chart is slightly better than "
                "output-only at step 60 but worse by step 120 and terminal; "
                "the dense 41-snapshot trajectory is smooth but curved, so "
                "test whether an early local tangent should be held fixed"
            ),
            "selection_endpoint": (
                "terminal fixed-window NLL against unrestricted bilateral "
                "5.7001, output-only 5.6906, output+activation 5.6832, and "
                "attention-only 5.4918"
            ),
            "hidden_chart_trust_region": {
                "start_iter": 0,
                "stop_iter": stop_iter,
                "stop_semantics": (
                    "updates 0 through stop_iter-1 are applied; hidden "
                    "rotation/gain gradients are then None so AdamW applies "
                    "neither updates nor decoupled weight decay"
                ),
                "base_and_output_chart_continue": True,
                "optimizer_state_identity_preserved": True,
                "learned_dense_basis": False,
                "lora_adapter": False,
            },
            "prior_unrestricted_bilateral_result": {
                "terminal_val": 5.700072765350342,
                "step_60_val": 6.3521,
                "step_120_val": 5.9372,
                "step_180_val": 5.7642,
                "hidden_coordinate_raw_rms": 0.005094465799629688,
                "hidden_coordinate_effective_rms": (
                    4.0 * 0.005094465799629688
                ),
                "output_coordinate_raw_rms": 0.004395646974444389,
            },
            "parent_unrestricted_bilateral_config": str(
                PARENT.relative_to(ROOT)
            ),
            "parent_unrestricted_bilateral_config_sha256": sha256(PARENT),
            "screen_only_resolution": (
                "124M/0.5TPP local tangent-duration screen on the fixed "
                "bilateral chart and fixed evaluation windows"
            ),
        }
    )
    path = CONFIG_DIR / f"{stem}_lr24e4.json"
    return path, config


def main() -> None:
    output = []
    for stop_iter in STOP_ITERS:
        path, config = make_config(stop_iter)
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "stop_iter": stop_iter,
                "out_dir": config["out_dir"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ],
            }
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
