"""Generate 124M staged bilateral MLP chart optimization configs."""

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
VARIANTS = (
    (60, False),
    (120, False),
    (120, True),
)


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


def make_config(
    start_iter: int,
    freeze_base: bool,
) -> tuple[Path, dict[str, object]]:
    parent = json.loads(PARENT.read_text())
    source_state = "fixedbase" if freeze_base else "movingbase"
    slot = (
        "hiddenblock32_s2_c4_g4_outblock32_s4_c4_g4_"
        f"start{start_iter}_{source_state}_cachevjp"
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
                "y400_mai_v3_mlp_staged_chart_screens/"
                f"{stem}"
            ),
            "hpo_stage": f"post_manifold_{slot}_0p5tpp",
            "ladder_slot": slot,
            "ladder_role": (
                "mlp_manifold_staged_chart_optimization_screen_provisional"
            ),
            "candidate_scope": (
                "identity-initialized two-stage hidden/four-stage output "
                f"fixed-basis c_proj chart enabled at {start_iter}; "
                + (
                    "generated c_proj latent held fixed after activation"
                    if freeze_base
                    else "generated c_proj latent continues training"
                )
            ),
            "block_fht_mlp_cproj_chart_start_iter": start_iter,
            "block_fht_mlp_cproj_chart_freeze_base_at_start": freeze_base,
            "block_fht_mlp_hidden_chart_stop_iter": -1,
            # The complete chart must be exact identity while gradients are
            # held; the previous output-only 0.125 initialization is removed.
            "block_fht_mlp_residual_output_log_gain_init": 0.0,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_staged_chart_screens/"
                f"performance_preflight_{slot}.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "post_oracle_moving_source_interference"
            ),
            "recipe_resolution_dependency": (
                "post-GELU/output fixed-basis charts recover 72.80% of "
                "held-out endpoint source error, but unrestricted joint "
                "training reaches only val 5.7001; paired and independent "
                "pre-GELU structures add no material held-out capacity"
            ),
            "selection_endpoint": (
                "terminal fixed-window NLL against unrestricted bilateral "
                "5.7001, output-only 5.6906, output+activation 5.6832, and "
                "attention-only 5.4918"
            ),
            "staged_chart_optimization": {
                "chart_start_iter": start_iter,
                "chart_identity_before_start": True,
                "residual_output_effective_log_gain_init": 0.0,
                "freeze_generated_cproj_base_at_start": freeze_base,
                "other_model_parameters_continue": True,
                "chart_coordinates_optimizer": "AdamW fallback",
                "generated_cproj_latent_optimizer": "Muon until frozen",
                "learned_dense_basis": False,
                "lora_adapter": False,
            },
            "oracle_evidence": {
                "post_only_five_layer_holdout_recovery": 0.7279548406600952,
                "independent_pre_post_five_layer_holdout_recovery": (
                    0.7311336040496826
                ),
                "independent_increment": 0.0031787633895874,
                "paired_layer6_holdout_recovery": 0.6250009536743164,
                "post_only_layer6_holdout_recovery": 0.6369572281837463,
                "decision": (
                    "retain the post-GELU/output function class and test "
                    "stagewise optimization against a more stable source"
                ),
            },
            "parent_unrestricted_bilateral_config": str(
                PARENT.relative_to(ROOT)
            ),
            "parent_unrestricted_bilateral_config_sha256": sha256(PARENT),
            "screen_only_resolution": (
                "124M/0.5TPP chart-start/source-freeze causal screen on "
                "fixed evaluation windows"
            ),
        }
    )
    path = CONFIG_DIR / f"{stem}_lr24e4.json"
    return path, config


def main() -> None:
    output = []
    for start_iter, freeze_base in VARIANTS:
        path, config = make_config(start_iter, freeze_base)
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "start_iter": start_iter,
                "freeze_base": freeze_base,
                "out_dir": config["out_dir"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ],
            }
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
