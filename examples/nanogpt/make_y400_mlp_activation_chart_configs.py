"""Generate the qualified 124M MLP activation-chart screen configs."""

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
    "blockorth32_s4_c4_g4_init0p125_cachevjp_"
    "muonchart154_0p5tpp_lr24e4.json"
)
SOURCE_PATHS = (
    "examples/nanogpt/train.py",
    "examples/nanogpt/model.py",
    "examples/nanogpt/parameter_trajectory.py",
    "latent_weight_lab/block_fht.py",
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


def make_config(metric: int) -> tuple[Path, dict[str, object]]:
    parent = json.loads(PARENT.read_text())
    slot = f"blockorth32_s4_c4_g4_actchart{metric}_cachevjp"
    stem = (
        "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
        f"{slot}_muonchart154_0p5tpp"
    )
    config = dict(parent)
    config.update(
        {
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_activation_screens/"
                f"{stem}"
            ),
            "hpo_stage": f"post_manifold_{slot}_0p5tpp",
            "ladder_slot": slot,
            "ladder_role": (
                "mlp_manifold_activation_output_composition_screen_"
                "provisional"
            ),
            "candidate_scope": (
                "four-stage fixed-basis block-orthogonal c_proj chart "
                "composed with the trajectory-derived common/gauge/centered "
                f"hidden-channel activation chart at metric {metric}"
            ),
            "block_fht_mlp_activation_chart": True,
            "block_fht_mlp_activation_chart_channel_scale": float(metric),
            "block_fht_mlp_activation_chart_common_scale": float(metric),
            "block_fht_mlp_activation_chart_gauge_scale": float(metric),
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_activation_screens/"
                f"performance_preflight_{slot}.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "post_paired_channel_manifold_and_heldout_activation_oracle"
            ),
            "recipe_resolution_dependency": (
                "dense 41-snapshot paired-channel trajectory plus distinct "
                "fit/holdout activation-chart oracle; retain the selected "
                "four-stage output chart and bracket only activation metric"
            ),
            "selection_endpoint": (
                "terminal held-out NLL versus matched attention-only, plain "
                "generated c_proj, four-stage output-chart control, and the "
                "other activation metric; then residual/activation spectra"
            ),
            "optimizer_assignment_expected": (
                "matrix-shaped c_proj BlockFHT latents are Muon-owned; "
                "Cayley coordinates, output gains, centered channel vector, "
                "and layerwise common/gauge scalars use AdamW fallback"
            ),
            "parent_output_chart_config": str(PARENT.relative_to(ROOT)),
            "parent_output_chart_config_sha256": sha256(PARENT),
            "activation_chart_manifold": {
                "coordinates_per_layer": 3074,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "identity_initialization": True,
                "pre_gelu_log_scale": (
                    "common - gauge + centered_channel"
                ),
                "post_gelu_log_scale": (
                    "common + gauge + centered_channel"
                ),
                "centered_channel_constraint": "exact zero mean",
                "channel_metric": float(metric),
                "common_metric": float(metric),
                "gauge_metric": float(metric),
                "paired_trajectory_csv_sha256": (
                    "5f38bdc851ec321ad97be95076b722a123873a55b1aeca14cf5edd5b407aeb97"
                ),
                "paired_trajectory_summary": {
                    "pc1_energy": 0.92273,
                    "pc1_pc2_energy": 0.98880,
                    "terminal_centered_common_energy": 0.66891,
                    "terminal_centered_gauge_energy": 0.33109,
                    "terminal_mean_c_fc_log_norm_change": -0.022068,
                    "terminal_mean_c_proj_log_norm_change": 0.056765,
                    "terminal_independent_radial_capture": 0.05785,
                },
            },
            "activation_chart_oracle": {
                "analysis_commit": "5b28b54",
                "source_sha256": (
                    "7fb79b5c4fbe0741d1a4506c0b72326b9c65f576603258158f99803e44f1ad86"
                ),
                "csv_sha256": (
                    "a7822c80ee455c89e39dfa4d17d1646f05a5d515f8d8ba9c3f982b59975170f0"
                ),
                "fit_token_sha256": (
                    "5ffbcbcb14dd284cded97fef7b9e80fbe4656b8c8de7cdf1beff1bcc6669350b"
                ),
                "holdout_token_sha256": (
                    "6f80d31cc9e111edcfe46a71efbaaa3332e651a78c109bba095b919379cd8d4e"
                ),
                "layers": [0, 3, 6, 9, 11],
                "samples_per_split": 2048,
                "plain_common_gauge_centered_holdout_error_recovery": (
                    0.436851
                ),
                "plain_independent_channels_holdout_error_recovery": (
                    0.445329
                ),
                "blockorth_common_gauge_centered_holdout_error_recovery": (
                    0.443012
                ),
                "blockorth_independent_channels_holdout_error_recovery": (
                    0.455655
                ),
                "decision": (
                    "shared manifold coordinates retain nearly all useful "
                    "independent diagonal freedom; compose them with the "
                    "already useful fixed-basis output orientation chart"
                ),
            },
            "screen_only_resolution": (
                "124M/0.5TPP identity-initialized activation/output "
                "composition screen on fixed evaluation windows"
            ),
            "prelaunch_provenance_requirements": (
                "record source hashes, resolved config SHA256, data manifest "
                "SHA256, runtime fixed-evaluation digest, literal command, "
                "and exact host-local MFU certificate"
            ),
        }
    )
    path = CONFIG_DIR / f"{stem}_lr24e4.json"
    return path, config


def main() -> None:
    output = []
    for metric in (4, 8):
        path, config = make_config(metric)
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "metric": metric,
                "out_dir": config["out_dir"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ],
            }
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
