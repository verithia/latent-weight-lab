"""Generate qualified 124M bilateral MLP ``c_proj`` screen configs."""

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
VARIANTS = (
    (1, 8),
    (2, 4),
    (2, 8),
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
    hidden_stages: int, hidden_metric: int
) -> tuple[Path, dict[str, object]]:
    parent = json.loads(PARENT.read_text())
    slot = (
        f"hiddenblock32_s{hidden_stages}_c{hidden_metric}_g4_"
        "outblock32_s4_c4_g4_cachevjp"
    )
    stem = (
        "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
        f"{slot}_muonchart154_0p5tpp"
    )
    hidden_coordinates = hidden_stages * 96 * (32 * 31 // 2)
    output_coordinates = 4 * 24 * (32 * 31 // 2)
    parameters_per_layer = (
        hidden_coordinates + 3072 + output_coordinates + 768
    )
    config = dict(parent)
    config.update(
        {
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_bilateral_screens/"
                f"{stem}"
            ),
            "hpo_stage": f"post_manifold_{slot}_0p5tpp",
            "ladder_slot": slot,
            "ladder_role": (
                "mlp_manifold_bilateral_orientation_screen_provisional"
            ),
            "candidate_scope": (
                f"{hidden_stages}-stage 3072-wide hidden-side and four-stage "
                "768-wide residual-side fixed-basis block-orthogonal c_proj "
                f"chart, hidden coordinate metric {hidden_metric}"
            ),
            "block_fht_mlp_hidden_block_rotation_stages": hidden_stages,
            "block_fht_mlp_hidden_block_rotation_size": 32,
            "block_fht_mlp_hidden_block_rotation_basis_size": 256,
            "block_fht_mlp_hidden_block_rotation_coordinate_scale": float(
                hidden_metric
            ),
            "block_fht_mlp_hidden_block_rotation_seed": 314159,
            "block_fht_mlp_hidden_gain": True,
            "block_fht_mlp_hidden_gain_scale": 4.0,
            "block_fht_mlp_hidden_log_gain_init": 0.0,
            "block_fht_mlp_activation_chart": False,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_bilateral_screens/"
                f"performance_preflight_{slot}.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "post_paper_trajectory_and_bilateral_heldout_oracle"
            ),
            "recipe_resolution_dependency": (
                "five-layer distinct-fit/heldout bilateral oracle; depth one "
                "and two are nearly tied, so bracket depth and hidden chart "
                "metric without changing the selected output-side chart"
            ),
            "selection_endpoint": (
                "terminal fixed-window NLL versus attention-only, plain "
                "generated c_proj, output-only chart, and activation/output "
                "composition; then residual and post-GELU spectra"
            ),
            "optimizer_assignment_expected": (
                "matrix-shaped c_proj BlockFHT latents are Muon-owned; "
                "one-dimensional hidden/output Cayley coordinates and log "
                "gains use the registered AdamW fallback"
            ),
            "parent_output_chart_config": str(PARENT.relative_to(ROOT)),
            "parent_output_chart_config_sha256": sha256(PARENT),
            "bilateral_manifold_structure": {
                "identity_initialization": True,
                "cached_weight_formula": (
                    "D_out R_out W_generated R_hidden^T D_hidden"
                ),
                "cache_semantics": (
                    "one complete charted-weight materialization per optimizer "
                    "step with exact VJP to base latent and both chart sides"
                ),
                "hidden_features": 3072,
                "hidden_stages": hidden_stages,
                "hidden_rotation_coordinates": hidden_coordinates,
                "hidden_log_gain_coordinates": 3072,
                "hidden_coordinate_scale": float(hidden_metric),
                "hidden_log_gain_scale": 4.0,
                "hidden_log_gain_init": 0.0,
                "output_features": 768,
                "output_stages": 4,
                "output_rotation_coordinates": output_coordinates,
                "output_log_gain_coordinates": 768,
                "output_coordinate_scale": 4.0,
                "output_log_gain_scale": 4.0,
                "output_effective_log_gain_init": 0.125,
                "parameters_per_layer": parameters_per_layer,
                "fraction_of_dense_cproj": (
                    parameters_per_layer / (768 * 3072)
                ),
                "learned_dense_basis": False,
                "lora_adapter": False,
            },
            "bilateral_oracle": {
                "analysis_commit": "5caae71",
                "analysis_source_sha256": (
                    "ca23bb14ab8dbde88d0662e94990a5751812608ebcda238a09dd87d47d81ff2f"
                ),
                "csv_sha256": (
                    "ed0612af99c68806d4aef15c76c16b086dfa86eee55395aa6697e3c8eb619ab3"
                ),
                "fit_token_sha256": (
                    "5ffbcbcb14dd284cded97fef7b9e80fbe4656b8c8de7cdf1beff1bcc6669350b"
                ),
                "holdout_token_sha256": (
                    "6f80d31cc9e111edcfe46a71efbaaa3332e651a78c109bba095b919379cd8d4e"
                ),
                "layers": [0, 3, 6, 9, 11],
                "samples_per_split": 2048,
                "output_only_holdout_error_recovery": 0.65360479,
                "hidden_two_stage_holdout_error_recovery": 0.70685036,
                "hidden_one_output_four_holdout_error_recovery": 0.72407537,
                "hidden_two_output_four_holdout_error_recovery": 0.72698910,
                "bilateral_activation_holdout_error_recovery": 0.72808304,
                "per_layer_bilateral_improves_output_only": True,
                "decision": (
                    "the dominant missing chart is hidden-side orientation; "
                    "use a shallow bilateral chart and do not add a learned "
                    "basis or activation chart in this causal screen"
                ),
            },
            "screen_only_resolution": (
                "124M/0.5TPP identity-initialized bilateral c_proj "
                "orientation screen on fixed evaluation windows"
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
    for hidden_stages, hidden_metric in VARIANTS:
        path, config = make_config(hidden_stages, hidden_metric)
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "hidden_stages": hidden_stages,
                "hidden_metric": hidden_metric,
                "out_dir": config["out_dir"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ],
            }
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
