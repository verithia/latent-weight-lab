"""Generate the 124M activation-only c_proj manifold ablation."""

from __future__ import annotations

import json

from examples.nanogpt.make_y400_mlp_activation_chart_configs import (
    CONFIG_DIR,
    ROOT,
    SOURCE_PATHS,
    git_head,
    sha256,
)


PARENT = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "0p5tpp_lr24e4_provisional.json"
)
STEM = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "actchart4_muonchart154_0p5tpp"
)


def main() -> None:
    config = json.loads(PARENT.read_text())
    config.update(
        {
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_activation_screens/"
                f"{STEM}"
            ),
            "hpo_stage": "post_manifold_cproj_actchart4_0p5tpp",
            "ladder_slot": "cproj_actchart4",
            "ladder_role": (
                "mlp_manifold_activation_only_ablation_provisional"
            ),
            "candidate_scope": (
                "plain generated c_proj plus the trajectory-derived "
                "common/gauge/centered hidden-channel activation chart; "
                "no residual-output chart"
            ),
            "block_fht_mlp_activation_chart": True,
            "block_fht_mlp_activation_chart_channel_scale": 4.0,
            "block_fht_mlp_activation_chart_common_scale": 4.0,
            "block_fht_mlp_activation_chart_gauge_scale": 4.0,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_activation_screens/"
                "performance_preflight_cproj_actchart4.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "paired_channel_manifold_activation_only_ablation"
            ),
            "recipe_resolution_dependency": (
                "compare the chart alone against plain generated c_proj and "
                "against priorities 91-92, which compose it with the selected "
                "fixed-basis residual-output chart"
            ),
            "selection_endpoint": (
                "terminal held-out NLL versus attention-only, plain c_proj, "
                "output-chart-only, and composed activation/output charts"
            ),
            "optimizer_assignment_expected": (
                "matrix-shaped c_proj BlockFHT latents are Muon-owned; "
                "centered channel vector and layerwise common/gauge scalars "
                "use AdamW fallback"
            ),
            "parent_plain_cproj_config": str(PARENT.relative_to(ROOT)),
            "parent_plain_cproj_config_sha256": sha256(PARENT),
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
                "channel_metric": 4.0,
                "common_metric": 4.0,
                "gauge_metric": 4.0,
                "paired_trajectory_csv_sha256": (
                    "5f38bdc851ec321ad97be95076b722a123873a55b1aeca14cf5edd5b407aeb97"
                ),
            },
            "activation_chart_oracle": {
                "analysis_commit": "5b28b54",
                "source_sha256": (
                    "7fb79b5c4fbe0741d1a4506c0b72326b9c65f576603258158f99803e44f1ad86"
                ),
                "csv_sha256": (
                    "a7822c80ee455c89e39dfa4d17d1646f05a5d515f8d8ba9c3f982b59975170f0"
                ),
                "plain_common_gauge_centered_holdout_error_recovery": (
                    0.436851
                ),
                "plain_independent_channels_holdout_error_recovery": (
                    0.445329
                ),
            },
            "screen_only_resolution": (
                "124M/0.5TPP identity-initialized activation-only ablation "
                "on fixed evaluation windows"
            ),
            "prelaunch_provenance_requirements": (
                "record source hashes, resolved config SHA256, data manifest "
                "SHA256, runtime fixed-evaluation digest, literal command, "
                "and exact host-local MFU certificate"
            ),
        }
    )
    path = CONFIG_DIR / f"{STEM}_lr24e4.json"
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "out_dir": config["out_dir"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
