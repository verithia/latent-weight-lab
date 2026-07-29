"""Generate the selected layer-1 post-GELU-conditioned MLP screen."""

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
    "postgelu_rotations_matched_0p5tpp_lr24e4.json"
)
STEM = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "postgelu_rotations_postgelucond_untied_l1_0p5tpp"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_mlp_postgelu_conditioned_screen"
)
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/train.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/parameter_trajectory.py",
    "latent_weight_lab/block_fht.py",
)
DIRECTION_RESULT = (
    CONFIG_DIR
    / "selection_artifacts"
    / "124m_mlp_postgelu_conditioned_direction_result.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    config = json.loads(PARENT.read_text())
    direction_result = json.loads(DIRECTION_RESULT.read_text())
    selected_layer = int(
        direction_result["selection"]["selected_layer"]
    )
    if selected_layer != 1:
        raise ValueError(
            f"registered selected layer changed: {selected_layer}"
        )
    config.update(
        {
            "out_dir": f"{OUTPUT_ROOT}/{STEM}",
            "hpo_stage": "postgelu_conditioned_untied_l1_0p5tpp",
            "ladder_slot": (
                "postgelu_conditioned_untied_fixed_basis_layer1"
            ),
            "ladder_role": (
                "mlp_activation_conditioned_nonlinear_direction_screen"
            ),
            "candidate_scope": (
                "matched post-GELU hidden/output frames plus one layer-1 "
                "identity-initialized activation-conditioned untied-basis "
                "bilinear output slope"
            ),
            "block_fht_mlp_residual_conditioned_output_gate": True,
            "block_fht_mlp_residual_conditioned_output_gate_scale": 1.0,
            "block_fht_mlp_residual_conditioned_output_gate_layers": [1],
            "block_fht_mlp_residual_conditioned_output_gate_bias": False,
            "block_fht_mlp_residual_conditioned_output_gate_fixed_basis": (
                True
            ),
            "block_fht_mlp_residual_conditioned_output_gate_untied_bases": (
                True
            ),
            "block_fht_mlp_residual_conditioned_output_gate_basis_block_size": (
                256
            ),
            "block_fht_mlp_residual_conditioned_output_gate_basis_seed": (
                271828
            ),
            "block_fht_mlp_residual_conditioned_output_gate_update_basis_seed": (
                376557
            ),
            "block_fht_mlp_residual_conditioned_output_gate_output_basis_seed": (
                481286
            ),
            "block_fht_mlp_conditioned_output_gate_source": "postgelu",
            "block_fht_mlp_conditioned_output_gate_projection_seed": 586015,
            "block_fht_mlp_conditioned_output_gate_rms_epsilon": 1e-6,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                f"{OUTPUT_ROOT}/"
                "performance_preflight_postgelucond_untied_l1.json"
            ),
            "failed_mfu_preflight": None,
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.2,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "postgelu_conditioned_direction_smallest_rung"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation CE versus the exact "
                "matched post-GELU-frame parent, attention-only, and the "
                "best prior MLP structure"
            ),
            "optimizer_assignment_expected": (
                "matrix-shaped BlockFHT latents are Muon-owned; static "
                "post-GELU chart coordinates and the selected dynamic slope "
                "use the registered AdamW fallback"
            ),
            "parent_config": str(PARENT.relative_to(ROOT)),
            "parent_config_sha256": sha256(PARENT),
            "direction_result": {
                "path": str(DIRECTION_RESULT.relative_to(ROOT)),
                "sha256": sha256(DIRECTION_RESULT),
                "selected_layer": selected_layer,
                "selected_layer_task_ce_fit_holdout_cosine": (
                    direction_result["selection"][
                        "selected_layer_cosine"
                    ]
                ),
                "selection_rule": direction_result["selection"]["rule"],
            },
            "activation_conditioned_gate": {
                "formula": (
                    "u + Q_out^-1[(Q_update u) * "
                    "(slope * Q_condition rmsnorm(P_fixed postgelu))]"
                ),
                "coordinates_total": 768,
                "selected_layers": [selected_layer],
                "selected_coordinate_group": "slope",
                "fixed_projection": (
                    "signed four-to-one expansion-to-residual projection"
                ),
                "condition_basis_seed": 271828,
                "update_basis_seed": 376557,
                "output_basis_seed": 481286,
                "projection_seed": 586015,
                "per_layer_seed_offset": 64,
                "identity_initialization": True,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "inference_input_dependent": True,
                "non_redundant_with_static_cproj": True,
            },
            "screen_only_resolution": (
                "one 124M/0.5TPP causal run after a foreground-polled "
                "20-percent-MFU qualification"
            ),
            "prelaunch_provenance_requirements": (
                "record commit, source/config/dataset/fixed-eval hashes, "
                "literal command, exact host-local MFU certificate, status, "
                "log, and terminal checkpoint"
            ),
        }
    )
    path = CONFIG_DIR / f"{STEM}_lr24e4.json"
    path.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
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
