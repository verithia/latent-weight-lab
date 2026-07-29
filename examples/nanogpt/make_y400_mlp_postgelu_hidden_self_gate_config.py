"""Generate the selected layer-0 full-width post-GELU self-mixer screen."""

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
    "postgelu_rotations_hidden_self_l0_0p5tpp"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_mlp_postgelu_hidden_self_screen"
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
    / "124m_mlp_postgelu_hidden_self_direction_result.json"
)
PRIOR_CAUSAL_RESULT = (
    CONFIG_DIR
    / "selection_artifacts"
    / "124m_mlp_postgelu_conditioned_causal_result.json"
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
    prior_causal_result = json.loads(PRIOR_CAUSAL_RESULT.read_text())
    selected_layer = int(
        direction_result["selection"]["selected_layer"]
    )
    if selected_layer != 0:
        raise ValueError(
            f"registered selected layer changed: {selected_layer}"
        )
    matched_parent_val = float(
        prior_causal_result["matched_comparison"][
            "exact_postgelu_frame_parent_val_ce"
        ]
    )
    config.update(
        {
            "out_dir": f"{OUTPUT_ROOT}/{STEM}",
            "hpo_stage": "postgelu_hidden_self_l0_0p5tpp",
            "ladder_slot": "postgelu_hidden_self_untied_fixed_basis_layer0",
            "ladder_role": "mlp_task_aligned_activation_direction_screen",
            "candidate_scope": (
                "matched post-GELU hidden/output frames plus one layer-0 "
                "identity-initialized full-expansion-width, task-selected "
                "untied fixed-basis self-bilinear slope"
            ),
            "block_fht_mlp_postgelu_hidden_self_gate": True,
            "block_fht_mlp_postgelu_hidden_self_gate_scale": 1.0,
            "block_fht_mlp_postgelu_hidden_self_gate_layers": [
                selected_layer
            ],
            "block_fht_mlp_postgelu_hidden_self_gate_basis_block_size": 256,
            "block_fht_mlp_postgelu_hidden_self_gate_condition_basis_seed": (
                271828
            ),
            "block_fht_mlp_postgelu_hidden_self_gate_update_basis_seed": (
                376557
            ),
            "block_fht_mlp_postgelu_hidden_self_gate_output_basis_seed": (
                481286
            ),
            "block_fht_mlp_postgelu_hidden_self_gate_rms_epsilon": 1e-6,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                f"{OUTPUT_ROOT}/performance_preflight_hidden_self_l0.json"
            ),
            "failed_mfu_preflight": None,
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.2,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "postgelu_hidden_self_task_direction_smallest_rung"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation CE versus the exact "
                "matched post-GELU-frame parent, attention-only control, "
                "and best prior MLP structure"
            ),
            "preregistered_decision_rule": {
                "primary_metric": (
                    "terminal fixed-window validation cross entropy at "
                    "update 238"
                ),
                "matched_parent_validation_ce": matched_parent_val,
                "practical_margin_ce": 0.02,
                "promote": (
                    "stable candidate improves on matched parent by at "
                    "least 0.020 CE"
                ),
                "directional_only": (
                    "candidate improvement is positive but below 0.020 CE"
                ),
                "reject": (
                    "candidate ties, regresses, is unstable, or does not "
                    "complete"
                ),
            },
            "optimizer_assignment_expected": (
                "matrix-shaped BlockFHT latents are Muon-owned; static "
                "post-GELU chart coordinates and the selected 3072-wide "
                "dynamic slope use the registered AdamW fallback"
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
                "selected_layer_teacher_direction_cosine": (
                    direction_result[
                        "selected_layer_teacher_direction_cosine"
                    ]
                ),
                "selection_rule": direction_result["selection"]["rule"],
            },
            "prior_causal_result": {
                "path": str(PRIOR_CAUSAL_RESULT.relative_to(ROOT)),
                "sha256": sha256(PRIOR_CAUSAL_RESULT),
            },
            "postgelu_hidden_self_gate": {
                "formula": (
                    "h + Q_out^-1[(Q_update h) * "
                    "(slope * Q_condition rmsnorm(h))]"
                ),
                "location": "after GELU and before c_proj",
                "condition_width": 3072,
                "coordinates_total": 3072,
                "selected_layers": [selected_layer],
                "selected_coordinate_group": "slope",
                "condition_basis_seed": 271828,
                "update_basis_seed": 376557,
                "output_basis_seed": 481286,
                "per_layer_seed_offset": 64,
                "identity_initialization": True,
                "learned_bias": False,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "inference_input_dependent": True,
                "changes_activation_spectrum_before_cproj": True,
                "non_redundant_with_static_cproj": True,
            },
            "screen_only_resolution": (
                "one 124M/0.5TPP causal run after a foreground-polled "
                "20-percent-MFU qualification"
            ),
            "monitoring_policy": (
                "short run is launched and polled directly; no aggregate "
                "watchdog, callback, queue heartbeat, or PRO6 execution"
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
