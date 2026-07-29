"""Register the selected two-head layer-0 post-GELU mixer screen."""

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
DIRECTION_RESULT = (
    CONFIG_DIR
    / "selection_artifacts"
    / "124m_mlp_postgelu_hidden_multihead_direction_result.json"
)
SINGLE_HEAD_CAUSAL_RESULT = (
    CONFIG_DIR
    / "selection_artifacts"
    / "124m_mlp_postgelu_hidden_self_causal_result.json"
)
STEM = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "postgelu_rotations_hidden_multihead2_l0_0p5tpp"
)
OUTPUT_ROOT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_v3_mlp_postgelu_hidden_multihead_screen"
)
SOURCE_PATHS = (
    "csrc/block_fht_ext.cpp",
    "csrc/block_fht_ext_cuda.cu",
    "examples/nanogpt/model.py",
    "examples/nanogpt/train.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/parameter_trajectory.py",
    "latent_weight_lab/__init__.py",
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
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    config = json.loads(PARENT.read_text())
    direction = json.loads(DIRECTION_RESULT.read_text())
    single_head = json.loads(SINGLE_HEAD_CAUSAL_RESULT.read_text())
    if not direction["selection"]["all_preregistered_floors_pass"]:
        raise ValueError("registered two-head direction screen did not pass")
    if int(direction["mechanism"]["heads"]) != 2:
        raise ValueError("registered direction result is not exactly two heads")
    selected_layer = 0
    matched_parent_val = float(
        single_head["matched_comparison"][
            "exact_postgelu_frame_parent_val_ce"
        ]
    )
    config.update(
        {
            "out_dir": f"{OUTPUT_ROOT}/{STEM}",
            "hpo_stage": "postgelu_hidden_multihead2_l0_0p5tpp",
            "ladder_slot": (
                "postgelu_hidden_multihead2_untied_fixed_basis_layer0"
            ),
            "ladder_role": (
                "mlp_task_aligned_quadratic_tensor_rank_screen"
            ),
            "candidate_scope": (
                "matched post-GELU hidden/output frames plus exactly two "
                "independently seeded layer-0 full-expansion-width "
                "identity-initialized fixed-basis self-bilinear slopes"
            ),
            "block_fht_mlp_postgelu_hidden_self_gate": True,
            "block_fht_mlp_postgelu_hidden_self_gate_scale": 1.0,
            "block_fht_mlp_postgelu_hidden_self_gate_layers": [
                selected_layer
            ],
            "block_fht_mlp_postgelu_hidden_self_gate_heads": 2,
            "block_fht_mlp_postgelu_hidden_self_gate_head_seed_stride": (
                1000003
            ),
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
                f"{OUTPUT_ROOT}/"
                "performance_preflight_hidden_multihead2_l0.json"
            ),
            "failed_mfu_preflight": [
                {
                    "classification": "ADMINISTRATIVE_LAUNCH_BLOCK",
                    "path": (
                        f"{OUTPUT_ROOT}/performance_preflight_"
                        "hidden_multihead2_l0_launch_blocked_failed.json"
                    ),
                    "sha256": (
                        "3e0e91e0963df2842400c2389aa6cd01d6c94d60116799d42a7d4bf530a16d92"
                    ),
                },
                {
                    "classification": "MFU_BELOW_20_PERCENT",
                    "mfu_fraction": 0.170802657,
                    "path": (
                        f"{OUTPUT_ROOT}/performance_preflight_"
                        "hidden_multihead2_l0_generic_fht_failed_17p08.json"
                    ),
                    "sha256": (
                        "6dc08d094c6c1264672e599d60b63589bf7cb89f7b5913234d7f51e1d41814b0"
                    ),
                },
                {
                    "classification": "MFU_BELOW_20_PERCENT",
                    "mfu_fraction": 0.192273007,
                    "path": (
                        f"{OUTPUT_ROOT}/performance_preflight_"
                        "hidden_multihead2_l0_warp_butterfly_failed_19p23.json"
                    ),
                    "sha256": (
                        "c95f8823f83678149cd00a833b707832f90e5513a57f5bfbf6a36bad26bc40c5"
                    ),
                },
                {
                    "classification": "MFU_BELOW_20_PERCENT",
                    "mfu_fraction": 0.19334875243269614,
                    "path": (
                        f"{OUTPUT_ROOT}/performance_preflight_"
                        "hidden_multihead2_l0_h8xh32_failed_19p33.json"
                    ),
                    "sha256": (
                        "8a61e31997e007e623bde9dbf198467b4287f13c750bbaed5b17d78cf0c7b946"
                    ),
                },
            ],
            "mfu_preflight_required": True,
            "mfu_min_fraction": 0.2,
            "launch_ready": True,
            "launch_block_reason": None,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "postgelu_hidden_multihead_task_direction_smallest_rung"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation CE versus the exact "
                "matched post-GELU-frame parent and the single-head result"
            ),
            "preregistered_decision_rule": {
                "primary_metric": (
                    "terminal fixed-window validation cross entropy at "
                    "update 238"
                ),
                "matched_parent_validation_ce": matched_parent_val,
                "single_head_validation_ce": float(
                    single_head["matched_comparison"]["candidate_val_ce"]
                ),
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
                "post-GELU chart coordinates and the selected 6144 dynamic "
                "slope coordinates use the registered AdamW fallback"
            ),
            "parent_config": str(PARENT.relative_to(ROOT)),
            "parent_config_sha256": sha256(PARENT),
            "direction_result": {
                "path": str(DIRECTION_RESULT.relative_to(ROOT)),
                "sha256": sha256(DIRECTION_RESULT),
                "combined_task_ce_fit_holdout_cosine": direction[
                    "combined_task_ce_fit_holdout"
                ]["cosine"],
                "headwise_task_ce_fit_holdout": direction[
                    "headwise_task_ce_fit_holdout"
                ],
                "selection_rule": direction["selection"],
            },
            "single_head_causal_result": {
                "path": str(SINGLE_HEAD_CAUSAL_RESULT.relative_to(ROOT)),
                "sha256": sha256(SINGLE_HEAD_CAUSAL_RESULT),
                "classification": single_head["decision"]["classification"],
            },
            "postgelu_hidden_multihead_gate": {
                "formula": (
                    "h + sum_head Q_out_head^-1[(Q_update_head h) * "
                    "(slope_head * Q_condition_head rmsnorm(h))]"
                ),
                "location": "after GELU and before c_proj",
                "heads": 2,
                "condition_width_per_head": 3072,
                "coordinates_total": 6144,
                "selected_layers": [selected_layer],
                "selected_coordinate_group": "slope",
                "condition_basis_seed": 271828,
                "update_basis_seed": 376557,
                "output_basis_seed": 481286,
                "head_seed_stride": 1000003,
                "per_layer_seed_offset": 64,
                "identity_initialization": True,
                "learned_bias": False,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "inference_input_dependent": True,
                "changes_activation_spectrum_before_cproj": True,
                "non_redundant_with_static_cproj": True,
                "cuda_execution": (
                    "two-warp register/shared H8xH32 transforms plus fused "
                    "correction, head reduction, and analytical VJP kernels"
                ),
            },
            "screen_only_resolution": (
                "one 124M/0.5TPP causal run after a foreground-polled "
                "20-percent-MFU qualification"
            ),
            "monitoring_policy": (
                "short preflight and run are polled directly; no aggregate "
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
                "implementation_commit": config["implementation_commit"],
                "mfu_preflight_certificate": config[
                    "mfu_preflight_certificate"
                ],
                "launch_ready": config["launch_ready"],
                "launch_block_reason": config["launch_block_reason"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
