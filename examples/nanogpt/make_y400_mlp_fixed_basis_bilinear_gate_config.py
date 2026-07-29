"""Generate the 124M fixed-basis bilinear MLP output-gate screen."""

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
    "hiddenblock32_s2_c4_g4_outblock32_s4_c4_g4_"
    "cachevjp_muonchart154_0p5tpp_lr24e4.json"
)
STEM = (
    "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "bilateral_fixedbilinear_l0_muonchart154_0p5tpp"
)
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/train.py",
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
    config.update(
        {
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_fixed_basis_bilinear_screens/"
                f"{STEM}"
            ),
            "hpo_stage": "post_direction_fixed_bilinear_l0_0p5tpp",
            "ladder_slot": "bilateral_fixed_basis_bilinear_output_gate",
            "ladder_role": (
                "mlp_token_conditioned_rotating_direction_screen_provisional"
            ),
            "candidate_scope": (
                "selected bilateral generated c_proj chart plus a layer-0 "
                "identity-initialized fixed-basis bilinear "
                "residual-conditioned output slope"
            ),
            "block_fht_mlp_residual_conditioned_output_gate": True,
            "block_fht_mlp_residual_conditioned_output_gate_scale": 1.0,
            "block_fht_mlp_residual_conditioned_output_gate_layers": [0],
            "block_fht_mlp_residual_conditioned_output_gate_bias": False,
            "block_fht_mlp_residual_conditioned_output_gate_fixed_basis": (
                True
            ),
            "block_fht_mlp_residual_conditioned_output_gate_basis_block_size": (
                256
            ),
            "block_fht_mlp_residual_conditioned_output_gate_basis_seed": (
                271828
            ),
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_fixed_basis_bilinear_screens/"
                "performance_preflight_fixedbilinear_l0_blockgemm.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "fixed_basis_bilinear_direction_smallest_rung"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation NLL versus attention-only, "
                "plain generated c_proj, unrestricted bilateral chart, "
                "best output-plus-activation chart, and raw diagonal gate"
            ),
            "optimizer_assignment_expected": (
                "matrix-shaped BlockFHT latents are Muon-owned; bilateral "
                "chart and fixed-basis gate slope use AdamW fallback"
            ),
            "parent_bilateral_config": str(PARENT.relative_to(ROOT)),
            "parent_bilateral_config_sha256": sha256(PARENT),
            "conditioned_output_gate": {
                "formula": (
                    "update + Q^-1[(Q update) * "
                    "(slope * Q layernorm_residual)]"
                ),
                "coordinates_per_layer": 768,
                "coordinates_total": 768,
                "selected_layers": [0],
                "selected_coordinate_group": "slope",
                "basis": (
                    "fixed signed/permuted 256-wide normalized "
                    "block-Hadamard"
                ),
                "basis_execution": (
                    "fixed 256x256 block-Hadamard accelerator GEMM"
                ),
                "basis_seed": 271828,
                "fit_teacher_direction_cosine": 0.1229742094874382,
                "holdout_teacher_direction_cosine": 0.09516574442386627,
                "all_diagnosed_layers_fit_teacher_direction_cosine": (
                    0.0545714870095253
                ),
                "all_diagnosed_layers_holdout_teacher_direction_cosine": (
                    0.040879715234041214
                ),
                "teacher_fit_holdout_cosine": 0.9809097647666931,
                "identity_initialization": True,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "inference_input_dependent": True,
                "alignment_diagnostic_commit": (
                    "7817d38f345c5aa23a1ed3fc6235022de0cf1ccf"
                ),
                "alignment_csv_sha256": (
                    "a57c9a492d08035f59a6d81f3f638729f73e5706a3bedec04f21279ef86b2489"
                ),
            },
            "screen_only_resolution": (
                "one 124M/0.5TPP causal run after a foreground-polled "
                "20-percent-MFU qualification"
            ),
            "prelaunch_provenance_requirements": (
                "record source hashes, resolved config SHA256, data manifest "
                "SHA256, runtime fixed-evaluation digest, literal command, "
                "exact host-local MFU certificate, and terminal checkpoint"
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
