"""Generate the 124M untied fixed-basis bilinear MLP output-gate screen."""

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
    "bilateral_untiedbilinear_l6_muonchart154_0p5tpp"
)
SOURCE_PATHS = (
    "examples/nanogpt/model.py",
    "examples/nanogpt/train.py",
    "latent_weight_lab/block_fht.py",
)
DIRECTION_RESULT = (
    CONFIG_DIR
    / "selection_artifacts"
    / "124m_mlp_untied_fixed_basis_direction_result.json"
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
                "y400_mai_v3_mlp_untied_fixed_basis_bilinear_screens/"
                f"{STEM}"
            ),
            "hpo_stage": "post_direction_untied_fixed_bilinear_l6_0p5tpp",
            "ladder_slot": (
                "bilateral_untied_fixed_basis_bilinear_output_gate"
            ),
            "ladder_role": (
                "mlp_token_conditioned_nonsymmetric_direction_screen_"
                "provisional"
            ),
            "candidate_scope": (
                "selected bilateral generated c_proj chart plus a layer-6 "
                "identity-initialized residual-conditioned bilinear slope "
                "with independent fixed condition, update, and output bases"
            ),
            "block_fht_mlp_residual_conditioned_output_gate": True,
            "block_fht_mlp_residual_conditioned_output_gate_scale": 1.0,
            "block_fht_mlp_residual_conditioned_output_gate_layers": [6],
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
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_untied_fixed_basis_bilinear_screens/"
                "performance_preflight_untiedbilinear_l6_blockgemm.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "untied_fixed_basis_bilinear_direction_smallest_rung"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation NLL versus attention-only, "
                "plain generated c_proj, unrestricted bilateral chart, best "
                "output-plus-activation chart, raw dynamic diagonal, and "
                "tied fixed-basis bilinear gate"
            ),
            "optimizer_assignment_expected": (
                "matrix-shaped BlockFHT latents are Muon-owned; bilateral "
                "chart and untied-bilinear gate slope use AdamW fallback"
            ),
            "parent_bilateral_config": str(PARENT.relative_to(ROOT)),
            "parent_bilateral_config_sha256": sha256(PARENT),
            "direction_result": {
                "path": str(DIRECTION_RESULT.relative_to(ROOT)),
                "sha256": sha256(DIRECTION_RESULT),
                "global_fit_teacher_direction_cosine": (
                    0.009625661186873913
                ),
                "global_holdout_teacher_direction_cosine": (
                    0.015025950968265533
                ),
                "selected_layer": 6,
                "selected_layer_fit_teacher_direction_cosine": (
                    0.08055470883846283
                ),
                "selected_layer_holdout_teacher_direction_cosine": (
                    0.058075133711099625
                ),
                "selection_rule": (
                    "maximum minimum fit/holdout cosine among layers "
                    "clearing 0.05"
                ),
            },
            "conditioned_output_gate": {
                "formula": (
                    "update + Q_output^-1[(Q_update update) * "
                    "(slope * Q_condition layernorm_residual)]"
                ),
                "coordinates_per_layer": 768,
                "coordinates_total": 768,
                "selected_layers": [6],
                "selected_coordinate_group": "slope",
                "basis": (
                    "three independently signed/permuted 256-wide normalized "
                    "block-Hadamard transforms"
                ),
                "basis_execution": (
                    "three fixed 256x256 block-Hadamard accelerator GEMMs"
                ),
                "condition_basis_seed": 271828,
                "update_basis_seed": 376557,
                "output_basis_seed": 481286,
                "identity_initialization": True,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "inference_input_dependent": True,
                "nonsymmetric": True,
                "non_diagonal_in_any_single_basis": True,
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
