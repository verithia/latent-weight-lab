"""Generate the 124M residual-conditioned MLP output-gate screen."""

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
    "bilateral_rescondgate_muonchart154_0p5tpp"
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
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    config = json.loads(PARENT.read_text())
    config.update(
        {
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_conditioned_gate_screens/"
                f"{STEM}"
            ),
            "hpo_stage": "post_direction_rescondgate_0p5tpp",
            "ladder_slot": "bilateral_residual_conditioned_output_gate",
            "ladder_role": (
                "mlp_token_conditioned_direction_screen_provisional"
            ),
            "candidate_scope": (
                "selected bilateral generated c_proj chart plus an "
                "identity-initialized residual-conditioned output diagonal"
            ),
            "block_fht_mlp_residual_conditioned_output_gate": True,
            "block_fht_mlp_residual_conditioned_output_gate_scale": 1.0,
            "block_fht_mlp_residual_conditioned_output_gate_layers": [0],
            "block_fht_mlp_residual_conditioned_output_gate_bias": False,
            "implementation_commit": git_head(),
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_conditioned_gate_screens/"
                "performance_preflight_bilateral_rescondgate.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "token_conditioned_direction_smallest_rung"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation NLL versus attention-only, "
                "plain generated c_proj, unrestricted bilateral chart, and "
                "best output-plus-activation chart"
            ),
            "optimizer_assignment_expected": (
                "matrix-shaped BlockFHT latents are Muon-owned; bilateral "
                "chart and conditioned-gate vectors use AdamW fallback"
            ),
            "parent_bilateral_config": str(PARENT.relative_to(ROOT)),
            "parent_bilateral_config_sha256": sha256(PARENT),
            "conditioned_output_gate": {
                "formula": (
                    "update + update * "
                    "(slope * layernorm_residual)"
                ),
                "coordinates_per_layer": 768,
                "coordinates_total": 768,
                "selected_layers": [0],
                "selected_coordinate_group": "slope",
                "bias_coordinate_reused_from_parent": (
                    "the bilateral parent already has a learned per-output "
                    "residual gain, so a second static bias coordinate is "
                    "redundant"
                ),
                "selection_rule": (
                    "strongest task-CE versus dense-teacher direction "
                    "cosine among diagnosed layers, positive on fit and "
                    "holdout"
                ),
                "fit_teacher_direction_cosine": 0.04898739978671074,
                "holdout_teacher_direction_cosine": 0.05935664474964142,
                "all_diagnosed_layers_fit_teacher_direction_cosine": (
                    0.04250643029808998
                ),
                "all_diagnosed_layers_holdout_teacher_direction_cosine": (
                    0.037691108882427216
                ),
                "identity_initialization": True,
                "learned_dense_basis": False,
                "lora_adapter": False,
                "inference_input_dependent": True,
                "alignment_diagnostic_commit": (
                    "1754a2135d257a9cf01bd4e05575ee02754897b9"
                ),
                "alignment_csv_sha256": (
                    "e5035d857b7adae7448f82cefe7916ceaf338dc59d25680fd4905916250480ee"
                ),
                "teacher_fit_holdout_cosine": 0.9799924492835999,
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
