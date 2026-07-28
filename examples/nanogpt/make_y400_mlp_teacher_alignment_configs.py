"""Generate foreground-polled 124M MLP functional-alignment screens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples" / "nanogpt" / "configs"
PARENT = (
    CONFIG_DIR
    / "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "hiddenblock32_s2_c4_g4_outblock32_s4_c4_g4_"
    "cachevjp_muonchart154_0p5tpp_lr24e4.json"
)
IMPLEMENTATION_COMMIT = "42407b64d6a725eb8d463c24b0137458b1adf841"
TEACHER_CHECKPOINT = (
    "/root/userdata/MappingNetworks/outputs/"
    "y400_mai_scaling_ladder_v3/diagnostics/"
    "124m_cproj_terminal_matched/checkpoints/"
    "attention_only_0p5tpp.model-only.pt"
)
TEACHER_SHA256 = (
    "56e72a50831342bf6fde75d8678fde2e5c14bef8413e48b20b5f0f17b1e1dbe5"
)
SOURCE_PATHS = (
    "examples/nanogpt/train.py",
    "examples/nanogpt/model.py",
    "examples/nanogpt/parameter_trajectory.py",
    "latent_weight_lab/block_fht.py",
)
LAMBDAS = (5.0, 20.0, 50.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lambda_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def make_config(value: float) -> tuple[Path, dict[str, object]]:
    parent = json.loads(PARENT.read_text())
    tag = lambda_tag(value)
    slot = f"bilateral_teacheralign_lam{tag}"
    stem = (
        "y400_mai_v3_124m_fullattn_plus_mlp_cproj_"
        "hiddenblock32_s2_c4_g4_outblock32_s4_c4_g4_"
        f"teacheralign_lam{tag}_cachevjp_muonchart154_0p5tpp_lr24e4"
    )
    config = dict(parent)
    config.update(
        {
            "out_dir": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_teacher_alignment_screens/"
                f"{stem}"
            ),
            "hpo_stage": f"post_gradient_alignment_{slot}_0p5tpp",
            "ladder_slot": slot,
            "ladder_role": (
                "mlp_functional_alignment_causal_screen_provisional"
            ),
            "candidate_scope": (
                "two-stage hidden/four-stage output fixed-basis bilateral "
                "c_proj chart with activation-weighted frozen dense-teacher "
                f"functional alignment, lambda={value:g}"
            ),
            "mlp_cproj_teacher_checkpoint": TEACHER_CHECKPOINT,
            "mlp_cproj_teacher_sha256": TEACHER_SHA256,
            "mlp_cproj_teacher_lambda": value,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "implementation_source_hashes": {
                path: sha256(ROOT / path) for path in SOURCE_PATHS
            },
            "mfu_preflight_certificate": (
                "/root/userdata/MappingNetworks/outputs/"
                "y400_mai_v3_mlp_teacher_alignment_screens/"
                f"performance_preflight_{slot}.json"
            ),
            "failed_mfu_preflight": None,
            "launch_ready": True,
            "recipe_resolution_required": False,
            "recipe_resolution_stage": (
                "post_ce_vs_teacher_chart_gradient_alignment"
            ),
            "recipe_resolution_dependency": (
                "eight-window deterministic chart-gradient diagnostic: "
                "CE/useful-direction cosine near zero on fit and holdout, "
                "teacher direction fit/holdout cosine 0.936"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation NLL versus attention-only "
                "5.4918, plain generated c_proj 5.7394, and best prior chart "
                "5.6832; teacher alignment is a causal diagnostic and is not "
                "eligible as a teacher-free final method"
            ),
            "short_run_execution_policy": (
                "foreground_polled_to_completion_no_watchdog_no_callbacks"
            ),
            "teacher_alignment_evidence": {
                "diagnostic_commit": "e9868cc",
                "diagnostic_csv_sha256": (
                    "11ab17bd62ab4baa1ff1fa7ac85a1cafe4de53a0cb0017c78cd2d136c9200f62"
                ),
                "identity_fit_cosine": 0.002838585991412401,
                "identity_holdout_cosine": 0.001727343420498073,
                "production_fit_cosine": -0.0024411554913967848,
                "production_holdout_cosine": -0.0034703270066529512,
                "teacher_fit_holdout_cosine": 0.9365761280059814,
                "fit_token_sha256": (
                    "5ffbcbcb14dd284cded97fef7b9e80fbe4656b8c8de7cdf1beff1bcc6669350b"
                ),
                "holdout_token_sha256": (
                    "6f80d31cc9e111edcfe46a71efbaaa3332e651a78c109bba095b919379cd8d4e"
                ),
            },
            "teacher_alignment_structure": {
                "teacher_trainable": False,
                "teacher_weight_only": True,
                "student_activation_detached": True,
                "target": (
                    "dense teacher c_proj applied to the same detached "
                    "student post-GELU activations"
                ),
                "gradient_targets": (
                    "student generated c_proj base and fixed-basis bilateral "
                    "chart; no c_fc or upstream activation gradient"
                ),
                "learned_dense_basis": False,
                "lora_adapter": False,
                "final_method_status": (
                    "causal diagnostic; requires a teacher-free replacement "
                    "before promotion"
                ),
            },
        }
    )
    return CONFIG_DIR / f"{stem}.json", config


def main() -> None:
    for value in LAMBDAS:
        path, config = make_config(value)
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{path.relative_to(ROOT)} sha256={sha256(path)}")


if __name__ == "__main__":
    main()
