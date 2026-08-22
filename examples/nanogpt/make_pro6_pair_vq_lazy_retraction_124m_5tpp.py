#!/usr/bin/env python3
"""Preregister the sole 124M/5TPP lazy Pair-VQ retraction causal test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qkonly_pairvq_mlp_5tpp_localization_lr24e4.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_pair_vq_lazy_retraction_plan.json"
)
SHORT_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_pair_vq_lazy_retraction_0p5tpp_result.json"
)
PER_STEP_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_qkonly_pair_vq_mlp_5tpp_localization_result.json"
)
DENSE_PARENT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_qk_only_partial_control_result.json"
)
OUTPUT = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qkonly_pairvq_mlp_lazyretract8_5tpp_lr24e4.json"
)
REGISTRATION = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_pair_vq_lazy_retraction_5tpp_registration.json"
)
REMOTE_ROOT = "/mnt/ssd-data/orj/MappingNetworks"
RUN_NAME = "pro6_mai_v3_124m_qkonly_pairvq_mlp_lazyretract8_5tpp_lr24e4"
ADAPTIVE_MOMENTUM_BYTES_MAX = 99_090_432


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualification() -> dict[str, object]:
    result = json.loads(SHORT_RESULT.read_text())
    if result.get("passed") is not True:
        raise RuntimeError("0.5TPP lazy-retraction endpoint did not pass")
    return result


def build_registration() -> dict[str, object]:
    short = qualification()
    per_step = json.loads(PER_STEP_RESULT.read_text())
    return {
        "schema_version": "mai_pair_vq_lazy_retraction_124m_5tpp_registration_v1",
        "registered_at": "2026-08-22",
        "hypothesis": (
            "The 0.0683 CE late-horizon full-MLP gap is caused primarily by "
            "per-step compact-chart retraction perturbing native Muon requests; "
            "carrying the ambient request for the already-frozen eight-step "
            "code-refresh interval should close the 5TPP gap while checkpoints "
            "remain compact."
        ),
        "qualification": {
            "short_result": str(SHORT_RESULT.relative_to(ROOT)),
            "short_result_sha256": sha256(SHORT_RESULT),
            "short_terminal_validation_ce": short["terminal"][
                "validation_ce_exact"
            ],
            "short_parent_gap_ce": short["terminal"][
                "candidate_minus_parent_validation_ce"
            ],
            "short_fixed_model_compute_penalty": short["terminal"][
                "fixed_model_compute_equivalent_penalty"
            ],
            "short_passed": True,
        },
        "causal_comparator": {
            "per_step_result": str(PER_STEP_RESULT.relative_to(ROOT)),
            "per_step_result_sha256": sha256(PER_STEP_RESULT),
            "per_step_terminal_validation_ce": per_step["terminal"][
                "validation_ce"
            ],
            "per_step_dense_parent_gap_ce": per_step["terminal"][
                "candidate_minus_qk_only_ce"
            ],
            "only_intended_mechanism_change": (
                "Pair-VQ retraction cadence changes from every optimizer step "
                "to every eight optimizer steps plus fixed evaluation boundaries"
            ),
        },
        "frozen_test": {
            "model": "124M QK-only attention plus full Pair-VQ MLP",
            "horizon_updates": 2373,
            "eval_steps": [0, 594, 1188, 1782, 2373],
            "lazy_retraction_interval": 8,
            "forced_compact_boundaries": [594, 1188, 1782, 2373],
            "matched_dense_mlp_parent_result": str(DENSE_PARENT.relative_to(ROOT)),
            "matched_dense_mlp_parent_result_sha256": sha256(DENSE_PARENT),
            "matched_dense_mlp_parent_validation_ce": 3.4858,
            "terminal_validation_ce_maximum": 3.4958,
            "fixed_model_compute_equivalent_penalty_maximum": 1.10,
            "minimum_exact_config_mfu_fraction": 0.20,
        },
        "decision_policy": {
            "on_pass": (
                "authorize a separately preregistered 124M composition with "
                "compact attention V/output only"
            ),
            "on_failure": (
                "reject eight-step lazy retraction as sufficient; do not sweep "
                "interval, bit width, seed, learning rate, horizon, or model size"
            ),
            "automatic_rerun": False,
            "automatic_sweep": False,
            "automatic_20tpp": False,
            "automatic_scale_up": False,
        },
    }


def build_config(registration_sha256: str) -> dict[str, object]:
    qualification()
    config = json.loads(BASE.read_text())
    remote_output = f"{REMOTE_ROOT}/outputs/pro6_mai_v3_pair_vq/{RUN_NAME}"
    config.update(
        {
            "schema_version": "mai_qkonly_pairvq_mlp_lazyretract8_5tpp_v1",
            "experiment_role": (
                "sole preregistered 124M/5TPP causal test of eight-step ambient "
                "Muon carry with compact-boundary Pair-VQ retraction"
            ),
            "scientific_parent": str(DENSE_PARENT.relative_to(ROOT)),
            "scientific_parent_sha256": sha256(DENSE_PARENT),
            "qualification_dependency": str(SHORT_RESULT.relative_to(ROOT)),
            "qualification_dependency_sha256": sha256(SHORT_RESULT),
            "causal_comparator": str(PER_STEP_RESULT.relative_to(ROOT)),
            "causal_comparator_sha256": sha256(PER_STEP_RESULT),
            "theory_plan": str(PLAN.relative_to(ROOT)),
            "theory_plan_sha256": sha256(PLAN),
            "preregistration": str(REGISTRATION.relative_to(ROOT)),
            "preregistration_sha256": registration_sha256,
            "implementation_commit": "0c0ddff5f1b1",
            "literal_command": (
                "CUDA_VISIBLE_DEVICES=0 "
                f"LD_LIBRARY_PATH={REMOTE_ROOT}/.compat/nvidia-595.71.05/root/usr/lib/x86_64-linux-gnu "
                f"TORCH_EXTENSIONS_DIR={REMOTE_ROOT}/.cache/torch_extensions_sm120_exactfamily "
                "TORCH_CUDA_ARCH_LIST=12.0 PYTHONPATH=. "
                f"{REMOTE_ROOT}/.venv/bin/python -u examples/nanogpt/train.py "
                f"--config examples/nanogpt/configs/{OUTPUT.name}"
            ),
            "mfu_preflight_certificate": (
                f"{REMOTE_ROOT}/outputs/pro6_mai_v3_pair_vq/preflights/"
                f"{RUN_NAME}_mfu.json"
            ),
            "persistent_momentum_bytes_max": ADAPTIVE_MOMENTUM_BYTES_MAX,
            "out_dir": f"{remote_output}/scientific",
            "block_fht_mlp_pair_vq_fp16_reserved_escape_granularity": (
                "adaptive_block"
            ),
            "block_fht_mlp_pair_vq_lazy_retraction_interval": 8,
            "block_fht_mlp_pair_vq_lazy_retraction_forced_steps": [
                594,
                1188,
                1782,
                2373,
            ],
            "monitoring_policy": (
                "one idempotent terminal/error-only @Codex watchdog; no progress "
                "milestones and no healthy heartbeat"
            ),
            "endpoint_gate": {
                "matched_parent_result": str(DENSE_PARENT.relative_to(ROOT)),
                "matched_parent_terminal_validation_ce": 3.4858,
                "terminal_candidate_validation_ce_max": 3.4958,
                "candidate_minus_parent_terminal_validation_ce_max": 0.01,
                "fixed_model_compute_equivalent_penalty_max": 1.10,
                "persistent_momentum_bytes_max": ADAPTIVE_MOMENTUM_BYTES_MAX,
                "persistent_raw_ambient_momentum_tensors": 0,
                "lazy_retraction_interval": 8,
                "forced_compact_boundary_steps": [594, 1188, 1782, 2373],
                "compact_checkpoint_required": True,
                "same_fixed_eval_protocol_required": True,
                "all_fixed_eval_steps_finite": True,
                "automatic_rerun": False,
                "automatic_sweep": False,
                "automatic_horizon_transfer": False,
                "automatic_scale_up": False,
            },
        }
    )
    config.pop("mfu_preflight_pair_vq_persistent_training_bytes_exact", None)
    return config


def main() -> None:
    registration = build_registration()
    REGISTRATION.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    OUTPUT.write_text(
        json.dumps(build_config(sha256(REGISTRATION)), indent=2, sort_keys=True)
        + "\n"
    )
    print(REGISTRATION.relative_to(ROOT), sha256(REGISTRATION))
    print(OUTPUT.relative_to(ROOT), sha256(OUTPUT))


if __name__ == "__main__":
    main()
