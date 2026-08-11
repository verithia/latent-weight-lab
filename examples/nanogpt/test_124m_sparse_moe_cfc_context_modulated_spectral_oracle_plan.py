from __future__ import annotations

import json
from pathlib import Path

from examples.nanogpt.analyze_sparse_moe_cfc_context_modulated_spectral_oracle import (
    validate_plan,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_sparse_moe_cfc_context_modulated_spectral_oracle_plan.json"


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_plan_identity_and_helper_hashes_are_frozen() -> None:
    plan = _plan()
    validate_plan(plan, PLAN)
    identity = plan["identity"]
    assert identity["preregistration_git_commit"].startswith("6933d02")
    assert identity["entrypoint_sha256"] == file_sha256(
        ROOT / identity["entrypoint"]
    )


def test_context_gate_is_the_only_registered_treatment() -> None:
    plan = _plan()
    candidate = plan["candidate"]
    control = plan["control"]
    assert candidate["context_beta"] == 1.0
    assert control["context_beta"] == 0.0
    assert candidate["trainable_coordinates_per_expert"] == (
        2 * candidate["padded_width"] + candidate["expert_hidden_width"]
    )
    assert candidate["cfc_compression_ratio"] >= 200.0
    assert not candidate["materialized_dense_cfc_allowed"]
    assert not candidate["learned_dense_basis_or_residual_allowed"]


def test_pass_still_requires_a_separate_training_gate() -> None:
    authorization = _plan()["authorization"]
    assert authorization["offline_zero_checkpoint_update_oracle"]
    assert not authorization["production_implementation"]
    assert not authorization["mfu_preflight"]
    assert not authorization["language_model_training"]
    assert not authorization["larger_rung"]
