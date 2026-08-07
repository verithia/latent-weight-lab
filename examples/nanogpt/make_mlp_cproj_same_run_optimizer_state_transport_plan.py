#!/usr/bin/env python3
"""Seal an accepted paired-state replay into a zero-update analysis plan."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / (
    "examples/nanogpt/analyze_mlp_cproj_same_run_optimizer_state_transport.py"
)
SCHEMA_VERSION = "mai_124m_mlp_cproj_same_run_optimizer_state_transport_plan_v2"
LAYERS = [8, 9, 10, 11]
SNAPSHOT_STEPS = [0, *range(99, 2373, 99), 2373]
PROBE_STEPS = [0, 98, 296, 593, 890, 1187, 1484, 1781, 2078, 2372]
REFERENCE_POST_STEPS = [99, 297, 594, 891, 1188, 1485, 1782, 2079, 2373]
DISCOVERY_STEPS = [
    0, 99, 198, 297, 396, 495, 594, 693, 792, 891,
    990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782,
]
HELDOUT_PROBE_STEPS = [1781, 2078, 2372]
COMPONENTS = ["requested", "feedback", "corrected", "realized", "unrepresented"]
SUPPORTING_SOURCES = (
    "examples/nanogpt/analyze_mlp_activation_update_alignment.py",
    "examples/nanogpt/analyze_mlp_cproj_activation_weighted_output_selector.py",
    "examples/nanogpt/analyze_mlp_cproj_multiscale_path.py",
    "examples/nanogpt/analyze_mlp_cproj_optimizer_state_transport.py",
    "examples/nanogpt/analyze_mlp_cproj_polynomial_oracle_ce.py",
    "examples/nanogpt/analyze_mlp_cproj_predictive_manifold.py",
    "examples/nanogpt/analyze_parameter_trajectory.py",
    "examples/nanogpt/model.py",
    "examples/nanogpt/muon.py",
    "examples/nanogpt/parameter_trajectory.py",
    "examples/nanogpt/train.py",
    "latent_weight_lab/block_fht.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_plan(
    *,
    acquisition_result_path: Path,
    acquisition_plan_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    snapshot_dir: Path,
    probe_dir: Path,
    analysis_output_dir: Path,
    invalid_analysis_audit_path: Path,
) -> dict[str, Any]:
    acquisition = load_json(acquisition_result_path)
    if acquisition.get("classification") != (
        "ACCEPTED_SAME_RUN_CPROJ_PARAMETER_OPTIMIZER_TRAJECTORY"
    ) or acquisition.get("passed") is not True:
        raise ValueError("paired-state acquisition is not accepted")
    if acquisition.get("authorization", {}).get(
        "zero_update_state_transport_analysis"
    ) is not True:
        raise ValueError("paired-state acquisition does not authorize analysis")
    acquisition_plan = load_json(acquisition_plan_path)
    invalid_analysis_audit = load_json(invalid_analysis_audit_path)
    if invalid_analysis_audit.get("classification") != (
        "INVALID_RECOMPUTED_REQUEST_NOT_EXACT_GPU_STATE"
    ):
        raise ValueError("prior invalid-analysis audit classification mismatch")
    if invalid_analysis_audit.get("authorization", {}).get(
        "exact_post_step_state_identity_reanalysis"
    ) is not True:
        raise ValueError("prior invalid-analysis audit does not authorize repair")
    identity = acquisition["identity"]
    if sha256_file(acquisition_plan_path) != identity["plan_sha256"]:
        raise ValueError("accepted acquisition plan SHA-256 mismatch")
    if sha256_file(config_path) != identity["config_sha256"]:
        raise ValueError("accepted config SHA-256 mismatch")
    if sha256_file(checkpoint_path) != identity["checkpoint_sha256"]:
        raise ValueError("accepted checkpoint SHA-256 mismatch")
    acquisition_identity = acquisition_plan["identity"]
    config = load_json(config_path)
    dataset_manifest = Path(config["data_dir"]) / "manifest.json"
    if sha256_file(dataset_manifest) != acquisition_identity[
        "dataset_manifest_sha256"
    ]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    inventory = acquisition["inventory"]
    if inventory.get("snapshot_count") != len(SNAPSHOT_STEPS):
        raise ValueError("accepted snapshot inventory count changed")
    if inventory.get("probe_count") != len(PROBE_STEPS):
        raise ValueError("accepted optimizer-probe inventory count changed")
    if sorted(map(int, inventory["snapshot_sha256_by_step"])) != SNAPSHOT_STEPS:
        raise ValueError("accepted snapshot step inventory changed")
    if sorted(map(int, inventory["probe_sha256_by_step"])) != PROBE_STEPS:
        raise ValueError("accepted optimizer-probe step inventory changed")
    future_targets = {
        str(step): (
            REFERENCE_POST_STEPS[index]
            if index < len(REFERENCE_POST_STEPS)
            else None
        )
        for index, step in enumerate(PROBE_STEPS)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_question": (
            "Does same-run structured-Muon optimizer state causally recover "
            "the function-critical late-c_proj residual that static charts miss?"
        ),
        "causal_basis": [
            "The accepted paired replay establishes bitwise parameter/probe pairing under one run identity.",
            "The earlier cross-run pairing was rejected because independent CUDA paths diverged despite matching initialization and CE.",
            "Static endpoint charts and output-additive fits did not establish causal state transport.",
            "The first same-run analysis was invalid because a CPU-recomputed Muon request was not exact GPU optimizer state.",
            "This repair uses exact captured post-step identities, performs zero parameter updates, and retains the frozen chronological threshold.",
        ],
        "identity": {
            "analyzer": str(ANALYZER.relative_to(ROOT)),
            "analyzer_sha256": sha256_file(ANALYZER),
            "acquisition_result": str(acquisition_result_path),
            "acquisition_result_sha256": sha256_file(acquisition_result_path),
            "acquisition_plan": str(acquisition_plan_path),
            "acquisition_plan_sha256": sha256_file(acquisition_plan_path),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "dataset_manifest": str(dataset_manifest),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "fixed_eval_indices_sha256": acquisition_identity[
                "fixed_eval_indices_sha256"
            ],
            "run_identity_sha256": identity["run_identity_sha256"],
            "invalid_analysis_audit": str(invalid_analysis_audit_path),
            "invalid_analysis_audit_sha256": sha256_file(
                invalid_analysis_audit_path
            ),
            "supporting_source_sha256": {
                path: sha256_file(ROOT / path) for path in SUPPORTING_SOURCES
            },
        },
        "analysis": {
            "parameter_updates": 0,
            "same_run_only": True,
            "layers": LAYERS,
            "snapshot_steps": SNAPSHOT_STEPS,
            "probe_steps": PROBE_STEPS,
            "reference_post_steps": REFERENCE_POST_STEPS,
            "discovery_steps": DISCOVERY_STEPS,
            "terminal_step": 2373,
            "polynomial_rank": 4,
            "polynomial_degree": 2,
            "activation_rows": 2048,
            "terminal_activations_from_same_run_checkpoint": True,
            "components": COMPONENTS,
            "heldout_probe_steps": HELDOUT_PROBE_STEPS,
            "output_additive_projection": True,
            "state_component_reconstruction": (
                "exact_post_step_optimizer_identity"
            ),
            "recomputed_polar_request": "diagnostic_only",
            "authorization_metric": (
                "heldout_future_functional_positive_line_recovery"
            ),
            "future_phase_target_by_probe": future_targets,
        },
        "decision_rule": {
            "thresholds": {
                "state_identity_max_relative_error": 1e-4,
                "causal_heldout_functional_line_recovery_minimum": 0.80,
            },
            "selection": (
                "Authorize only when the mechanically reconstructed state is "
                "valid and one frozen component reaches at least 0.80 "
                "energy-weighted functional positive-line recovery for the "
                "chronological heldout future phase."
            ),
            "threshold_change_after_observation": False,
        },
        "authorization": {
            "run_zero_update_state_transport_analysis": True,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
        "execution": {
            "host": "PRO6",
            "device": "cuda",
            "snapshot_dir": str(snapshot_dir),
            "probe_dir": str(probe_dir),
            "analysis_output_dir": str(analysis_output_dir),
            "foreground_direct_polling": True,
            "watchdog": False,
            "parameter_updates": 0,
        },
        "limitations": [
            "Terminal post-GELU activations define a local functional metric, not a global nonlinear equivalence proof.",
            "Requested and corrected components are exact post-step algebraic identities, not a replay of the pre-step GPU Newton-Schulz tensor.",
            "Passing authorizes only a separately preregistered compact candidate; it does not authorize language-model training directly.",
            "Failing rejects this optimizer-state transport hypothesis but does not prove that no state-dependent generator can close the gap.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--acquisition-plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--analysis-output-dir", type=Path, required=True)
    parser.add_argument("--invalid-analysis-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"analysis plan already exists: {args.output}")
    plan = build_plan(
        acquisition_result_path=args.acquisition_result,
        acquisition_plan_path=args.acquisition_plan,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        snapshot_dir=args.snapshot_dir,
        probe_dir=args.probe_dir,
        analysis_output_dir=args.analysis_output_dir,
        invalid_analysis_audit_path=args.invalid_analysis_audit,
    )
    encoded = (
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {"output": str(args.output), "sha256": hashlib.sha256(encoded).hexdigest()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
