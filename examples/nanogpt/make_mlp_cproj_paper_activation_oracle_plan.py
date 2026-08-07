#!/usr/bin/env python3
"""Create the immutable paper-activation oracle plan after state rejection."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
)
from examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle import (
    FUTURE_STEP_BY_PROBE,
    HELDOUT_PROBE_STEPS,
    LAYERS,
    PLAN_SCHEMA,
    PROBE_STEPS,
    TIGHT_PLAN_SCHEMA,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTING_SOURCES = (
    "examples/nanogpt/analyze_mlp_activation_update_alignment.py",
    "examples/nanogpt/analyze_mlp_cproj_activation_weighted_output_selector.py",
    "examples/nanogpt/analyze_mlp_cproj_same_run_optimizer_state_transport.py",
    "examples/nanogpt/analyze_parameter_trajectory.py",
    "examples/nanogpt/model.py",
    "examples/nanogpt/train.py",
    "latent_weight_lab/block_fht.py",
)


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-plan", type=Path, required=True)
    parser.add_argument("--exact-result", type=Path, required=True)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--quadratic-result", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--tight-condition-number-ceiling", type=float)
    parser.add_argument("--parent-activation-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    exact_plan = json.loads(args.exact_plan.read_text())
    exact_result = json.loads(args.exact_result.read_text())
    acquisition = json.loads(args.acquisition_result.read_text())
    quadratic = json.loads(args.quadratic_result.read_text())
    tight = args.tight_condition_number_ceiling is not None
    if tight:
        if args.tight_condition_number_ceiling != 10.0:
            raise ValueError("the preregistered tight condition ceiling is exactly 10")
        if args.parent_activation_result is None:
            raise ValueError("tight plan requires the sealed parent activation result")
        parent_activation = json.loads(args.parent_activation_result.read_text())
        if parent_activation.get("classification") != "PAPER_ACTIVATION_IMAGE_INSUFFICIENT":
            raise ValueError("parent activation capacity result is not rejected")
    elif args.parent_activation_result is not None:
        raise ValueError("parent activation result is only valid for the tight plan")
    if exact_result.get("classification") != "OPTIMIZER_STATE_TRANSPORT_INSUFFICIENT":
        raise ValueError("exact optimizer-state family is not rejected")
    if not exact_result.get("exact_state_identity", {}).get("passed"):
        raise ValueError("exact optimizer-state mechanical identity did not pass")
    if acquisition.get("classification") != (
        "ACCEPTED_SAME_RUN_CPROJ_PARAMETER_OPTIMIZER_TRAJECTORY"
    ):
        raise ValueError("paired acquisition is not accepted")
    if quadratic.get("decision", {}).get("promote") is True:
        raise ValueError("generic quadratic chart is not rejected")
    prior = exact_plan["identity"]
    if file_sha256(args.acquisition_result) != prior["acquisition_result_sha256"]:
        raise ValueError("acquisition result changed")
    if file_sha256(args.exact_result) != (
        "ec3634d08cbe321d4df5b13fd4aa5cbf3ea850bbf4b33fb1d84d689d44f1b5ea"
    ):
        raise ValueError("exact-state result seal changed")
    scale_multiplier = math.sqrt(10.0 / 9.0) if tight else 2.0
    activation_scale_name = (
        "sqrt(10/9)*max_abs_step0_per_layer"
        if tight
        else "2*max_abs_step0_per_layer"
    )
    identity = {
        "analyzer": repo_relative(args.analyzer),
        "analyzer_sha256": file_sha256(args.analyzer),
        "exact_state_plan": repo_relative(args.exact_plan),
        "exact_state_plan_sha256": file_sha256(args.exact_plan),
        "exact_state_result": repo_relative(args.exact_result),
        "exact_state_result_sha256": file_sha256(args.exact_result),
        "acquisition_result": repo_relative(args.acquisition_result),
        "acquisition_result_sha256": file_sha256(args.acquisition_result),
        "quadratic_result": repo_relative(args.quadratic_result),
        "quadratic_result_sha256": file_sha256(args.quadratic_result),
        "checkpoint": prior["checkpoint"],
        "checkpoint_sha256": prior["checkpoint_sha256"],
        "config": prior["config"],
        "config_sha256": prior["config_sha256"],
        "dataset_manifest": prior["dataset_manifest"],
        "dataset_manifest_sha256": prior["dataset_manifest_sha256"],
        "fixed_eval_indices_sha256": prior["fixed_eval_indices_sha256"],
        "run_identity_sha256": prior["run_identity_sha256"],
        "supporting_source_sha256": {
            relative: file_sha256(REPO_ROOT / relative)
            for relative in SUPPORTING_SOURCES
        },
    }
    if tight:
        assert args.parent_activation_result is not None
        identity.update(
            {
                "parent_activation_result": repo_relative(
                    args.parent_activation_result
                ),
                "parent_activation_result_sha256": file_sha256(
                    args.parent_activation_result
                ),
            }
        )
    analysis = {
        "parameter_updates": 0,
        "same_run_only": True,
        "layers": list(LAYERS),
        "probe_steps": list(PROBE_STEPS),
        "heldout_probe_steps": list(HELDOUT_PROBE_STEPS),
        "future_step_by_probe": {
            str(key): value for key, value in FUTURE_STEP_BY_PROBE.items()
        },
        "activation": "signed_scaled_tanh",
        "activation_scale": activation_scale_name,
        "activation_bias": "s*atanh(W_step0/s)",
        "fixed_operator": "production_seeded_BlockFHT",
        "latent_ratio": 0.01,
        "block_fht_layers": 2,
        "activation_rows": 2048,
        "coordinate_fit": "oracle_cgls_in_inverse_activation_preactivation",
        "future_tangent_fit": "oracle_cgls_in_terminal_post_gelu_metric",
        "primary_control": "identity_BlockFHT_tangent",
        "generic_quadratic_control": "sealed_124m_quadratic_screen",
        "cgls_iterations": 32,
    }
    if tight:
        analysis.update(
            {
                "activation_scale_multiplier": scale_multiplier,
                "minimum_step0_activation_derivative": 0.1,
                "step0_jacobian_condition_ceiling": 10.0,
            }
        )
    plan = {
        "schema_version": TIGHT_PLAN_SCHEMA if tight else PLAN_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "question": (
            "Can the paper's elementwise mapping activation make a fixed 1% "
            "BlockFHT image and Jacobian contain the accepted late-c_proj path?"
        ),
        "identity": identity,
        "analysis": analysis,
        "decision_rule": {
            "thresholds": {
                "heldout_current_functional_image_recovery_minimum": 0.80,
                "heldout_future_functional_tangent_recovery_minimum": 0.80,
            },
            "interpretation": {
                "range_failure": "fixed init-derived tanh scale cannot contain the path",
                "image_failure": "fixed nonlinear manifold cannot represent current states even with oracle coordinates",
                "tangent_failure": "state-dependent activation Jacobian cannot represent future action even with oracle coefficients",
                "pass": "capacity only; requires a separately preregistered causal coordinate-update selector before any training",
            },
        },
        "authorization": {
            "run_zero_update_oracle": True,
            "implement_production_candidate": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
