#!/usr/bin/env python3
"""Build the preregistered attention c-proj capacity intervention."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_cproj_only_qk64_outputgain_5tpp_lr24e4.json"
)
OUTPUT = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_cproj_only_qk64_outputgain_cprojratio10_5tpp_lr24e4.json"
)
PLAN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_vo_factorial_cproj_capacity_plan.json"
)
QK = "attn.c_attn.qk_headwise"
PROJECTION = "attn.c_proj"


def build(parent: dict[str, object]) -> dict[str, object]:
    """Change only c-proj fixed-chart capacity plus execution metadata."""
    if parent.get("block_fht_targets") != [QK, PROJECTION]:
        raise ValueError("parent must be the registered QK+c_proj/dense-V control")
    if parent.get("block_fht_latent_ratio") != 0.01:
        raise ValueError("parent global latent ratio must be 0.01")
    if parent.get("block_fht_latent_ratios") is not None:
        raise ValueError("parent must not already override target latent ratios")

    config = copy.deepcopy(parent)
    config.update(
        {
            "block_fht_latent_ratios": {QK: 0.01, PROJECTION: 0.10},
            "candidate_scope": (
                "From-scratch QK+c_proj partial attention replacement with "
                "dense V. Change only the fixed BlockFHT c_proj latent ratio "
                "from 0.01 to 0.10; retain QK ratio 0.01, QK64/c_proj8 "
                "Cayley charts, output gains, optimizer, schedule, data, and "
                "all seeds. This is a single preregistered capacity "
                "intervention, not a ratio sweep."
            ),
            "confirmation_slot": "qk_cproj_only_cprojratio10_5tpp",
            "hpo_stage": "attention_cproj_fixed_chart_capacity_124m_5tpp",
            "ladder_role": "attention_cproj_capacity_causal_test",
            "ladder_slot": "qk_cproj_only_cprojratio10",
            "launch_ready": False,
            "launch_block_reason": (
                "Await the terminal QK+c_proj/dense-V factorial result. "
                "Authorize only if c_proj is the dominant positive Shapley "
                "component or the V-by-c_proj interaction is material."
            ),
            "operator_override": {
                "accepted_as_formal_dense_fit_conditioned_result": False,
                "reason": (
                    "One fixed-ratio causal capacity intervention selected "
                    "from the measured tight-frame random-share behavior."
                ),
                "recorded_at": "2026-08-05",
                "scope": "124M/5TPP attention c_proj 10% latent-ratio gate",
            },
            "out_dir": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_attention_cproj_capacity/"
                "pro6_mai_v3_124m_qk_cproj_only_qk64_outputgain_"
                "cprojratio10_5tpp_lr24e4"
            ),
            "practical_equivalence_policy": (
                "Do not launch until the registered factorial gate selects "
                "this branch. Then require a fresh exact-config MFU >=20%, "
                "finite preflight losses, and a verified persistent aggregate "
                "watchdog with 20/50/100 callbacks and a 90-minute heartbeat."
            ),
            "resolved_from_template": str(PARENT.relative_to(ROOT)),
        }
    )
    return config


def authorize(
    candidate: dict[str, object],
    decision: dict[str, object],
    plan: dict[str, object],
    *,
    decision_path: str,
    decision_sha256: str,
) -> dict[str, object]:
    """Unblock only the branch selected by the immutable factorial result."""
    if candidate.get("launch_ready") is not False:
        raise ValueError("candidate must start launch-blocked")
    if decision.get("schema_version") != "mai_124m_attention_vo_factorial_result_v1":
        raise ValueError("factorial decision schema is incompatible")
    if decision.get("source_plan") != str(PLAN.relative_to(ROOT)):
        raise ValueError("factorial decision does not name the registered plan")
    if decision.get("identity") != plan.get("identity"):
        raise ValueError("factorial decision identity does not match the plan")
    selected = decision.get("decision")
    if not isinstance(selected, dict):
        raise ValueError("factorial decision payload is missing")
    if (
        selected.get("authorize_cproj_ratio10") is not True
        or selected.get("selected_branch") != "qk_cproj_only_cprojratio10"
    ):
        raise ValueError("factorial result does not authorize c-proj ratio 0.10")
    if len(decision_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in decision_sha256
    ):
        raise ValueError("factorial decision SHA-256 is invalid")

    resolved = copy.deepcopy(candidate)
    resolved.update(
        {
            "factorial_decision_artifact": decision_path,
            "factorial_decision_artifact_sha256": decision_sha256,
            "launch_ready": True,
            "launch_block_reason": None,
        }
    )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-result", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(PARENT.read_text())
    plan = json.loads(PLAN.read_text())
    decision_bytes = args.decision_result.read_bytes()
    decision = json.loads(decision_bytes)
    try:
        decision_path = str(args.decision_result.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        decision_path = str(args.decision_result.resolve())
    config = authorize(
        build(parent),
        decision,
        plan,
        decision_path=decision_path,
        decision_sha256=hashlib.sha256(decision_bytes).hexdigest(),
    )
    with OUTPUT.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
