#!/usr/bin/env python3
"""Resolve the preregistered 124M/20TPP QK+V attention confirmation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qkv_only_qk64_outputgain_5tpp_lr24e4.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_qkv_only_20tpp_plan.json"
QKV_RESULT = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_qkv_only_partial_control_result.json"
CPROJ_RESULT = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_cproj_activation_metric_result.json"
VO_RESULT = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_vo_functional_direction_result.json"
OUTPUT = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_qkv_only_qk64_outputgain_20tpp_lr24e4.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_evidence(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != "mai_124m_attention_qkv_only_20tpp_plan_v1":
        raise ValueError("unexpected QK+V 20TPP plan schema")
    expected = {
        PARENT: plan["registered_config_transform"]["parent_config_sha256"],
        QKV_RESULT: plan["promotion_basis"]["qkv_only_5tpp_result"]["sha256"],
        CPROJ_RESULT: plan["promotion_basis"]["cproj_activation_metric_result"]["sha256"],
        VO_RESULT: plan["promotion_basis"]["vo_functional_direction_result"]["sha256"],
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"evidence SHA-256 mismatch: {path.name}")
    qkv = json.loads(QKV_RESULT.read_text())
    cproj = json.loads(CPROJ_RESULT.read_text())
    vo = json.loads(VO_RESULT.read_text())
    if qkv["run"]["classification"] != "clean" or qkv["run"]["exit_code"] != 0:
        raise ValueError("QK+V 5TPP result is not clean")
    if cproj["decision"]["classification"] != "REJECT_FIXED_RANDOM_CPROJ_CHART_CERTIFIED":
        raise ValueError("c_proj fixed-chart branch is not terminally rejected")
    if vo["decision"]["classification"] != "REJECT_JOINT_VO_DIRECTION_ADVANTAGE":
        raise ValueError("joint V/O direction branch is not terminally rejected")


def build(parent: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    if parent.get("model_tier") != "124m" or float(parent.get("planned_tpp", 0)) != 5.0:
        raise ValueError("parent is not the registered 124M/5TPP config")
    if parent.get("block_fht_targets") != [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
    ]:
        raise ValueError("parent is not the registered QK+V scope")
    changes = plan["registered_config_transform"]["allowed_changes"]
    candidate = copy.deepcopy(parent)
    candidate.update(copy.deepcopy(changes))
    return candidate


def main() -> None:
    plan = json.loads(PLAN.read_text())
    validate_evidence(plan)
    parent = json.loads(PARENT.read_text())
    candidate = build(parent, plan)
    OUTPUT.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.relative_to(ROOT))
    print(sha256(OUTPUT))


if __name__ == "__main__":
    main()
