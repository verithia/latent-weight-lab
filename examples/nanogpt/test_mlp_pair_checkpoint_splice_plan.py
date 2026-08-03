from __future__ import annotations

import hashlib
import json
from pathlib import Path

from examples.nanogpt.analyze_mlp_pair_checkpoint_splice import variant_specs


REPO = Path(__file__).resolve().parents[2]
PLAN = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/mlp_pair_checkpoint_splice_124m_350m_plan.json"
)
RESULT = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/mlp_pair_checkpoint_splice_124m_350m_result.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cross_scale_pair_splice_plan_binds_every_repository_input() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["schema_version"] == "mlp_pair_checkpoint_splice_plan_v1"
    for path, digest in plan["source_hashes"].items():
        assert sha256(REPO / path) == digest
    assert [item["scale"] for item in plan["experiments"]] == ["124m", "350m"]
    for experiment in plan["experiments"]:
        for kind in ("parent", "candidate"):
            assert sha256(REPO / experiment[f"{kind}_config"]) == (
                experiment[f"{kind}_config_sha256"]
            )
            assert sha256(REPO / experiment[f"{kind}_result"]) == (
                experiment[f"{kind}_result_sha256"]
            )
        n_layer = int(experiment["n_layer"])
        assert plan["variants_by_layer_count"][str(n_layer)] == variant_specs(n_layer)


def test_cross_scale_pair_splice_is_zero_update_and_fresh_windowed() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["execution"]["parameter_updates"] == 0
    assert "no watchdog" in plan["execution"]["monitoring"]
    assert plan["protocol"]["validation_seeds"] == [20260833, 20260834]
    assert plan["protocol"]["batches_per_window"] == 32
    assert "No production structure" in plan["authorization"]


def test_cross_scale_pair_splice_result_is_fail_closed_and_nonarchitectural() -> None:
    result = json.loads(RESULT.read_text())
    assert result["schema_version"] == "mlp_pair_checkpoint_splice_result_v1"
    assert result["decision"] == "WIDER_RESIDUAL_BLOCK_CONTEXT_DOMINATES"
    assert sha256(PLAN) == result["inputs"]["plan_sha256"]
    assert result["execution"]["parameter_updates"] == 0
    assert result["execution"]["direct_foreground_polling"] is True
    assert set(result["classification_by_scale"].values()) == {
        "WIDER_RESIDUAL_BLOCK_CONTEXT_DOMINATES"
    }
    for scale in ("124m", "350m"):
        for metrics in result["recovery_by_scale_and_window"][scale].values():
            assert metrics["mlp_pair_plus_ln2"] <= 0.25
    assert "HyperConnection or learned residual topology" in (
        result["next_action"]["not_authorized"]
    )
