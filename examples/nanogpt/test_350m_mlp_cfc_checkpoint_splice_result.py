from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULT = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/350m_mlp_cfc_checkpoint_splice_result.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_splice_result_binds_plan_source_and_registered_decision() -> None:
    result = json.loads(RESULT.read_text())
    assert result["schema_version"] == "350m_mlp_cfc_checkpoint_splice_result_v1"
    assert result["execution"]["parameter_updates"] == 0
    inputs = result["inputs"]
    assert sha256(REPO / inputs["plan_path"]) == inputs["plan_sha256"]
    assert sha256(REPO / inputs["source_path"]) == inputs["source_sha256"]
    decision = result["decision"]
    assert decision["classification"] == "DOWNSTREAM_COADAPTATION_DOMINATES"
    assert max(decision["recovery_by_window"].values()) <= (
        decision["registered_boundaries"][
            "coadaptation_dominates_maximum_each_window"
        ]
    )


def test_splice_result_records_bidirectional_harm_and_depth_ordering() -> None:
    result = json.loads(RESULT.read_text())
    measurements = result["measurements"]
    means = measurements["mean_validation_ce"]
    assert means["candidate_parent_cfc_all"] > means["candidate"]
    assert means["parent_candidate_cfc_all"] > means["parent"]
    bands = measurements["depth_band_transplant_delta_ce_vs_candidate"]
    assert bands["early_layers_0_7"] > bands["middle_layers_8_15"]
    assert bands["middle_layers_8_15"] > bands["late_layers_16_23"]
    weights = measurements["weight_means"]
    assert weights["weight_cosine"] > 0.94
    assert weights["candidate_parent_frobenius_ratio_layer23"] < 0.94
