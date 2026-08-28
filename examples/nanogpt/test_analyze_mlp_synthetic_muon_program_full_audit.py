from __future__ import annotations

from examples.nanogpt.analyze_mlp_synthetic_muon_program_full_audit import (
    LATE_START_STEP,
    LAYER6_PARAMETERS,
    self_test,
)


def test_frozen_full_audit_inventory() -> None:
    assert LATE_START_STEP == 159
    assert LAYER6_PARAMETERS == (
        "transformer.h.6.mlp.c_fc.weight",
        "transformer.h.6.mlp.c_proj.weight",
    )


def test_joint_pca_path_algebra() -> None:
    record = self_test("cpu")
    assert float(record["minimum_pc_absolute_cosine"]) > 0.99999
