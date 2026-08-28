from __future__ import annotations

from examples.nanogpt.analyze_mlp_synthetic_muon_program_joint import (
    FROZEN_PARAMETERS,
    self_test,
)


def test_joint_inventory_is_exact() -> None:
    assert FROZEN_PARAMETERS == (
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.6.mlp.c_fc.weight",
        "transformer.h.6.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    )


def test_joint_mixed_hessian_polar_self_test() -> None:
    record = self_test("cpu")
    assert float(record["path_energy_capture"]) > 0.999
