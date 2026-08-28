from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_synthetic_program_residual_structure import (
    CANONICAL_SHAPE,
    canonicalize,
    self_test,
)


def test_canonical_orientation() -> None:
    c_fc = torch.randn(3072, 768)
    c_proj = torch.randn(768, 3072)
    assert canonicalize("transformer.h.0.mlp.c_fc.weight", c_fc).shape == CANONICAL_SHAPE
    assert torch.equal(
        canonicalize("transformer.h.0.mlp.c_proj.weight", c_proj), c_proj.T.contiguous()
    )


def test_residual_structure_helpers() -> None:
    record = self_test("cpu")
    assert float(record["dense_atom_capture"]) > 0.999
