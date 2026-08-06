from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from examples.nanogpt.parameter_trajectory import FULL_STATE_SCHEMA_VERSION
from examples.nanogpt.verify_full_state_functional_replay import (
    expected_buffer_names,
    parse_logged_losses,
    validate_full_state_inventory,
)


def test_parse_logged_losses_uses_terminal_occurrence() -> None:
    lines = []
    for step, loss in zip((0, 594, 1188, 1782, 2373), (11.0, 4.2, 3.8, 3.7, 3.6)):
        lines.append(f"step {step}: train loss {loss + 0.1:.4f}, val loss {loss:.4f}")
    lines.append("step 594: train loss 4.2500, val loss 4.2100")
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "train.log"
        path.write_text("\n".join(lines) + "\n")
        parsed = parse_logged_losses(path)
    assert parsed[594]["val"] == 4.21
    assert parsed[2373]["val"] == 3.6


def test_validate_full_state_inventory_fails_closed() -> None:
    buffers = {
        name: (
            torch.tensor(7, dtype=torch.int64)
            if name.endswith("optimizer_step")
            else torch.ones(1)
        )
        for name in expected_buffer_names(2)
    }
    snapshot = {
        "schema_version": FULL_STATE_SCHEMA_VERSION,
        "all_parameters": True,
        "all_buffers": True,
        "parameters": {f"p{index}": torch.ones(1) for index in range(327)},
        "buffers": buffers,
    }
    validate_full_state_inventory(snapshot, n_layer=2)
    changed = dict(snapshot)
    changed["buffers"] = dict(buffers)
    changed["buffers"].pop("transformer.h.0.mlp.c_fc.weight")
    with pytest.raises(ValueError, match="persistent-buffer inventory"):
        validate_full_state_inventory(changed, n_layer=2)
