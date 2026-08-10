from __future__ import annotations

import json

import pytest
import torch

from examples.nanogpt.extract_moe_paired_snapshot import (
    SCHEMA_VERSION,
    create_snapshot,
    reclaim_source,
)


def write_source(path, *, layers=(0, 1)) -> None:
    model = {}
    for layer in layers:
        model[f"transformer.h.{layer}.mlp.expert_c_fc"] = torch.randn(2, 5, 3)
        model[f"transformer.h.{layer}.mlp.expert_c_proj"] = torch.randn(2, 3, 5)
        model[f"transformer.h.{layer}.mlp.router.weight"] = torch.randn(2, 3)
    model["transformer.wte.weight"] = torch.randn(7, 3)
    torch.save(
        {
            "schema_version": "nanogpt_exact_resume_v2",
            "model": model,
            "model_config": {"n_layer": len(layers)},
            "next_iter": 17,
            "run_identity": {"config_sha256": "a" * 64},
            "execution_provenance": {"git_commit": "b" * 40},
        },
        path,
    )


def test_extracts_exact_paired_expert_tensors_and_reclaims_auxiliary_source(tmp_path) -> None:
    source = tmp_path / "full_snapshot.pt"
    destination = tmp_path / "paired.pt"
    receipt = tmp_path / "paired.receipt.json"
    write_source(source)

    result = create_snapshot(source, destination, receipt, [1])
    assert result["state"] == "verified"
    assert result["snapshot"]["tensor_count"] == 3
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    original = torch.load(source, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["step"] == 17
    assert set(payload["model"]) == {
        "transformer.h.1.mlp.expert_c_fc",
        "transformer.h.1.mlp.expert_c_proj",
        "transformer.h.1.mlp.router.weight",
    }
    for key, value in payload["model"].items():
        assert torch.equal(value, original["model"][key])

    reclaimed = reclaim_source(source, destination, receipt)
    assert reclaimed["state"] == "reclaimed"
    assert not source.exists()
    assert json.loads(receipt.read_text())["source_deleted"] is True


def test_refuses_to_reclaim_live_checkpoint(tmp_path) -> None:
    source = tmp_path / "ckpt.pt"
    destination = tmp_path / "paired.pt"
    receipt = tmp_path / "paired.receipt.json"
    write_source(source)
    create_snapshot(source, destination, receipt, [0])
    with pytest.raises(ValueError, match="refusing to reclaim"):
        reclaim_source(source, destination, receipt)
