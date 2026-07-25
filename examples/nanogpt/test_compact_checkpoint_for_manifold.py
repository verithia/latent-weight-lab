import json

import torch

from examples.nanogpt.compact_checkpoint_for_manifold import (
    create_snapshot,
    reclaim_source,
)


def write_checkpoint(path) -> None:
    torch.save(
        {
            "schema_version": "nanogpt_exact_resume_v2",
            "model": {"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
            "model_config": {"n_layer": 1},
            "next_iter": 17,
            "best_val_loss": 2.5,
            "run_identity": {"config_sha256": "a" * 64},
            "execution_provenance": {"sha256": "b" * 64},
            "saved_at_unix": 1.0,
        },
        path,
    )


def test_snapshot_preserves_tensors_before_guarded_reclaim(tmp_path) -> None:
    source = tmp_path / "ckpt.pt"
    destination = tmp_path / "manifold_snapshot.pt"
    receipt = tmp_path / "manifold_snapshot.receipt.json"
    write_checkpoint(source)

    created = create_snapshot(source, destination, receipt)
    assert created["state"] == "verified"
    assert source.exists()
    snapshot = torch.load(destination, map_location="cpu", weights_only=False)
    assert torch.equal(snapshot["model"]["weight"], torch.arange(8).reshape(2, 4))

    reclaimed = reclaim_source(source, destination, receipt)
    assert reclaimed["state"] == "reclaimed"
    assert not source.exists()
    assert destination.exists()
    assert json.loads(receipt.read_text())["source_deleted"] is True
