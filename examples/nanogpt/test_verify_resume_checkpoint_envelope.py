import json

import pytest
import torch

from examples.nanogpt.verify_resume_checkpoint_envelope import verify


def checkpoint(next_iter: int = 17) -> dict:
    return {
        "schema_version": "nanogpt_exact_resume_v2",
        "model": {},
        "optimizer": {},
        "grad_scaler": {},
        "model_config": {},
        "next_iter": next_iter,
        "best_val_loss": 1.0,
        "train_data_generator_state": torch.tensor([1], dtype=torch.uint8),
        "run_identity": {"config_sha256": "a" * 64},
        "saved_at_unix": 1.0,
        "block_fht_cache_state": "flushed_not_serialized",
        "cpu_torch_rng_state": torch.tensor([2], dtype=torch.uint8),
        "cuda_rng_states": [],
        "python_random_state": (),
        "numpy_rng_state": (),
    }


def write_pair(tmp_path, payload: dict) -> None:
    torch.save(payload, tmp_path / "ckpt.pt")
    (tmp_path / "ckpt.meta.json").write_text(
        json.dumps(
            {
                "next_iter": payload["next_iter"],
                "run_identity": payload["run_identity"],
            }
        )
    )


def test_verify_accepts_consistent_resume_envelope(tmp_path) -> None:
    write_pair(tmp_path, checkpoint())
    result = verify(tmp_path / "ckpt.pt")
    assert result["readable"] is True
    assert result["metadata_consistent"] is True
    assert result["next_iter"] == 17


def test_verify_rejects_metadata_mismatch(tmp_path) -> None:
    payload = checkpoint()
    write_pair(tmp_path, payload)
    metadata = json.loads((tmp_path / "ckpt.meta.json").read_text())
    metadata["next_iter"] = 18
    (tmp_path / "ckpt.meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="next_iter"):
        verify(tmp_path / "ckpt.pt")
