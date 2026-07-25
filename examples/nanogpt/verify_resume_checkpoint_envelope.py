#!/usr/bin/env python3
"""Read and validate the durable, device-independent resume envelope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_KEYS = {
    "schema_version",
    "model",
    "optimizer",
    "grad_scaler",
    "model_config",
    "next_iter",
    "best_val_loss",
    "train_data_generator_state",
    "run_identity",
    "saved_at_unix",
    "block_fht_cache_state",
    "cpu_torch_rng_state",
    "cuda_rng_states",
    "python_random_state",
    "numpy_rng_state",
}


def verify(checkpoint_path: Path) -> dict[str, Any]:
    metadata_path = checkpoint_path.with_name("ckpt.meta.json")
    metadata = json.loads(metadata_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a dictionary")
    missing = sorted(REQUIRED_KEYS - checkpoint.keys())
    if missing:
        raise ValueError("checkpoint is missing required keys: " + ", ".join(missing))
    if checkpoint["next_iter"] != metadata.get("next_iter"):
        raise ValueError("checkpoint and metadata next_iter disagree")
    if checkpoint["run_identity"] != metadata.get("run_identity"):
        raise ValueError("checkpoint and metadata run identity disagree")
    if checkpoint["block_fht_cache_state"] != "flushed_not_serialized":
        raise ValueError("checkpoint cache state is not resume-safe")
    if checkpoint.get("train_data_generator_state") is None:
        raise ValueError("checkpoint lacks the pre-current-batch data-generator state")
    if not isinstance(checkpoint.get("cpu_torch_rng_state"), torch.Tensor):
        raise ValueError("checkpoint CPU RNG state is invalid")
    model_tensor_bytes = sum(
        value.numel() * value.element_size()
        for value in checkpoint["model"].values()
        if isinstance(value, torch.Tensor)
    )
    return {
        "checkpoint": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "model_tensor_bytes": model_tensor_bytes,
        "next_iter": checkpoint["next_iter"],
        "saved_at_unix": checkpoint["saved_at_unix"],
        "schema_version": checkpoint["schema_version"],
        "readable": True,
        "metadata_consistent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
