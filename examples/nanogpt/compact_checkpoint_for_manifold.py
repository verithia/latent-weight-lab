#!/usr/bin/env python3
"""Preserve exact model tensors before reclaiming superseded resume payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "nanogpt_manifold_snapshot_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tensor_inventory(model: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, value in sorted(model.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"model state contains a non-tensor value: {name}")
        records[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "numel": value.numel(),
            "bytes": value.numel() * value.element_size(),
        }
    return records


def build_snapshot(
    checkpoint: dict[str, Any],
    source: Path,
    allow_legacy_missing_provenance: bool = False,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "model",
        "model_config",
        "next_iter",
        "best_val_loss",
        "run_identity",
        "saved_at_unix",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError("checkpoint is missing snapshot fields: " + ", ".join(missing))
    legacy_missing_fields: list[str] = []
    if "execution_provenance" not in checkpoint:
        if not allow_legacy_missing_provenance:
            raise ValueError("checkpoint is missing snapshot fields: execution_provenance")
        legacy_missing_fields.append("execution_provenance")
    if not isinstance(checkpoint["model"], dict) or not checkpoint["model"]:
        raise ValueError("checkpoint model state is invalid")
    inventory = tensor_inventory(checkpoint["model"])
    return {
        "schema_version": SCHEMA_VERSION,
        "model": checkpoint["model"],
        "model_config": checkpoint["model_config"],
        "next_iter": checkpoint["next_iter"],
        "best_val_loss": checkpoint["best_val_loss"],
        "run_identity": checkpoint["run_identity"],
        "execution_provenance": checkpoint.get("execution_provenance"),
        "legacy_missing_fields": legacy_missing_fields,
        "source_checkpoint": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
            "schema_version": checkpoint["schema_version"],
            "saved_at_unix": checkpoint["saved_at_unix"],
        },
        "tensor_inventory": inventory,
        "created_at_unix": time.time(),
    }


def verify_snapshot(snapshot_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    observed = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if not isinstance(observed, dict) or observed.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifold snapshot schema is invalid")
    for field in (
        "model_config",
        "next_iter",
        "best_val_loss",
        "run_identity",
        "execution_provenance",
        "legacy_missing_fields",
        "source_checkpoint",
        "tensor_inventory",
    ):
        if observed.get(field) != expected.get(field):
            raise ValueError(f"manifold snapshot {field} mismatch")
    expected_model = expected["model"]
    observed_model = observed.get("model")
    if not isinstance(observed_model, dict) or observed_model.keys() != expected_model.keys():
        raise ValueError("manifold snapshot model keys mismatch")
    for name, expected_tensor in expected_model.items():
        if not torch.equal(observed_model[name], expected_tensor):
            raise ValueError(f"manifold snapshot tensor mismatch: {name}")
    return observed


def create_snapshot(
    source: Path,
    destination: Path,
    receipt: Path,
    allow_legacy_missing_provenance: bool = False,
) -> dict[str, Any]:
    if destination.exists() or receipt.exists():
        raise ValueError("destination or receipt already exists")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is invalid")
    snapshot = build_snapshot(checkpoint, source, allow_legacy_missing_provenance)
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(snapshot, temporary)
    os.replace(temporary, destination)
    verify_snapshot(destination, snapshot)
    inventory = snapshot["tensor_inventory"]
    result = {
        "state": "verified",
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": snapshot["source_checkpoint"],
        "snapshot": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
            "next_iter": snapshot["next_iter"],
            "tensor_count": len(inventory),
            "tensor_bytes": sum(item["bytes"] for item in inventory.values()),
            "legacy_missing_fields": snapshot["legacy_missing_fields"],
        },
        "source_deleted": False,
        "verified_at_unix": time.time(),
    }
    atomic_json(receipt, result)
    return result


def reclaim_source(source: Path, destination: Path, receipt: Path) -> dict[str, Any]:
    result = json.loads(receipt.read_text())
    if result.get("state") != "verified" or result.get("source_deleted") is not False:
        raise ValueError("receipt is not a verified unreclaimed snapshot")
    source_expected = result["source_checkpoint"]
    snapshot_expected = result["snapshot"]
    if source.stat().st_size != source_expected["bytes"] or file_sha256(source) != source_expected["sha256"]:
        raise ValueError("source checkpoint changed after snapshot verification")
    if (
        destination.stat().st_size != snapshot_expected["bytes"]
        or file_sha256(destination) != snapshot_expected["sha256"]
    ):
        raise ValueError("manifold snapshot changed before reclaim")
    observed = torch.load(destination, map_location="cpu", weights_only=False)
    if observed.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifold snapshot became unreadable before reclaim")
    source.unlink()
    if source.exists():
        raise ValueError("source checkpoint remained after reclaim")
    result.update(
        {
            "state": "reclaimed",
            "source_deleted": True,
            "reclaimed_bytes": source_expected["bytes"],
            "finished_at_unix": time.time(),
        }
    )
    atomic_json(receipt, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--reclaim-source", action="store_true")
    parser.add_argument(
        "--allow-legacy-missing-provenance",
        action="store_true",
        help="Permit only a legacy missing execution_provenance field and record that absence explicitly.",
    )
    args = parser.parse_args()
    if args.reclaim_source:
        result = reclaim_source(args.source, args.destination, args.receipt)
    else:
        result = create_snapshot(
            args.source,
            args.destination,
            args.receipt,
            allow_legacy_missing_provenance=args.allow_legacy_missing_provenance,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
