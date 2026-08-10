#!/usr/bin/env python3
"""Extract a compact, exact same-gauge sparse-MoE geometry snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "nanogpt_moe_paired_snapshot_v1"
TENSOR_SUFFIXES = (
    "mlp.expert_c_fc",
    "mlp.expert_c_proj",
    "mlp.router.weight",
)


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


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item) for item in value.split(",") if item.strip()})
    if not layers or layers[0] < 0:
        raise ValueError("layers must be a non-empty comma-separated list")
    return layers


def selected_keys(layers: list[int]) -> list[str]:
    return [
        f"transformer.h.{layer}.{suffix}"
        for layer in layers
        for suffix in TENSOR_SUFFIXES
    ]


def validate_layer_tensors(model: dict[str, torch.Tensor], layer: int) -> None:
    prefix = f"transformer.h.{layer}.mlp."
    c_fc = model[prefix + "expert_c_fc"]
    c_proj = model[prefix + "expert_c_proj"]
    router = model[prefix + "router.weight"]
    if c_fc.ndim != 3 or c_proj.ndim != 3 or router.ndim != 2:
        raise ValueError(f"layer {layer} sparse-MoE tensor ranks are invalid")
    experts, hidden, embedding = c_fc.shape
    if c_proj.shape != (experts, embedding, hidden):
        raise ValueError(f"layer {layer} c_fc/c_proj paired shapes are invalid")
    if router.shape != (experts, embedding):
        raise ValueError(f"layer {layer} router/expert shapes are invalid")


def build_snapshot(source_payload: dict[str, Any], source: Path, layers: list[int]) -> dict[str, Any]:
    model = source_payload.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError("source does not contain a model state")
    keys = selected_keys(layers)
    missing = [key for key in keys if key not in model]
    if missing:
        raise ValueError("source is missing sparse-MoE tensors: " + ", ".join(missing))
    selected = {
        key: model[key].detach().to(device="cpu", copy=True).contiguous()
        for key in keys
    }
    for layer in layers:
        validate_layer_tensors(selected, layer)
    inventory = {
        key: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "numel": value.numel(),
            "bytes": value.numel() * value.element_size(),
        }
        for key, value in sorted(selected.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "step": int(source_payload.get("next_iter", source_payload.get("step", -1))),
        "layers": layers,
        "pairing_semantics": {
            "expert_axis": 0,
            "neuron_axis_c_fc": 1,
            "neuron_axis_c_proj": 2,
            "pair": "expert_c_fc[e,j,:] with expert_c_proj[e,:,j]",
            "scale_gauge": "retained; GELU is not positively homogeneous",
        },
        "model": selected,
        "model_config": source_payload.get("model_config"),
        "run_identity": source_payload.get("run_identity"),
        "execution_provenance": source_payload.get("execution_provenance"),
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
            "schema_version": source_payload.get("schema_version"),
        },
        "tensor_inventory": inventory,
        "created_at_unix": time.time(),
    }


def verify_snapshot(path: Path, expected: dict[str, Any]) -> None:
    observed = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(observed, dict) or observed.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("paired snapshot schema is invalid")
    for field in (
        "step",
        "layers",
        "pairing_semantics",
        "model_config",
        "run_identity",
        "execution_provenance",
        "source",
        "tensor_inventory",
    ):
        if observed.get(field) != expected.get(field):
            raise ValueError(f"paired snapshot {field} mismatch")
    observed_model = observed.get("model")
    if not isinstance(observed_model, dict) or observed_model.keys() != expected["model"].keys():
        raise ValueError("paired snapshot model keys mismatch")
    for key, value in expected["model"].items():
        if not torch.equal(observed_model[key], value):
            raise ValueError(f"paired snapshot tensor mismatch: {key}")


def create_snapshot(source: Path, destination: Path, receipt: Path, layers: list[int]) -> dict[str, Any]:
    if destination.exists() or receipt.exists():
        raise ValueError("destination or receipt already exists")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("source checkpoint is invalid")
    snapshot = build_snapshot(payload, source, layers)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(snapshot, temporary)
    os.replace(temporary, destination)
    verify_snapshot(destination, snapshot)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "state": "verified",
        "schema_version": SCHEMA_VERSION,
        "source": snapshot["source"],
        "snapshot": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
            "step": snapshot["step"],
            "layers": layers,
            "tensor_count": len(snapshot["tensor_inventory"]),
            "tensor_bytes": sum(
                item["bytes"] for item in snapshot["tensor_inventory"].values()
            ),
        },
        "source_deleted": False,
        "verified_at_unix": time.time(),
    }
    atomic_json(receipt, result)
    return result


def reclaim_source(source: Path, destination: Path, receipt: Path) -> dict[str, Any]:
    if source.name == "ckpt.pt":
        raise ValueError("refusing to reclaim a live/resumable ckpt.pt")
    result = json.loads(receipt.read_text())
    if result.get("state") != "verified" or result.get("source_deleted") is not False:
        raise ValueError("receipt is not a verified unreclaimed snapshot")
    if source.stat().st_size != result["source"]["bytes"] or file_sha256(source) != result["source"]["sha256"]:
        raise ValueError("source changed after paired snapshot verification")
    if destination.stat().st_size != result["snapshot"]["bytes"] or file_sha256(destination) != result["snapshot"]["sha256"]:
        raise ValueError("paired snapshot changed before reclaim")
    verify_snapshot(
        destination,
        torch.load(destination, map_location="cpu", weights_only=False),
    )
    source.unlink()
    result.update(
        {
            "state": "reclaimed",
            "source_deleted": True,
            "reclaimed_bytes": result["source"]["bytes"],
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
    parser.add_argument("--layers", default="0,5,11")
    parser.add_argument("--reclaim-source", action="store_true")
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    if args.reclaim_source:
        result = reclaim_source(args.source, args.destination, args.receipt)
    else:
        result = create_snapshot(args.source, args.destination, args.receipt, layers)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
