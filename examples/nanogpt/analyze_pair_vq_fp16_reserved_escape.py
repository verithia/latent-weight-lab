#!/usr/bin/env python3
"""Audit a bit-exact scope-adaptive FP16 high-byte reserved-escape codec."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from examples.nanogpt.analyze_pair_vq_fp16_highbyte_dictionary import top_k_table
from examples.nanogpt.analyze_pair_vq_fp16_momentum_entropy import (
    _extract_ambient,
    _sha256,
    _tensor_words,
)


SCHEMA = "mai_pair_vq_fp16_reserved_escape_capacity_result_v1"
K_FC = (8, 16, 32)
K_PROJ = (16, 32, 64)
BLOCK_WORDS = (256, 1024, 4096)
GRANULARITIES = ("scope", "matrix", "block")
CONTAINER_METADATA_BYTES = 64 + 24 * 16


def _histogram(high: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(high, dtype=np.uint8), minlength=256)


def _k_for(item: dict, k_fc: int, k_proj: int) -> int:
    return k_fc if item["scope"] == "c_fc" else k_proj


def _table_map(table: np.ndarray) -> np.ndarray:
    mapping = np.full(256, -1, dtype=np.int16)
    mapping[np.asarray(table, dtype=np.uint8)] = np.arange(table.size, dtype=np.int16)
    return mapping


def _region_tables(
    items: list[dict], *, k_fc: int, k_proj: int, granularity: str
) -> dict:
    if granularity == "scope":
        result = {}
        for scope in ("c_fc", "c_proj"):
            k = k_fc if scope == "c_fc" else k_proj
            counts = sum(
                (_histogram(item["high"]) for item in items if item["scope"] == scope),
                np.zeros(256, dtype=np.int64),
            )
            result[scope] = top_k_table(counts, k - 1)
        return result
    if granularity == "matrix":
        return {
            item["name"]: top_k_table(
                _histogram(item["high"]), _k_for(item, k_fc, k_proj) - 1
            )
            for item in items
        }
    if granularity != "block":
        raise ValueError(f"unknown granularity {granularity}")
    return {}


def encode_item(
    item: dict,
    *,
    k: int,
    block_words: int,
    granularity: str,
    region_tables: dict,
) -> tuple[dict, dict]:
    high = item["high"]
    count = high.size
    code_bits = int(math.log2(k))
    escape_code = k - 1
    indices = np.full(count, escape_code, dtype=np.uint8)
    tables: list[np.ndarray] = []
    literals: list[np.ndarray] = []
    block_offsets = [0]
    shared_table = None
    if granularity == "scope":
        shared_table = region_tables[item["scope"]]
    elif granularity == "matrix":
        shared_table = region_tables[item["name"]]
    for start in range(0, count, block_words):
        stop = min(start + block_words, count)
        block = high[start:stop]
        table = (
            top_k_table(_histogram(block), k - 1)
            if granularity == "block"
            else shared_table
        )
        assert table is not None
        mapped = _table_map(table)[block]
        escaped = mapped < 0
        indices[start:stop] = np.where(escaped, escape_code, mapped).astype(np.uint8)
        literals.append(block[escaped].copy())
        block_offsets.append(block_offsets[-1] + int(escaped.sum()))
        if granularity == "block":
            tables.append(table.copy())
    literal_stream = np.concatenate(literals) if literals else np.empty(0, dtype=np.uint8)
    encoded = {
        "low": item["low"].copy(),
        "indices": indices,
        "literals": literal_stream,
        "block_offsets": np.asarray(block_offsets, dtype=np.uint32),
        "tables": tables,
        "shared_table": None if shared_table is None else shared_table.copy(),
        "escape_code": escape_code,
    }
    blocks = math.ceil(count / block_words)
    account = {
        "coordinates": count,
        "low_literal_bytes": count,
        "index_bytes": math.ceil(count * code_bits / 8),
        "exception_literal_bytes": int(literal_stream.size),
        "dictionary_bytes": blocks * (k - 1) if granularity == "block" else 0,
        "block_offset_bytes": (blocks + 1) * 4,
        "exceptions": int(literal_stream.size),
        "blocks": blocks,
    }
    return encoded, account


def decode_item(
    encoded: dict, *, count: int, block_words: int, granularity: str
) -> np.ndarray:
    high = np.empty(count, dtype=np.uint8)
    for block_index, start in enumerate(range(0, count, block_words)):
        stop = min(start + block_words, count)
        table = (
            encoded["tables"][block_index]
            if granularity == "block"
            else encoded["shared_table"]
        )
        codes = encoded["indices"][start:stop]
        escaped = codes == encoded["escape_code"]
        decoded = np.empty(codes.size, dtype=np.uint8)
        decoded[~escaped] = table[codes[~escaped]]
        literal_start = int(encoded["block_offsets"][block_index])
        literal_stop = int(encoded["block_offsets"][block_index + 1])
        decoded[escaped] = encoded["literals"][literal_start:literal_stop]
        high[start:stop] = decoded
    return (high.astype(np.uint16) << 8) | encoded["low"].astype(np.uint16)


def evaluate_candidate(
    items: list[dict], *, k_fc: int, k_proj: int, block_words: int, granularity: str
) -> dict:
    tables = _region_tables(
        items, k_fc=k_fc, k_proj=k_proj, granularity=granularity
    )
    start = time.perf_counter()
    encoded_items = []
    accounts = []
    for item in items:
        encoded, account = encode_item(
            item,
            k=_k_for(item, k_fc, k_proj),
            block_words=block_words,
            granularity=granularity,
            region_tables=tables,
        )
        encoded_items.append(encoded)
        accounts.append(account)
    encode_seconds = time.perf_counter() - start

    start = time.perf_counter()
    digest = hashlib.sha256()
    exact = True
    for item, encoded in zip(items, encoded_items, strict=True):
        recovered = decode_item(
            encoded,
            count=item["words"].size,
            block_words=block_words,
            granularity=granularity,
        )
        exact = exact and bool(np.array_equal(recovered, item["words"]))
        digest.update(recovered.astype("<u2", copy=False).tobytes())
    decode_seconds = time.perf_counter() - start

    dictionary_bytes = sum(row["dictionary_bytes"] for row in accounts)
    if granularity != "block":
        dictionary_bytes = sum(table.size for table in tables.values())
    components = {
        key: sum(row[key] for row in accounts)
        for key in (
            "low_literal_bytes",
            "index_bytes",
            "exception_literal_bytes",
            "block_offset_bytes",
        )
    }
    components["dictionary_bytes"] = dictionary_bytes
    components["container_metadata_bytes"] = CONTAINER_METADATA_BYTES
    total_bytes = sum(components.values())
    elements = sum(item["words"].size for item in items)
    scope_escape = {}
    for scope in ("c_fc", "c_proj"):
        positions = [i for i, item in enumerate(items) if item["scope"] == scope]
        scope_count = sum(accounts[i]["coordinates"] for i in positions)
        escaped = sum(accounts[i]["exceptions"] for i in positions)
        scope_escape[scope] = {
            "coordinates": scope_count,
            "exceptions": escaped,
            "escape_fraction": escaped / scope_count,
        }
    return {
        "name": f"{granularity}_kfc{k_fc}_kproj{k_proj}_b{block_words}",
        "granularity": granularity,
        "k_fc": k_fc,
        "k_proj": k_proj,
        "block_words": block_words,
        "components": components,
        "total_bytes": total_bytes,
        "bits_per_coordinate": total_bytes * 8.0 / elements,
        "scope_escape": scope_escape,
        "encode_seconds_reference": encode_seconds,
        "decode_seconds_reference": decode_seconds,
        "exact_roundtrip": exact,
        "decoded_word_stream_sha256": digest.hexdigest(),
    }


def select(candidates: list[dict], target_bytes: int) -> tuple[dict | None, dict | None]:
    passing = [row for row in candidates if row["exact_roundtrip"] and row["total_bytes"] <= target_bytes]
    if not passing:
        return None, None
    granularity_rank = {name: i for i, name in enumerate(GRANULARITIES)}
    primary = min(
        passing,
        key=lambda row: (
            row["total_bytes"],
            -row["block_words"],
            granularity_rank[row["granularity"]],
            row["k_fc"] + row["k_proj"],
        ),
    )
    near = [row for row in passing if row["total_bytes"] <= primary["total_bytes"] * 1.02]
    control = min(
        near,
        key=lambda row: (
            -row["block_words"],
            granularity_rank[row["granularity"]],
            row["k_fc"] + row["k_proj"],
            row["total_bytes"],
        ),
    )
    return primary, control


def run_audit(checkpoint_path: Path, checkpoint_sha256: str, plan_path: Path) -> dict:
    observed_sha = _sha256(checkpoint_path)
    if observed_sha != checkpoint_sha256:
        raise RuntimeError(f"checkpoint SHA mismatch: {observed_sha}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tensors = _extract_ambient(checkpoint)
    items = []
    source_digest = hashlib.sha256()
    for item in tensors:
        words = _tensor_words(item["tensor"]).copy()
        source_digest.update(words.astype("<u2", copy=False).tobytes())
        items.append(
            {
                **{key: item[key] for key in ("name", "scope", "layer", "shape")},
                "words": words,
                "low": (words & 255).astype(np.uint8),
                "high": (words >> 8).astype(np.uint8),
            }
        )
    if len(items) != 24 or sum(item["words"].size for item in items) != 56623104:
        raise RuntimeError("registered source shape/count mismatch")

    candidates = []
    for granularity in GRANULARITIES:
        for k_fc in K_FC:
            for k_proj in K_PROJ:
                for block_words in BLOCK_WORDS:
                    row = evaluate_candidate(
                        items,
                        k_fc=k_fc,
                        k_proj=k_proj,
                        block_words=block_words,
                        granularity=granularity,
                    )
                    if row["decoded_word_stream_sha256"] != source_digest.hexdigest():
                        raise RuntimeError(f"digest mismatch for {row['name']}")
                    candidates.append(row)

    target_bytes = 99090432
    primary, control = select(candidates, target_bytes)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema_version": SCHEMA,
        "status": "finished",
        "classification": (
            "PASS_BIT_EXACT_RESERVED_ESCAPE_CAPACITY"
            if primary is not None
            else "REJECT_BIT_EXACT_RESERVED_ESCAPE_CAPACITY"
        ),
        "passed": primary is not None,
        "identity": {
            "source_commit": commit,
            "plan": str(plan_path),
            "plan_sha256": _sha256(plan_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": observed_sha,
            "source_word_stream_sha256": source_digest.hexdigest(),
        },
        "gate": {
            "maximum_momentum_bytes": target_bytes,
            "maximum_bits_per_coordinate": 14.0,
            "passing_candidates": sum(row["total_bytes"] <= target_bytes for row in candidates),
            "primary": None if primary is None else primary["name"],
            "performance_control": None if control is None else control["name"],
        },
        "primary": primary,
        "performance_control": control,
        "candidates": candidates,
        "storage": (
            None
            if primary is None
            else {
                "common_compact_forward_and_feedback_bytes": 157427136,
                "total_training_bytes": 157427136 + primary["total_bytes"],
                "training_reduction_factor": 452984832 / (157427136 + primary["total_bytes"]),
                "working_fp16_training_reduction_factor": 1.6735479944415952,
            }
        ),
        "decision": {
            "automatic_endpoint": False,
            "automatic_scale_up": False,
            "automatic_horizon_transfer": False,
            "automatic_sweep": False,
            "next_authorized_work": (
                "Implement only primary and distinct performance-control CUDA codecs, then require exact state/resume tests and >=20% exact-config MFU."
                if primary is not None
                else "Reject fixed-width table codecs and preregister a finite-state entropy or temporal/shared predictor audit."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.checkpoint, args.checkpoint_sha256, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"classification": result["classification"], "gate": result["gate"], "storage": result["storage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
