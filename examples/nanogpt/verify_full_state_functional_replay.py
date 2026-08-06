#!/usr/bin/env python3
"""Verify that full-state trajectory snapshots replay their logged function."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.parameter_trajectory import FULL_STATE_SCHEMA_VERSION
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    get_batch,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)


LOSS_PATTERN = re.compile(
    r"^step (?P<step>\d+): train loss "
    r"(?P<train>[-+0-9.eE]+), val loss (?P<val>[-+0-9.eE]+)$"
)
REQUIRED_STEPS = (0, 594, 1188, 1782, 2373)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_logged_losses(path: Path) -> dict[int, dict[str, float]]:
    losses: dict[int, dict[str, float]] = {}
    for raw in path.read_text(errors="replace").splitlines():
        match = LOSS_PATTERN.match(raw.strip())
        if match is None:
            continue
        step = int(match.group("step"))
        losses[step] = {
            "train": float(match.group("train")),
            "val": float(match.group("val")),
        }
    missing = sorted(set(REQUIRED_STEPS) - set(losses))
    if missing:
        raise ValueError(f"training log is missing fixed losses: {missing}")
    return losses


def expected_buffer_names(n_layer: int) -> set[str]:
    return {
        f"transformer.h.{layer}.mlp.c_fc.{suffix}"
        for layer in range(n_layer)
        for suffix in ("weight", "optimizer_step")
    }


def validate_full_state_inventory(
    snapshot: dict[str, Any], *, n_layer: int
) -> None:
    if snapshot.get("schema_version") != FULL_STATE_SCHEMA_VERSION:
        raise ValueError("snapshot is not full-state trajectory v2")
    if snapshot.get("all_parameters") is not True:
        raise ValueError("snapshot does not contain all named parameters")
    if snapshot.get("all_buffers") is not True:
        raise ValueError("snapshot does not contain all persistent buffers")
    parameters = snapshot.get("parameters")
    buffers = snapshot.get("buffers")
    if not isinstance(parameters, dict) or len(parameters) != 327:
        raise ValueError("snapshot named-parameter inventory is not exactly 327")
    expected = expected_buffer_names(n_layer)
    if not isinstance(buffers, dict) or set(buffers) != expected:
        missing = sorted(expected - set(buffers or {}))
        unexpected = sorted(set(buffers or {}) - expected)
        raise ValueError(
            "snapshot persistent-buffer inventory mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    for name, value in parameters.items():
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"invalid parameter tensor: {name}")
    for name, value in buffers.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"non-finite persistent buffer: {name}")
        if name.endswith(".optimizer_step") and value.dtype != torch.int64:
            raise ValueError(f"optimizer step dtype was not preserved: {name}")


@torch.no_grad()
def evaluate_validation_ce(
    model: torch.nn.Module,
    *,
    data_dir: Path,
    args: SimpleNamespace,
    indices: torch.Tensor,
    ctx: Any,
) -> float:
    source = TokenBatchSource(data_dir)
    losses = torch.zeros(args.eval_iters)
    model.eval()
    use_cache = bool(args.block_fht_cache_weights and "cuda" in args.device)
    if use_cache:
        model.prepare_block_fht_cache(dtype=args._ptdtype)
    try:
        for index in range(args.eval_iters):
            x, y = get_batch(
                data_dir,
                "val",
                args.eval_batch_size,
                args.block_size,
                args.device,
                indices=indices[index],
                source=source,
            )
            with ctx:
                _logits, loss = model(x, y)
            losses[index] = loss.item()
    finally:
        if use_cache:
            model.flush_block_fht_cache()
    return float(losses.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.output.exists():
        raise FileExistsError(f"output already exists: {cli.output}")
    if cli.tolerance != 0.005:
        raise ValueError("the preregistered functional-replay tolerance is 0.005")
    config = json.loads(cli.config.read_text())
    if config.get("trajectory_snapshot_all_buffers") is not True:
        raise ValueError("config did not register full-state trajectory snapshots")
    if [int(value) for value in REQUIRED_STEPS] != [
        0,
        int(config["eval_interval"]),
        2 * int(config["eval_interval"]),
        3 * int(config["eval_interval"]),
        int(config["max_iters"]),
    ]:
        raise ValueError("config phase boundaries disagree with the v3 contract")
    require_block_fht_native_extension(
        bool(config["block_fht_native_extension_required"])
    )
    data_dir = Path(config["data_dir"])
    manifest = data_dir / "manifest.json"
    if file_sha256(manifest) != config["data_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    fixed = make_fixed_eval_indices(
        data_dir,
        int(config["eval_batch_size"]),
        int(config["block_size"]),
        int(config["eval_iters"]),
        int(config["eval_seed"]),
    )
    fixed_digest = fixed_eval_indices_digest(fixed)
    expected_fixed_digest = (
        "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
    )
    if fixed_digest != expected_fixed_digest:
        raise ValueError("fixed evaluation indices SHA-256 mismatch")

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    args = SimpleNamespace(**config)
    args._ptdtype = dtype
    ctx = (
        contextlib.nullcontext()
        if "cuda" not in args.device
        else torch.amp.autocast(device_type="cuda", dtype=dtype)
    )
    logged = parse_logged_losses(cli.training_log)
    started = time.time()
    rows = []
    identity = None
    for step in REQUIRED_STEPS:
        path = cli.snapshot_dir / f"step_{step:06d}.pt"
        snapshot = load_snapshot(path)
        validate_full_state_inventory(snapshot, n_layer=int(config["n_layer"]))
        if int(snapshot["step"]) != step:
            raise ValueError(f"snapshot step mismatch: {path}")
        if identity is None:
            identity = snapshot["run_identity_sha256"]
        elif snapshot["run_identity_sha256"] != identity:
            raise ValueError("snapshot run identities disagree")
        model = model_from_snapshot(snapshot, args.device)
        replay = evaluate_validation_ce(
            model,
            data_dir=data_dir,
            args=args,
            indices=fixed["val"],
            ctx=ctx,
        )
        delta = replay - logged[step]["val"]
        rows.append(
            {
                "step": step,
                "snapshot": str(path),
                "snapshot_sha256": file_sha256(path),
                "logged_validation_ce": logged[step]["val"],
                "replayed_validation_ce": replay,
                "replay_minus_logged_validation_ce": delta,
                "absolute_delta_ce": abs(delta),
                "within_tolerance": abs(delta) <= cli.tolerance,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passed = all(row["within_tolerance"] for row in rows)
    result = {
        "schema_version": "nanogpt_full_state_functional_replay_result_v1",
        "classification": (
            "ACCEPTED_EXACT_FUNCTIONAL_REPLAY"
            if passed
            else "REJECTED_FUNCTIONAL_REPLAY_MISMATCH"
        ),
        "passed": passed,
        "config": str(cli.config),
        "config_sha256": file_sha256(cli.config),
        "training_log": str(cli.training_log),
        "training_log_sha256": file_sha256(cli.training_log),
        "run_identity_sha256": identity,
        "dataset_manifest_sha256": file_sha256(manifest),
        "fixed_eval_indices_sha256": fixed_digest,
        "tolerance_ce": cli.tolerance,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "authorization": {
            "seal_full_state_acquisition": passed,
            "run_metric_calibration": passed,
            "implement_candidate_structure": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
