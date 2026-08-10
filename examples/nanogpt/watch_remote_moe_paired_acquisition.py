#!/usr/bin/env python3
"""Preserve compact paired-MoE snapshots at published checkpoint steps."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_steps(value: str) -> list[int]:
    steps = sorted({int(item) for item in value.split(",") if item.strip()})
    if not steps or steps[0] < 0:
        raise ValueError("steps must be a non-empty comma-separated list")
    return steps


def snapshot_stem(step: int, layers: str = "0,5,11") -> str:
    layer_slug = "_".join(f"l{int(item)}" for item in layers.split(",") if item.strip())
    return f"step_{step:06d}_moe_paired_{layer_slug}"


def ssh(host: str, command: str, timeout: int = 180) -> str:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            host,
            command,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def remote_probe(
    host: str,
    out_dir: str,
    snapshot_dir: str,
    steps: list[int],
    layers: str,
) -> dict[str, Any]:
    names = " ".join(
        shlex.quote(snapshot_stem(step, layers) + ".pt") for step in steps
    )
    script = f"""
set -euo pipefail
python3 - {shlex.quote(out_dir)} {shlex.quote(snapshot_dir)} {names} <<'PY'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1])
snapshot_dir = pathlib.Path(sys.argv[2])
names = sys.argv[3:]
metadata = json.loads((out_dir / 'ckpt.meta.json').read_text())
print(json.dumps({{
    'next_iter': int(metadata['next_iter']),
    'existing': sorted(name for name in names if (snapshot_dir / name).is_file()),
}}))
PY
"""
    return json.loads(ssh(host, script).strip().splitlines()[-1])


def remote_extract(
    host: str,
    out_dir: str,
    snapshot_dir: str,
    extractor: str,
    python: str,
    layers: str,
    step: int,
) -> dict[str, Any]:
    stem = snapshot_stem(step, layers)
    destination = f"{snapshot_dir}/{stem}.pt"
    receipt = f"{snapshot_dir}/{stem}_receipt.json"
    command = " ".join(
        shlex.quote(value)
        for value in (
            python,
            extractor,
            "--source",
            f"{out_dir}/ckpt.pt",
            "--destination",
            destination,
            "--receipt",
            receipt,
            "--layers",
            layers,
        )
    )
    output = ssh(host, command, timeout=600)
    return json.loads(output.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="PRO6")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--layers", default="0,5,11")
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    args = parser.parse_args()
    if args.poll_seconds < 15:
        raise ValueError("poll interval must be at least 15 seconds")
    steps = parse_steps(args.steps)
    state: dict[str, Any] = {
        "schema_version": "remote_moe_paired_acquisition_state_v1",
        "host": args.host,
        "out_dir": args.out_dir,
        "snapshot_dir": args.snapshot_dir,
        "steps": steps,
        "layers": args.layers,
        "completed": {},
        "consecutive_errors": 0,
        "state": "running",
        "started_at_unix": time.time(),
    }
    if args.state.exists():
        previous = json.loads(args.state.read_text())
        if previous.get("steps") != steps or previous.get("out_dir") != args.out_dir:
            raise ValueError("existing acquisition state identity mismatch")
        state.update(previous)
        state["state"] = "running"
    atomic_json(args.state, state)

    expected_names = {
        snapshot_stem(step, args.layers) + ".pt": step for step in steps
    }
    while len(state["completed"]) < len(steps):
        try:
            sample = remote_probe(
                args.host,
                args.out_dir,
                args.snapshot_dir,
                steps,
                args.layers,
            )
            for name in sample["existing"]:
                state["completed"].setdefault(str(expected_names[name]), {"state": "present"})
            current = int(sample["next_iter"])
            if current in steps and str(current) not in state["completed"]:
                result = remote_extract(
                    args.host,
                    args.out_dir,
                    args.snapshot_dir,
                    args.extractor,
                    args.remote_python,
                    args.layers,
                    current,
                )
                if result.get("state") != "verified" or result["snapshot"]["step"] != current:
                    raise ValueError(f"invalid extraction result for step {current}: {result}")
                state["completed"][str(current)] = result["snapshot"]
            state.update(
                {
                    "last_published_next_iter": current,
                    "consecutive_errors": 0,
                    "last_error": None,
                    "updated_at_unix": time.time(),
                }
            )
            atomic_json(args.state, state)
        except Exception as error:
            state["consecutive_errors"] = int(state.get("consecutive_errors", 0)) + 1
            state["last_error"] = repr(error)
            state["updated_at_unix"] = time.time()
            atomic_json(args.state, state)
            if state["consecutive_errors"] >= args.max_consecutive_errors:
                state["state"] = "failed"
                atomic_json(args.state, state)
                raise
        if len(state["completed"]) < len(steps):
            time.sleep(args.poll_seconds)

    state["state"] = "finished"
    state["finished_at_unix"] = time.time()
    atomic_json(args.state, state)
    print(json.dumps(state, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
