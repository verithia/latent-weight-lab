#!/usr/bin/env python3
"""Run one registered training config and atomically maintain terminal status."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    command = [
        sys.executable,
        "-u",
        "examples/nanogpt/train.py",
        "--config",
        str(config_path),
    ]
    started = time.time()
    status: dict[str, Any] = {
        "schema_version": "tracked_nanogpt_training_v1",
        "state": "running",
        "run_label": args.run_label,
        "started_at_unix": started,
        "supervisor_pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "git_commit": commit,
        "entrypoint": "examples/nanogpt/train.py",
        "literal_command": command,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "dataset_manifest": {
            "path": str(Path(config["data_dir"]) / "manifest.json"),
            "sha256": sha256(Path(config["data_dir"]) / "manifest.json"),
        },
        "log": str(args.log),
        "out_dir": str(out_dir),
    }
    atomic_json(args.status, status)

    with args.log.open("wb") as log_handle:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        status["child_pid"] = child.pid
        atomic_json(args.status, status)
        exit_code = child.wait()

    finished = time.time()
    checkpoint = out_dir / "ckpt.pt"
    metadata = out_dir / "ckpt.meta.json"
    status.update(
        {
            "state": "finished" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "finished_at_unix": finished,
            "elapsed_seconds": finished - started,
            "log_sha256": sha256(args.log),
            "checkpoint": {
                "path": str(checkpoint),
                "exists": checkpoint.is_file(),
                "sha256": sha256(checkpoint),
            },
            "checkpoint_metadata": {
                "path": str(metadata),
                "exists": metadata.is_file(),
                "sha256": sha256(metadata),
            },
        }
    )
    atomic_json(args.status, status)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
