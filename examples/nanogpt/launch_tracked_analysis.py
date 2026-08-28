#!/usr/bin/env python3
"""Run one registered analysis command and atomically track terminal state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


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
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")

    cwd = args.cwd.resolve()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    started = time.time()
    status: dict[str, Any] = {
        "schema_version": "tracked_analysis_v1",
        "state": "running",
        "run_label": args.run_label,
        "started_at_unix": started,
        "supervisor_pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "git_commit": commit,
        "literal_command": command,
        "cwd": str(cwd),
        "log": str(args.log),
        "result": str(args.result),
        "plan": {
            "path": str(args.plan) if args.plan else None,
            "sha256": sha256(args.plan) if args.plan else None,
        },
    }
    atomic_json(args.status, status)

    with args.log.open("wb") as log_handle:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        status["child_pid"] = child.pid
        atomic_json(args.status, status)
        exit_code = child.wait()

    finished = time.time()
    status.update(
        {
            "state": "finished" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "finished_at_unix": finished,
            "elapsed_seconds": finished - started,
            "log_sha256": sha256(args.log),
            "result_exists": args.result.is_file(),
            "result_sha256": sha256(args.result),
        }
    )
    atomic_json(args.status, status)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
