#!/usr/bin/env python3
"""Launch a local command in its own session and write an atomic PID receipt."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")

    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    with args.stdout.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    atomic_json(
        args.receipt,
        {
            "schema_version": "local_detached_process_receipt_v1",
            "pid": process.pid,
            "command": command,
            "cwd": str(args.cwd.resolve()),
            "stdout": str(args.stdout.resolve()),
            "launched_at_unix": time.time(),
            "start_new_session": True,
        },
    )
    print(process.pid)


if __name__ == "__main__":
    main()
