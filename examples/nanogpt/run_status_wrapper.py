#!/usr/bin/env python3
"""Run one command with an atomic machine-readable terminal status."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")

    started = time.time()
    base: dict[str, Any] = {
        "schema_version": "nanogpt_command_status_v1",
        "state": "running",
        "pid": os.getpid(),
        "process_group": os.getpgrp(),
        "started_at_unix": started,
        "cwd": str(args.cwd),
        "command": command,
        "log": str(args.log),
    }
    atomic_json(args.status, base)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=args.cwd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        exit_code = int(completed.returncode)
    except BaseException as error:
        exit_code = 125
        base["wrapper_error"] = repr(error)
    finished = time.time()
    base.update(
        {
            "state": "finished" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "finished_at_unix": finished,
            "wall_seconds": finished - started,
        }
    )
    atomic_json(args.status, base)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
