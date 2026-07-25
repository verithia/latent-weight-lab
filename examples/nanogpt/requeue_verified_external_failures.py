#!/usr/bin/env python3
"""Guardedly requeue externally failed runs after their resume envelopes are verified."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


STALE_RUNTIME_KEYS = {
    "gpu",
    "last_progress_at",
    "missing_since",
    "pgid",
    "run_name",
    "session",
    "stall_notified_marker",
    "status_path",
    "submission_error_pending",
    "submit_log",
    "submitted_at",
    "terminal_notified",
    "terminal_pending",
    "terminal_signature",
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_verified(values: list[str]) -> dict[str, int]:
    observed: dict[str, int] = {}
    for value in values:
        name, separator, raw_iter = value.partition("=")
        if not separator or not name or name in observed:
            raise ValueError("verified checkpoint must be a unique TASK=NEXT_ITER value")
        observed[name] = int(raw_iter)
    return observed


def requeue(
    state: dict[str, Any],
    manifest: dict[str, Any],
    verified_next_iters: dict[str, int],
    host: str,
) -> list[dict[str, Any]]:
    tasks = {task["name"]: task for task in manifest["entries"]}
    unknown = sorted(set(verified_next_iters) - tasks.keys())
    if unknown:
        raise ValueError("unknown queue tasks: " + ", ".join(unknown))
    if not verified_next_iters:
        raise ValueError("no verified checkpoints supplied")
    records: list[dict[str, Any]] = []
    for name, verified_next_iter in verified_next_iters.items():
        variant = tasks[name].get("variants", {}).get(host)
        if not variant or variant.get("resume") is not True:
            raise ValueError(f"task is not an enabled {host} resume: {name}")
        expected_next_iter = variant.get("expected_checkpoint_next_iter")
        if expected_next_iter != verified_next_iter:
            raise ValueError(
                f"verified checkpoint mismatch for {name}: "
                f"expected {expected_next_iter}, observed {verified_next_iter}"
            )
        runtime = state.get("entries", {}).get(name)
        if not runtime or runtime.get("state") != "failed_external":
            raise ValueError(f"task is not in failed_external state: {name}")
        if runtime.get("terminal_notified") is not True:
            raise ValueError(f"external failure has not been reported: {name}")
        attempts = dict(runtime.get("attempts_by_host", {}))
        milestones = list(runtime.get("sent_milestones", []))
        for key in STALE_RUNTIME_KEYS:
            runtime.pop(key, None)
        runtime.update(
            {
                "state": "pending",
                "assigned_host": None,
                "last_iter": verified_next_iter,
                "attempts_by_host": attempts,
                "sent_milestones": milestones,
                "rejected_hosts": [],
            }
        )
        records.append(
            {
                "task": name,
                "host": host,
                "next_iter": verified_next_iter,
                "prior_attempts": attempts.get(host, 0),
                "preserved_milestones": milestones,
            }
        )
    state["updated_at"] = time.time()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--host", default="Y400")
    parser.add_argument("--verified", action="append", default=[], metavar="TASK=NEXT_ITER")
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    manifest = json.loads(args.queue.read_text())
    records = requeue(state, manifest, parse_verified(args.verified), args.host)
    atomic_json(args.state, state)
    print(json.dumps({"state": "requeued", "entries": records}, sort_keys=True))


if __name__ == "__main__":
    main()
