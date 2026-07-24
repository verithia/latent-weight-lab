#!/usr/bin/env python3
"""Archive selected completed remote directories and verify them byte-for-byte.

The selected sources must be direct children of one remote root. The source is
never deleted: a later operator may reclaim it only after the terminal state is
``verified`` and an independent deletion guard rechecks both manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


CALLBACK_URL = "http://127.0.0.1:8766/send-opencode-test"
AGENT_MENTION = "@Codex"
REMOTE_MANIFEST_SCRIPT = r'''import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
names = sys.argv[2:]
result = {}
for name in names:
    selected = root / name
    if not selected.is_dir():
        raise SystemExit(f"selected source is not a directory: {selected}")
    for path in sorted(item for item in selected.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
print(json.dumps(result, sort_keys=True))'''


def validate_names(names: Iterable[str]) -> list[str]:
    selected = list(names)
    if not selected:
        raise ValueError("at least one --include-name is required")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate --include-name values are not allowed")
    for name in selected:
        if not name or name in {".", ".."} or "/" in name or "\0" in name:
            raise ValueError(f"include names must be direct child names: {name!r}")
    return selected


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def send(chat_id: str, message: str) -> bool:
    request = urllib.request.Request(
        CALLBACK_URL,
        data=json.dumps(
            {"chat_id": chat_id, "text": f"{AGENT_MENTION} {message}"}
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        opener.open(request, timeout=20).read()
        return True
    except OSError:
        return False


def local_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def local_manifest(root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return result


def remote_bytes(host: str, source_root: str, names: Sequence[str]) -> int:
    paths = [f"{source_root.rstrip('/')}/{name}" for name in names]
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "Compression=no",
            "-o",
            "ConnectTimeout=20",
            host,
            "du",
            "-sb",
            *paths,
        ],
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    return sum(int(line.split()[0]) for line in completed.stdout.splitlines())


def remote_manifest(
    host: str, source_root: str, names: Sequence[str]
) -> Dict[str, Any]:
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "Compression=no",
            "-o",
            "ConnectTimeout=20",
            host,
            "python3",
            "-",
            source_root,
            *names,
        ],
        input=REMOTE_MANIFEST_SCRIPT,
        text=True,
        capture_output=True,
        timeout=3600,
        check=True,
    )
    return json.loads(completed.stdout)


def rsync_command(
    host: str, source_root: str, destination: Path, names: Sequence[str]
) -> list[str]:
    includes: list[str] = []
    for name in names:
        includes.extend((f"--include=/{name}/", f"--include=/{name}/***"))
    return [
        "/usr/bin/rsync",
        "-a",
        "--partial",
        "--append",
        "-e",
        "/usr/bin/ssh -o BatchMode=yes -o Compression=no -o ConnectTimeout=20",
        *includes,
        "--exclude=*",
        f"{host}:{source_root.rstrip('/')}/",
        str(destination.resolve()) + "/",
    ]


def partition_names(names: Sequence[str], parallel_transfers: int) -> list[list[str]]:
    transfer_count = min(len(names), parallel_transfers)
    groups: list[list[str]] = [[] for _ in range(transfer_count)]
    for index, name in enumerate(names):
        groups[index % transfer_count].append(name)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--include-name", action="append", default=[])
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--label", default="remote selection archive")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-minutes", type=int, default=90)
    parser.add_argument("--parallel-transfers", type=int, default=1)
    args = parser.parse_args()
    if (
        args.poll_seconds < 15
        or args.heartbeat_minutes < 1
        or args.parallel_transfers < 1
    ):
        parser.error(
            "poll-seconds must be >=15, heartbeat-minutes >=1, and "
            "parallel-transfers >=1"
        )
    try:
        names = validate_names(args.include_name)
    except ValueError as error:
        parser.error(str(error))

    args.destination.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    total = remote_bytes(args.host, args.source_root, names)
    started = time.time()
    state: Dict[str, Any] = {
        "state": "copying",
        "host": args.host,
        "source_root": args.source_root,
        "include_names": names,
        "destination": str(args.destination.resolve()),
        "source_bytes": total,
        "started_at": started,
        "source_deleted": False,
        "sent_milestones": [],
    }
    atomic_json(args.state, state)
    with args.log.open("ab") as log:
        processes = [
            subprocess.Popen(
                rsync_command(
                    args.host,
                    args.source_root,
                    args.destination,
                    group,
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            for group in partition_names(names, args.parallel_transfers)
        ]
        state["parallel_transfers"] = len(processes)
        last_callback = started
        while any(process.poll() is None for process in processes):
            copied = local_bytes(args.destination)
            now = time.time()
            state.update({"copied_bytes": copied, "updated_at": now})
            if copied * 2 >= total and 50 not in state["sent_milestones"]:
                if send(
                    args.chat_id,
                    f"{args.label} PROGRESS: 50% "
                    f"({copied}/{total} bytes copied; verification pending)",
                ):
                    state["sent_milestones"].append(50)
                    last_callback = now
            elif now - last_callback >= args.heartbeat_minutes * 60:
                if send(
                    args.chat_id,
                    f"{args.label} HEARTBEAT: {copied}/{total} bytes copied; "
                    "verification pending",
                ):
                    last_callback = now
            atomic_json(args.state, state)
            time.sleep(args.poll_seconds)
        return_codes = [process.wait() for process in processes]

    if any(return_code != 0 for return_code in return_codes):
        state.update(
            {
                "state": "failed",
                "exit_codes": return_codes,
                "finished_at": time.time(),
            }
        )
        atomic_json(args.state, state)
        send(
            args.chat_id,
            f"{args.label} ERROR: rsync failed exits={return_codes}; "
            "partial archive retained",
        )
        raise SystemExit(next(code for code in return_codes if code != 0))

    state.update(
        {
            "state": "verifying",
            "copied_bytes": local_bytes(args.destination),
            "updated_at": time.time(),
        }
    )
    atomic_json(args.state, state)
    remote = remote_manifest(args.host, args.source_root, names)
    local = local_manifest(args.destination)
    verified = remote == local
    manifest_bytes = json.dumps(
        remote, sort_keys=True, separators=(",", ":")
    ).encode()
    state.update(
        {
            "state": "verified" if verified else "verification_failed",
            "finished_at": time.time(),
            "file_count": len(remote),
            "verified_bytes": sum(item["bytes"] for item in remote.values()),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest": remote,
            "source_deleted": False,
        }
    )
    atomic_json(args.state, state)
    if not verified:
        send(
            args.chat_id,
            f"{args.label} ERROR: source/destination manifest mismatch; "
            "source retained",
        )
        raise SystemExit(3)
    send(
        args.chat_id,
        f"{args.label} PROGRESS: 100% verified {len(remote)} files / "
        f"{state['verified_bytes']} bytes; source retained pending reclaim",
    )


if __name__ == "__main__":
    main()
