#!/usr/bin/env python3
"""Resume a remote archive locally, verify every file, and notify Codex.

The source is never deleted.  A later operator may reclaim it only after the
terminal state says ``verified`` and the two manifests are identical.
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
from typing import Any, Dict


CALLBACK_URL = "http://127.0.0.1:8766/send-opencode-test"
AGENT_MENTION = "@Codex"
REMOTE_MANIFEST_SCRIPT = r'''import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
result = {}
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
print(json.dumps(result, sort_keys=True))'''


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def send(chat_id: str, message: str) -> bool:
    request = urllib.request.Request(
        CALLBACK_URL,
        data=json.dumps({"chat_id": chat_id, "text": f"{AGENT_MENTION} {message}"}).encode(),
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


def remote_bytes(host: str, source: str) -> int:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "Compression=no", "-o", "ConnectTimeout=20", host, "du", "-sb", source],
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    return int(completed.stdout.split()[0])


def remote_manifest(host: str, source: str) -> Dict[str, Any]:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "Compression=no", "-o", "ConnectTimeout=20", host, "python3", "-", source],
        input=REMOTE_MANIFEST_SCRIPT,
        text=True,
        capture_output=True,
        timeout=3600,
        check=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--label", default="remote archive")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--heartbeat-minutes", type=int, default=90)
    args = parser.parse_args()
    if args.poll_seconds < 15 or args.heartbeat_minutes < 1:
        parser.error("poll-seconds must be >=15 and heartbeat-minutes >=1")

    args.destination.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    total = remote_bytes(args.host, args.source)
    started = time.time()
    state: Dict[str, Any] = {
        "state": "copying",
        "host": args.host,
        "source": args.source,
        "destination": str(args.destination.resolve()),
        "source_bytes": total,
        "started_at": started,
        "source_deleted": False,
        "sent_milestones": [],
    }
    atomic_json(args.state, state)
    remote_source = f"{args.host}:{args.source.rstrip('/')}/"
    command = [
        "/usr/bin/rsync", "-a", "--partial", "--append",
        "-e", "/usr/bin/ssh -o BatchMode=yes -o Compression=no -o ConnectTimeout=20",
        remote_source, str(args.destination.resolve()) + "/",
    ]
    with args.log.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        last_callback = started
        while process.poll() is None:
            copied = local_bytes(args.destination)
            now = time.time()
            state.update({"copied_bytes": copied, "updated_at": now})
            if copied * 2 >= total and 50 not in state["sent_milestones"]:
                if send(args.chat_id, f"{args.label} PROGRESS: 50% ({copied}/{total} bytes copied; verification pending)"):
                    state["sent_milestones"].append(50)
                    last_callback = now
            elif now - last_callback >= args.heartbeat_minutes * 60:
                if send(args.chat_id, f"{args.label} HEARTBEAT: {copied}/{total} bytes copied; verification pending"):
                    last_callback = now
            atomic_json(args.state, state)
            time.sleep(args.poll_seconds)
        return_code = process.wait()

    if return_code != 0:
        state.update({"state": "failed", "exit_code": return_code, "finished_at": time.time()})
        atomic_json(args.state, state)
        send(args.chat_id, f"{args.label} ERROR: rsync failed exit={return_code}; partial archive retained")
        raise SystemExit(return_code)

    state.update({"state": "verifying", "copied_bytes": local_bytes(args.destination), "updated_at": time.time()})
    atomic_json(args.state, state)
    remote = remote_manifest(args.host, args.source)
    local = local_manifest(args.destination)
    verified = remote == local
    manifest_bytes = json.dumps(remote, sort_keys=True, separators=(",", ":")).encode()
    state.update(
        {
            "state": "verified" if verified else "verification_failed",
            "finished_at": time.time(),
            "file_count": len(remote),
            "verified_bytes": sum(item["bytes"] for item in remote.values()),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_deleted": False,
        }
    )
    atomic_json(args.state, state)
    if not verified:
        send(args.chat_id, f"{args.label} ERROR: source/destination manifest mismatch; source retained")
        raise SystemExit(3)
    send(
        args.chat_id,
        f"{args.label} PROGRESS: 100% verified {len(remote)} files / {state['verified_bytes']} bytes; source retained pending reclaim",
    )


if __name__ == "__main__":
    main()
