#!/usr/bin/env python3
"""Reclaim remote files only after two hosts match a pinned SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict


REMOTE_INSPECT = r'''import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
entries = json.loads(sys.argv[2])
result = {}
for entry in entries:
    rel = entry["path"]
    path = root / rel
    if not path.is_file():
        result[rel] = {"exists": False}
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result[rel] = {
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
print(json.dumps(result, sort_keys=True))'''

REMOTE_DELETE = r'''import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
entries = json.loads(sys.argv[2])
deleted = {}
for entry in entries:
    rel = entry["path"]
    path = root / rel
    if not path.is_file():
        raise SystemExit(f"primary file disappeared before reclaim: {rel}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    expected = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
    if actual != expected:
        raise SystemExit(f"primary file changed before reclaim: {rel}")
    path.unlink()
    deleted[rel] = actual
print(json.dumps(deleted, sort_keys=True))'''


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_manifest(payload: Dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest entries must be a non-empty list")
    paths: set[str] = set()
    total = 0
    for entry in entries:
        path = entry.get("path")
        pure = PurePosixPath(path) if isinstance(path, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) in {"", "."}
        ):
            raise ValueError(f"unsafe relative path: {path!r}")
        if path in paths:
            raise ValueError(f"duplicate path: {path}")
        paths.add(path)
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"invalid byte size for {path}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid SHA-256 for {path}")
        total += size
    if total != payload.get("expected_total_bytes"):
        raise ValueError("expected_total_bytes does not match entries")
    for role in ("primary", "authority"):
        host = payload.get(role, {}).get("host")
        root = payload.get(role, {}).get("root")
        if not isinstance(host, str) or not host:
            raise ValueError(f"missing {role} host")
        if not isinstance(root, str) or not root.startswith("/"):
            raise ValueError(f"missing absolute {role} root")


def manifest_sha256(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def remote_python(host: str, script: str, root: str, entries: list[dict]) -> dict:
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
            root,
            json.dumps(entries, separators=(",", ":")),
        ],
        input=script,
        text=True,
        capture_output=True,
        timeout=3600,
        check=True,
    )
    return json.loads(completed.stdout)


def expected_observation(entries: list[dict]) -> dict:
    return {
        entry["path"]: {
            "exists": True,
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in entries
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    validate_manifest(payload)
    entries = payload["entries"]
    primary = payload["primary"]
    authority = payload["authority"]
    expected = expected_observation(entries)
    started = time.time()

    primary_before = remote_python(
        primary["host"], REMOTE_INSPECT, primary["root"], entries
    )
    authority_before = remote_python(
        authority["host"], REMOTE_INSPECT, authority["root"], entries
    )
    verified = primary_before == expected and authority_before == expected
    state: Dict[str, Any] = {
        "state": "verified_dry_run" if verified else "verification_failed",
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256(payload),
        "started_at": started,
        "finished_at": time.time(),
        "expected_total_bytes": payload["expected_total_bytes"],
        "primary_before": primary_before,
        "authority_before": authority_before,
        "executed": False,
    }
    atomic_json(args.state, state)
    if not verified:
        raise SystemExit("both hosts did not match the pinned manifest")
    if not args.execute:
        print(json.dumps(state, sort_keys=True))
        return

    deleted = remote_python(
        primary["host"], REMOTE_DELETE, primary["root"], entries
    )
    authority_after = remote_python(
        authority["host"], REMOTE_INSPECT, authority["root"], entries
    )
    primary_after = remote_python(
        primary["host"], REMOTE_INSPECT, primary["root"], entries
    )
    primary_absent = all(
        observation == {"exists": False} for observation in primary_after.values()
    )
    completed = (
        deleted
        == {
            entry["path"]: {
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            }
            for entry in entries
        }
        and authority_after == expected
        and primary_absent
    )
    state.update(
        {
            "state": "reclaimed" if completed else "post_reclaim_verification_failed",
            "finished_at": time.time(),
            "executed": True,
            "reclaimed_bytes": sum(item["bytes"] for item in deleted.values()),
            "deleted_primary": deleted,
            "primary_after": primary_after,
            "authority_after": authority_after,
        }
    )
    atomic_json(args.state, state)
    if not completed:
        raise SystemExit("post-reclaim verification failed")
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
