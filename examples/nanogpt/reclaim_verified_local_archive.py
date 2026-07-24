#!/usr/bin/env python3
"""Reclaim remote directories only after independently verifying a local archive."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any


REMOTE_RECLAIM = r'''import base64, hashlib, json, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
names = json.loads(base64.urlsafe_b64decode(sys.argv[2]))
expected = json.loads(base64.urlsafe_b64decode(sys.argv[3]))
if not root.is_absolute() or root.is_symlink() or not root.is_dir():
    raise SystemExit("unsafe remote source root")
actual = {}
for name in names:
    selected = root / name
    if selected.is_symlink() or not selected.is_dir() or selected.parent != root:
        raise SystemExit(f"unsafe selected source: {selected}")
    for path in sorted(selected.rglob("*")):
        if path.is_symlink():
            raise SystemExit(f"remote archive source contains symlink: {path}")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
if actual != expected:
    raise SystemExit("remote source changed after archive verification")
for name in names:
    shutil.rmtree(root / name)
remaining = {name: (root / name).exists() for name in names}
if any(remaining.values()):
    raise SystemExit(f"remote source remained after reclaim: {remaining}")
print(json.dumps({"deleted": names, "remaining": remaining}, sort_keys=True))'''


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_direct_names(names: Any) -> list[str]:
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("include_names must be a non-empty unique list")
    for name in names:
        pure = PurePosixPath(name) if isinstance(name, str) else None
        if pure is None or pure.is_absolute() or len(pure.parts) != 1 or name in {".", ".."}:
            raise ValueError(f"unsafe include name: {name!r}")
    return names


def validate_archive_state(
    state: dict[str, Any], archive_root: Path
) -> tuple[list[str], dict[str, Any], Path]:
    if state.get("state") != "verified" or state.get("source_deleted") is not False:
        raise ValueError("archive state must be verified with source_deleted=false")
    host = state.get("host")
    source_root = state.get("source_root")
    if not isinstance(host, str) or not host or host in {"localhost", "127.0.0.1"}:
        raise ValueError("archive state must name a remote host")
    if not isinstance(source_root, str) or not source_root.startswith("/"):
        raise ValueError("archive source_root must be absolute")
    names = validate_direct_names(state.get("include_names"))
    manifest = state.get("manifest")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("archive state has no pinned manifest")
    allowed_prefixes = tuple(name + "/" for name in names)
    for relative, record in manifest.items():
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or not relative.startswith(allowed_prefixes)
        ):
            raise ValueError(f"unsafe manifest path: {relative!r}")
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise ValueError(f"invalid manifest record: {relative!r}")
    if set(path.split("/", 1)[0] for path in manifest) != set(names):
        raise ValueError("manifest does not cover every selected directory")
    if state.get("file_count") != len(manifest):
        raise ValueError("archive file_count disagrees with manifest")
    if state.get("verified_bytes") != sum(item["bytes"] for item in manifest.values()):
        raise ValueError("archive verified_bytes disagrees with manifest")
    if state.get("manifest_sha256") != canonical_manifest_sha256(manifest):
        raise ValueError("archive manifest SHA-256 disagrees with manifest")
    destination = Path(state.get("destination", "")).resolve()
    trusted_root = archive_root.resolve()
    try:
        destination.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError("archive destination escapes trusted archive root") from exc
    if destination == trusted_root or not destination.is_dir() or destination.is_symlink():
        raise ValueError("archive destination must be a real child directory")
    return names, manifest, destination


def local_manifest(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"local archive contains symlink: {path}")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return result


def encode(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def reclaim_remote(
    host: str, source_root: str, names: list[str], manifest: dict[str, Any]
) -> dict[str, Any]:
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
            encode(names),
            encode(manifest),
        ],
        input=REMOTE_RECLAIM,
        text=True,
        capture_output=True,
        timeout=3600,
        check=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-state", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/Users/verithia/research/archives"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    archive_state = json.loads(args.archive_state.read_text())
    names, manifest, destination = validate_archive_state(
        archive_state, args.archive_root
    )
    started = time.time()
    local_before = local_manifest(destination)
    verified = local_before == manifest
    state: dict[str, Any] = {
        "state": "verified_dry_run" if verified else "verification_failed",
        "archive_state": str(args.archive_state.resolve()),
        "archive_state_sha256": hashlib.sha256(args.archive_state.read_bytes()).hexdigest(),
        "manifest_sha256": canonical_manifest_sha256(manifest),
        "host": archive_state["host"],
        "source_root": archive_state["source_root"],
        "destination": str(destination),
        "include_names": names,
        "expected_bytes": sum(item["bytes"] for item in manifest.values()),
        "started_at": started,
        "finished_at": time.time(),
        "executed": False,
    }
    atomic_json(args.state, state)
    if not verified:
        raise SystemExit("local archive no longer matches the pinned manifest")
    if not args.execute:
        print(json.dumps(state, sort_keys=True))
        return

    remote_result = reclaim_remote(
        archive_state["host"],
        archive_state["source_root"],
        names,
        manifest,
    )
    local_after = local_manifest(destination)
    completed = (
        local_after == manifest
        and remote_result.get("deleted") == names
        and not any(remote_result.get("remaining", {}).values())
    )
    state.update(
        {
            "state": "reclaimed" if completed else "post_reclaim_verification_failed",
            "finished_at": time.time(),
            "executed": True,
            "reclaimed_bytes": state["expected_bytes"],
            "remote_result": remote_result,
            "local_authority_preserved": local_after == manifest,
        }
    )
    atomic_json(args.state, state)
    if not completed:
        raise SystemExit("post-reclaim verification failed")
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
