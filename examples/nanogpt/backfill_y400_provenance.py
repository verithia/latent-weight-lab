"""Create explicit, verifiable post-hoc provenance for legacy Y400 runs.

This tool never claims that a command was captured at launch.  It only emits a
record after the checkpoint source hashes match a recovered Git checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "y400_experiment_provenance_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_value(source_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source_root), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def checkpoint_identity(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = config.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        raise ValueError("config has no out_dir for checkpoint-identity verification")
    metadata_path = Path(out_dir) / "ckpt.meta.json"
    if not metadata_path.is_file():
        raise ValueError(f"checkpoint metadata is absent: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    identity = metadata.get("run_identity")
    if not isinstance(identity, dict):
        raise ValueError("checkpoint metadata has no run_identity")
    source_hashes = identity.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("checkpoint metadata has no source hashes")
    return identity


def source_hashes_match(source_root: Path, expected: dict[str, Any]) -> None:
    for relative_path, expected_hash in expected.items():
        candidate = source_root / relative_path
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise ValueError(f"recovered source does not match checkpoint hash: {relative_path}")


def backfill_one(status_path: Path, source_root: Path, output_dir: Path) -> Path:
    status = json.loads(status_path.read_text())
    config_path = Path(status["config"])
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError(f"config is not an object: {config_path}")
    identity = checkpoint_identity(config)
    source_hashes = identity["source_hashes"]
    source_hashes_match(source_root, source_hashes)
    data_dir = config.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("config has no data_dir")
    manifest = (Path(data_dir) / "manifest.json").resolve()
    manifest_sha256 = sha256_file(manifest)
    if config.get("data_manifest_sha256") != manifest_sha256:
        raise ValueError("config data manifest hash differs from current manifest")

    stem = status_path.stem
    config_archive = output_dir / f"{stem}.config.json"
    provenance_path = output_dir / f"{stem}.historical.json"
    atomic_write(config_archive, raw_config)
    commit = git_value(source_root, "rev-parse", "HEAD")
    origin = git_value(source_root, "remote", "get-url", "origin")
    entrypoint = ["python3", "-u", "-m", "examples.nanogpt.train"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": stem,
        "run_name": status["run_name"],
        "historical_recovery": {
            "status": "post_launch_reconstructed",
            "source_capture": "recovery checkout matches checkpoint source hashes",
            "command_capture": "reconstructed from the archived detached-launcher contract; not captured argv",
            "status_sidecar": str(status_path.resolve()),
        },
        "repository": {
            "root": str(source_root.resolve()),
            "origin": origin,
            "git_commit": commit,
            "worktree_dirty": False,
        },
        "entrypoint": entrypoint,
        "command": [*entrypoint, "--config", status["config"]],
        "config": {
            "source_path": str(config_path.resolve()),
            "archive_path": str(config_archive.resolve()),
            "sha256": sha256_file(config_archive),
            "resolved_config_sha256": identity.get("config_sha256"),
        },
        "dataset_manifest": {"path": str(manifest), "sha256": manifest_sha256},
        "source_hashes": source_hashes,
    }
    atomic_write(
        provenance_path,
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n",
    )
    return provenance_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="append", required=True, help="legacy status JSON; repeat")
    parser.add_argument("--source-root", required=True, help="Git recovery checkout matching checkpoint hashes")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    if git_value(source_root, "status", "--porcelain"):
        raise ValueError("recovery source checkout must be clean")
    output_dir = Path(args.output_dir).resolve()
    for raw_status in args.status:
        output = backfill_one(Path(raw_status).resolve(), source_root, output_dir)
        print(output)


if __name__ == "__main__":
    main()
