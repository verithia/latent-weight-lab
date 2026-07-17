"""Archive published exact-resume checkpoints without changing training state.

The trainer atomically replaces ``ckpt.pt`` and then publishes ``ckpt.meta.json``.
This sidecar waits for selected ``next_iter`` values and creates a hard link to
the already-published checkpoint.  Later atomic replacements leave each linked
inode immutable, so snapshots do not race a writer or alter resume behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def archive_checkpoint(out_dir: Path, archive_dir: Path, expected_iter: int) -> bool:
    metadata_path = out_dir / "ckpt.meta.json"
    checkpoint_path = out_dir / "ckpt.pt"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        return False
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    actual_iter = int(metadata["next_iter"])
    if actual_iter != expected_iter:
        return False
    archive_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_archive = archive_dir / f"ckpt_iter{actual_iter:06d}.pt"
    metadata_archive = archive_dir / f"ckpt_iter{actual_iter:06d}.meta.json"
    if checkpoint_archive.exists() and metadata_archive.exists():
        return True
    if not checkpoint_archive.exists():
        os.link(checkpoint_path, checkpoint_archive)
    payload = json.loads(metadata_bytes)
    payload["analysis_snapshot"] = {
        "source_checkpoint": str(checkpoint_path),
        "snapshot_checkpoint": str(checkpoint_archive),
        "published_next_iter": actual_iter,
    }
    temporary = metadata_archive.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, metadata_archive)
    print(f"archived next_iter={actual_iter} checkpoint={checkpoint_archive}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", required=True, help="comma-separated exact next_iter snapshot values")
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be > 0")
    out_dir = Path(args.out_dir)
    archive_dir = Path(args.archive_dir) if args.archive_dir else out_dir / "analysis_snapshots"
    pending = {int(value) for value in args.steps.split(",") if value}
    if not pending:
        raise ValueError("--steps must contain at least one iteration")
    while pending:
        completed = {step for step in pending if archive_checkpoint(out_dir, archive_dir, step)}
        pending.difference_update(completed)
        if pending:
            time.sleep(args.poll_seconds)
    print("all requested snapshots archived", flush=True)


if __name__ == "__main__":
    main()
