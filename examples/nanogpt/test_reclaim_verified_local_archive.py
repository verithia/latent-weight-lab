import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from examples.nanogpt.reclaim_verified_local_archive import (
    canonical_manifest_sha256,
    local_manifest,
    validate_archive_state,
    validate_direct_names,
)


class ReclaimVerifiedLocalArchiveTest(unittest.TestCase):
    def test_names_must_be_unique_direct_children(self):
        self.assertEqual(validate_direct_names(["a", "b"]), ["a", "b"])
        for names in ([], ["a", "a"], ["../a"], ["a/b"], ["/a"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_direct_names(names)

    def test_verified_state_binds_local_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            archive_root = Path(raw)
            destination = archive_root / "selection"
            selected = destination / "run"
            selected.mkdir(parents=True)
            payload = b"checkpoint"
            (selected / "ckpt.pt").write_bytes(payload)
            manifest = local_manifest(destination)
            state = {
                "state": "verified",
                "source_deleted": False,
                "host": "Y400",
                "source_root": "/remote/root",
                "include_names": ["run"],
                "destination": str(destination),
                "file_count": 1,
                "verified_bytes": len(payload),
                "manifest_sha256": canonical_manifest_sha256(manifest),
                "manifest": manifest,
            }
            names, pinned, actual_destination = validate_archive_state(
                state, archive_root
            )
            self.assertEqual(names, ["run"])
            self.assertEqual(pinned, manifest)
            self.assertEqual(actual_destination, destination.resolve())

            state["manifest_sha256"] = hashlib.sha256(b"wrong").hexdigest()
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                validate_archive_state(state, archive_root)

    def test_destination_cannot_escape_archive_root(self):
        with tempfile.TemporaryDirectory() as archive_raw, tempfile.TemporaryDirectory() as other_raw:
            destination = Path(other_raw) / "selection"
            selected = destination / "run"
            selected.mkdir(parents=True)
            (selected / "ckpt.pt").write_bytes(b"x")
            manifest = local_manifest(destination)
            state = {
                "state": "verified",
                "source_deleted": False,
                "host": "Y400",
                "source_root": "/remote/root",
                "include_names": ["run"],
                "destination": str(destination),
                "file_count": 1,
                "verified_bytes": 1,
                "manifest_sha256": canonical_manifest_sha256(manifest),
                "manifest": manifest,
            }
            with self.assertRaisesRegex(ValueError, "escapes"):
                validate_archive_state(state, Path(archive_raw))


if __name__ == "__main__":
    unittest.main()
