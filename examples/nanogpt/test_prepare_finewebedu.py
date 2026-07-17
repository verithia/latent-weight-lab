import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("prep", Path(__file__).with_name("prepare_finewebedu.py"))
prep = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(prep)


class PrepareSafetyTests(unittest.TestCase):
    def test_complete_manifest_requires_exact_hash_and_counts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "train.bin").write_bytes(b"\0\0\1\0")
            (root / "val.bin").write_bytes(b"\2\0")
            files = {name: {"path": f"{name}.bin", "bytes": (root / f"{name}.bin").stat().st_size, "tokens": (root / f"{name}.bin").stat().st_size // 2, "sha256": prep.sha256(root / f"{name}.bin")} for name in ("train", "val")}
            manifest = {"complete": True, "target_tokens": 2, "val_tokens": 1, "files": files}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(prep.complete_manifest(root))
            (root / "train.bin").write_bytes(b"\0\0")
            self.assertFalse(prep.complete_manifest(root))

    def test_commit_shard_never_marks_part_done(self):
        with tempfile.TemporaryDirectory() as raw:
            part = Path(raw) / "x.bin.part"; part.write_bytes(b"\0\0")
            checkpoint = {"commit_sequence": 1}
            done = part.with_suffix(""); prep.commit_shard(part, done, 1, checkpoint)
            self.assertFalse(part.exists()); self.assertTrue(done.exists())
            self.assertEqual(json.loads(done.with_suffix(".bin.done").read_text())["tokens"], 1)

    def test_smoke_resume_after_interruption_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); script = str(Path(__file__).with_name("prepare_finewebedu.py"))
            common = [sys.executable, script, "--smoke", "--target-tokens", "70", "--val-tokens", "20", "--shard-tokens", "25"]
            subprocess.run(common + ["--output-dir", str(root / "full")], check=True, capture_output=True, text=True)
            stage = root / "resumed.staging"
            interrupted = subprocess.run(common + ["--output-dir", str(root / "resumed"), "--staging-dir", str(stage)], env={**os.environ, "PREP_TEST_STOP_AFTER_COMMITS": "2"}, capture_output=True, text=True)
            self.assertNotEqual(interrupted.returncode, 0)
            # This is deliberately not checkpointed and must be deleted/replayed.
            (stage / "shards" / "train-000001.bin.part").write_bytes(b"\xff\xff\xff\xff")
            subprocess.run(common + ["--output-dir", str(root / "resumed"), "--staging-dir", str(stage)], check=True, capture_output=True, text=True)
            for name in ("train.bin", "val.bin", "manifest.json"):
                self.assertEqual((root / "full" / name).read_bytes(), (root / "resumed" / name).read_bytes())


if __name__ == "__main__": unittest.main()
