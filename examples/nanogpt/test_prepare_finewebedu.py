import importlib.util
from array import array
import copy
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

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
            smoke_texts = ("safe prep ", "resumes without truncation ", "fineweb smoke ") * 10
            serial = [token for text in smoke_texts for token in (list(text.encode("utf-8")) + [50256])]
            self.assertEqual((root / "full" / "val.bin").read_bytes(), array("H", serial[:20]).tobytes())
            self.assertEqual((root / "full" / "train.bin").read_bytes(), array("H", serial[20:90]).tobytes())

    def test_batched_tokenization_matches_serial_with_order_and_bounds(self):
        class FakeEncoding:
            eot_token = 50256
            def encode_ordinary(self, text): return list(text.encode("utf-8"))
        texts = ["", "é", "short", "x" * 64, "终"]
        rows = iter({"text": text} for text in texts)
        batches = list(prep.bounded_document_batches(rows, max_docs=2, max_utf8_bytes=12))
        encoding = FakeEncoding()
        serial_backend = prep.TokenizerBackend("serial", encoding, "gpt2", 1).start()
        thread_backend = prep.TokenizerBackend("threadpool", encoding, "gpt2", 2).start()
        try:
            serial_batched = [tokens for batch in batches for tokens in serial_backend.tokenize(batch)]
            batched = [tokens for batch in batches for tokens in thread_backend.tokenize(batch)]
        finally:
            serial_backend.close()
            thread_backend.close()
        serial = [encoding.encode_ordinary(text) + [encoding.eot_token] for text in texts]
        self.assertEqual(batched, serial)
        self.assertEqual(serial_batched, serial)
        self.assertEqual(array("H", [x for tokens in batched for x in tokens]).tobytes(), array("H", [x for tokens in serial for x in tokens]).tobytes())
        self.assertIn(["x" * 64], batches)

    def test_processpool_lifecycle_uses_ordered_map_and_terminates_on_failure(self):
        events = []
        class FakePool:
            def map(self, fn, texts, chunksize):
                events.append(("map", list(texts), chunksize))
                return [array("H", list(text.encode("utf-8"))) for text in texts]
            def close(self): events.append(("close",))
            def terminate(self): events.append(("terminate",))
            def join(self): events.append(("join",))
        class FakeContext:
            def Pool(self, processes, initializer, initargs):
                events.append(("pool", processes, initargs)); return FakePool()
        class FakeEncoding:
            eot_token = 50256
            def encode_ordinary(self, text): return list(text.encode("utf-8"))
        with mock.patch.object(prep.multiprocessing, "get_context", return_value=FakeContext()):
            for _ in range(2):
                backend = prep.TokenizerBackend("processpool", FakeEncoding(), "gpt2", 8).start()
                self.assertEqual(backend.tokenize(["a", "é"]), [[97, 50256], [195, 169, 50256]])
                backend.close()
            failed = prep.TokenizerBackend("processpool", FakeEncoding(), "gpt2", 8).start()
            failed.close(failed=True)
        self.assertIn(("close",), events)
        self.assertIn(("terminate",), events)
        self.assertGreaterEqual(events.count(("join",)), 3)

    def test_default_benchmark_plan_excludes_unsafe_threadpool(self):
        self.assertEqual(prep.benchmark_backends("serial,processpool"), ["serial", "processpool"])
        self.assertEqual(prep.benchmark_backends("processpool"), ["serial", "processpool"])
        self.assertEqual(prep.benchmark_backends("serial,threadpool,processpool"), ["serial", "threadpool", "processpool"])

    def test_hard_exit_smoke_returns_cleanly_after_promotion(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "hard-exit-output"
            run = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("prepare_finewebedu.py")), "--smoke", "--hard-exit", "--output-dir", str(output), "--target-tokens", "30", "--val-tokens", "10", "--shard-tokens", "20"],
                capture_output=True, text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("promoted verified output", run.stdout)
            self.assertTrue((output / "manifest.json").is_file())

    def test_fast_continuation_migrates_immutable_prefix_and_restarts_fresh_segment(self):
        args = SimpleNamespace(dataset="HuggingFaceFW/fineweb-edu", name="sample-100BT", revision="commit-100bt", split="train", tokenizer="gpt2", seed=1337)
        legacy_shards = {"val": [{"path": "shards/val-000000.bin", "tokens": 20_000_000}], "train": [{"path": "shards/train-000000.bin", "tokens": 8_150_000_000}]}
        state = {"version": 2, "split_tokens": {"val": 20_000_000, "train": 8_150_000_000}, "shards": copy.deepcopy(legacy_shards), "documents_seen": 99, "document_offset": 7}
        first = prep.begin_fast_continuation(state, args)
        self.assertEqual(state["segments"][0]["name"], "sample-10BT")
        self.assertEqual((state["segments"][0]["train_start_token"], state["segments"][0]["train_end_token"]), (0, 8_150_000_000))
        self.assertEqual(state["shards"], legacy_shards)
        self.assertEqual((first["train_start_token"], first["train_end_token"], state["documents_seen"], state["document_offset"]), (8_150_000_000, 8_150_000_000, 0, 0))
        state["split_tokens"]["train"] += 25
        prep.update_fast_segment_end(state)
        second = prep.begin_fast_continuation(state, args)
        self.assertFalse(state["segments"][1]["active"])
        self.assertEqual(state["segments"][1]["closed_reason"], "restart_new_seed")
        self.assertNotEqual(first["seed"], second["seed"])
        self.assertEqual((state["segments"][1]["train_start_token"], state["segments"][1]["train_end_token"], second["train_start_token"]), (8_150_000_000, 8_150_000_025, 8_150_000_025))

    def test_mirror_url_helpers_preserve_datafiles_filesystem_type(self):
        calls = []
        url_calls = []
        prepared_paths = []
        pagination_calls = []
        class FakeFileSystem:
            def __init__(self, *args, **kwargs): calls.append(kwargs)
        datasets = types.ModuleType("datasets")
        datasets.config = types.SimpleNamespace(HF_ENDPOINT="https://old.example")
        data_files = types.ModuleType("datasets.data_files"); data_files.HfFileSystem = FakeFileSystem
        def url_to_fs(url, *args, **kwargs):
            url_calls.append((url, kwargs))
            return "fs", url
        data_files.url_to_fs = url_to_fs
        def prepare_path_and_storage_options(path, *args, **kwargs):
            prepared_paths.append(path)
            return path, kwargs
        data_files._prepare_path_and_storage_options = prepare_path_and_storage_options
        load_module = types.ModuleType("datasets.load"); load_module.HfFileSystem = FakeFileSystem
        datasets.data_files = data_files; datasets.load = load_module
        pagination = types.ModuleType("huggingface_hub.utils._pagination")
        def get_next_page(value):
            pagination_calls.append(value)
            return value
        pagination._get_next_page = get_next_page
        hub_utils = types.ModuleType("huggingface_hub.utils"); hub_utils._pagination = pagination
        hub = types.ModuleType("huggingface_hub"); hub.utils = hub_utils
        def load_dataset(*args, **kwargs):
            filesystem = data_files.HfFileSystem(token="data-files")
            assert isinstance(filesystem, data_files.HfFileSystem)
            data_files._prepare_path_and_storage_options("https://huggingface.co/datasets/repo/resolve/main/data")
            data_files.url_to_fs("hf://datasets/repo@main/data")
            return "streaming-metadata"
        datasets.load_dataset = load_dataset
        with mock.patch.dict(sys.modules, {"datasets": datasets, "datasets.data_files": data_files, "datasets.load": load_module, "huggingface_hub": hub, "huggingface_hub.utils": hub_utils, "huggingface_hub.utils._pagination": pagination}):
            with mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}, clear=False):
                result = prep.load_streaming_dataset(types.SimpleNamespace(dataset="repo", name="cfg", revision="main", split="train", streaming=True))
                self.assertEqual(result, "streaming-metadata")
                self.assertIsInstance(data_files.HfFileSystem, type)
                self.assertIs(data_files.HfFileSystem, FakeFileSystem)
                self.assertIs(load_module.HfFileSystem, FakeFileSystem)
                data_files.url_to_fs("hf://datasets/HuggingFaceFW/fineweb-edu@main/data")
                data_files.url_to_fs("hf://explicit", endpoint="https://explicit.example")
                data_files.url_to_fs("/local/path")
                data_files._prepare_path_and_storage_options("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/data")
                data_files._prepare_path_and_storage_options("https://other.example/data")
                self.assertEqual(pagination._get_next_page("https://huggingface.co/api/datasets?page=2"), "https://hf-mirror.com/api/datasets?page=2")
                self.assertIsNone(pagination._get_next_page(None))
                self.assertEqual(pagination._get_next_page("https://other.example/page=2"), "https://other.example/page=2")
        self.assertEqual(datasets.config.HF_ENDPOINT, "https://hf-mirror.com")
        self.assertEqual(calls, [{"token": "data-files"}])
        self.assertEqual(url_calls, [("hf://datasets/repo@main/data", {"endpoint": "https://hf-mirror.com"}), ("hf://datasets/HuggingFaceFW/fineweb-edu@main/data", {"endpoint": "https://hf-mirror.com"}), ("hf://explicit", {"endpoint": "https://explicit.example"}), ("/local/path", {})])
        self.assertEqual(prepared_paths, ["https://hf-mirror.com/datasets/repo/resolve/main/data", "https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/data", "https://other.example/data"])
        self.assertEqual(pagination_calls, ["https://huggingface.co/api/datasets?page=2", None, "https://other.example/page=2"])


if __name__ == "__main__": unittest.main()
