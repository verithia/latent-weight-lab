#!/usr/bin/env python3
"""Crash-safe, resumable FineWeb-Edu -> GPT-2 uint16 preparation.

Completed shards are immutable.  A run writes only to a unique staging directory
and promotes it with ``rename(2)`` after validating every byte and checksum.
Resume an interrupted run by passing the printed ``--staging-dir`` explicitly.
"""
from __future__ import annotations

import argparse
from array import array
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
from pathlib import Path

DTYPE_BYTES = 2
DEFAULT_BATCH_DOCS = int(os.environ.get("FINEWEB_TOKENIZER_BATCH_DOCS", "128"))
DEFAULT_BATCH_BYTES = int(os.environ.get("FINEWEB_TOKENIZER_BATCH_BYTES", str(2 * 1024 * 1024)))
DEFAULT_TOKENIZER_THREADS = int(os.environ.get("FINEWEB_TOKENIZER_THREADS", "8"))
DEFAULT_TOKENIZER_BACKEND = os.environ.get("TOKENIZER_BACKEND", "processpool")
_PROCESS_ENCODING = None


def enforce_hf_pagination_endpoint(endpoint: str) -> None:
    """Keep Hub pagination links on the configured endpoint, process-locally."""
    try:
        from huggingface_hub.utils import _pagination
    except ImportError:
        return
    original = _pagination._get_next_page
    if getattr(original, "_finewebedu_endpoint", None) == endpoint:
        return
    default_hub = "https://huggingface.co"
    mirror = endpoint.rstrip("/")

    def endpoint_next_page(*args, **kwargs):
        next_url = original(*args, **kwargs)
        if isinstance(next_url, str) and (next_url == default_hub or next_url.startswith(default_hub + "/")):
            return mirror + next_url[len(default_hub):]
        return next_url

    endpoint_next_page._finewebedu_endpoint = endpoint
    _pagination._get_next_page = endpoint_next_page


def enforce_datasets_datafiles_endpoint() -> str | None:
    """Pin datasets 3.2 DataFiles listings to the configured Hub endpoint.

    datasets 3.2 resolves DataFiles through module-local URL helpers. Patch
    only those helpers before importing/calling public builder/load callables;
    retain ``HfFileSystem`` as a class because datasets uses it in isinstance.
    """
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if not endpoint:
        return None
    import datasets
    data_files = importlib.import_module("datasets.data_files")
    importlib.import_module("datasets.load")

    datasets.config.HF_ENDPOINT = endpoint
    original_url_to_fs = getattr(data_files, "url_to_fs", None)
    if original_url_to_fs is not None and getattr(original_url_to_fs, "_finewebedu_endpoint", None) != endpoint:
        def endpoint_url_to_fs(url, *args, **kwargs):
            if isinstance(url, str) and url.startswith("hf://"):
                kwargs.setdefault("endpoint", endpoint)
            return original_url_to_fs(url, *args, **kwargs)

        endpoint_url_to_fs._finewebedu_endpoint = endpoint
        data_files.url_to_fs = endpoint_url_to_fs
    original_prepare = getattr(data_files, "_prepare_path_and_storage_options", None)
    if original_prepare is not None and getattr(original_prepare, "_finewebedu_endpoint", None) != endpoint:
        default_hub = "https://huggingface.co"
        mirror = endpoint.rstrip("/")

        def endpoint_prepare_path_and_storage_options(path, *args, **kwargs):
            # datasets 3.2 only turns configured-endpoint HTTP URLs into hf://;
            # rewrite the default Hub prefix before its own preprocessing.
            if isinstance(path, str) and path.startswith(default_hub):
                path = mirror + path[len(default_hub):]
            return original_prepare(path, *args, **kwargs)

        endpoint_prepare_path_and_storage_options._finewebedu_endpoint = endpoint
        data_files._prepare_path_and_storage_options = endpoint_prepare_path_and_storage_options
    enforce_hf_pagination_endpoint(endpoint)
    return endpoint


def load_streaming_dataset(args):
    enforce_datasets_datafiles_endpoint()
    from datasets import load_dataset

    return load_dataset(args.dataset, name=args.name, split=args.split, revision=args.revision, streaming=args.streaming)


def probe_dataset(args) -> None:
    """Resolve builder and streaming DataFiles metadata without writing tokens."""
    endpoint = enforce_datasets_datafiles_endpoint()
    from datasets import load_dataset_builder

    builder = load_dataset_builder(args.dataset, name=args.name, revision=args.revision)
    load_streaming_dataset(args)
    if args.probe_commit:
        from huggingface_hub import HfFileSystem
        filesystem = HfFileSystem(endpoint=endpoint) if endpoint else HfFileSystem()
        # A pinned data-directory glob exercises the paginated tree listing but
        # writes no token output. Supply the commit from the target dataset run.
        matches = filesystem.glob(f"hf://datasets/{args.dataset}@{args.probe_commit}/data/*")
        print(f"probe glob ok: commit={args.probe_commit} entries={len(matches)}")
    print(f"probe ok: dataset={args.dataset} name={args.name} endpoint={endpoint or 'default'} builder={type(builder).__name__}")


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def complete_manifest(directory: Path) -> bool:
    manifest = directory / "manifest.json"
    if not manifest.is_file(): return False
    try:
        data = load_json(manifest)
        return data.get("complete") is True and validate(directory, data, checksums=True)
    except (OSError, ValueError, KeyError):
        return False


def validate(directory: Path, manifest: dict, checksums: bool = True) -> bool:
    """Validate exact expected files. Checksums are generated only at final pass."""
    for split in ("train", "val"):
        item = manifest["files"][split]
        path = directory / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]: return False
        if item["bytes"] != item["tokens"] * DTYPE_BYTES: return False
        if checksums and sha256(path) != item["sha256"]: return False
    return manifest["files"]["train"]["tokens"] == manifest["target_tokens"] and manifest["files"]["val"]["tokens"] == manifest["val_tokens"]


def status(stage: Path, phase: str, state: dict) -> None:
    atomic_json(stage / "prep-status.json", {"phase": phase, "committed_tokens": state["committed_tokens"], "target_tokens": state["target_tokens"] + state["val_tokens"]})


def commit_shard(part: Path, done: Path, tokens: int, checkpoint: dict) -> None:
    """Make one shard durable; state is advanced only after this returns.

    ``.commit`` bridges the two-file rename window.  Recovery can finish that
    rename from a fully fsynced part without ever accepting arbitrary .part data.
    """
    if tokens <= 0: return
    with part.open("ab") as f:
        f.flush(); os.fsync(f.fileno())
    commit = done.with_suffix(done.suffix + ".commit")
    atomic_json(commit, {"tokens": tokens, "bytes": tokens * DTYPE_BYTES, "checkpoint": checkpoint})
    os.replace(part, done)
    os.replace(commit, done.with_suffix(done.suffix + ".done"))


def recover_checkpoint(stage: Path, initial: dict) -> dict:
    """Recover the newest fully-described shard commit, then discard all parts."""
    candidates = list((stage / "shards").glob("*.bin.done")) + list((stage / "shards").glob("*.bin.commit"))
    newest = initial
    for marker in candidates:
        record = load_json(marker)
        checkpoint = record.get("checkpoint")
        if not checkpoint or checkpoint.get("commit_sequence", -1) < newest.get("commit_sequence", -1): continue
        data = marker.with_suffix("")
        if marker.suffix == ".commit":
            part = data.with_suffix(data.suffix + ".part")
            if not data.exists() and part.exists() and part.stat().st_size == record["bytes"]:
                os.replace(part, data)
            if data.exists() and data.stat().st_size == record["bytes"]:
                os.replace(marker, data.with_suffix(data.suffix + ".done"))
        if data.is_file() and data.stat().st_size == record["bytes"]:
            newest = checkpoint
    for part in (stage / "shards").glob("*.part"):
        part.unlink()  # Not checkpointed: replay from newest durable cursor.
    return newest


def migrate_legacy_fast_state(state: dict, args) -> None:
    """Describe immutable legacy shards as the first segment without rewriting them."""
    if "segments" in state:
        return
    train_end = int(state["split_tokens"]["train"])
    state.update({
        "version": 3,
        "fast_continuation": True,
        "segments": [{
            "segment_id": 0, "active": False, "policy": "legacy_immutable_prefix",
            "dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "name": "sample-10BT", "revision": "main", "split": "train",
            "hf_endpoint": os.environ.get("HF_ENDPOINT", ""), "shuffle_buffer": 10_000,
            "tokenizer": args.tokenizer, "tokenizer_version": "tiktoken", "seed": args.seed,
            "train_start_token": 0, "train_end_token": train_end,
        }],
        "next_segment_id": 1,
        "active_segment_id": None,
    })


def begin_fast_continuation(state: dict, args) -> dict:
    """Close any crashed active segment and atomically describe a fresh no-replay one."""
    migrate_legacy_fast_state(state, args)
    train_boundary = int(state["split_tokens"]["train"])
    for segment in state["segments"]:
        if segment.get("active"):
            segment["active"] = False
            segment["train_end_token"] = train_boundary
            segment["closed_reason"] = "restart_new_seed"
    segment_id = int(state["next_segment_id"])
    segment = {
        "segment_id": segment_id, "active": True, "policy": "fresh_no_replay_after_durable_boundary",
        "dataset": args.dataset, "config": args.name, "name": args.name, "revision": args.revision, "split": args.split,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""), "shuffle_buffer": 10_000,
        "tokenizer": args.tokenizer, "tokenizer_version": "tiktoken",
        "seed": int(args.seed) + segment_id * 1_000_003,
        "train_start_token": train_boundary, "train_end_token": train_boundary,
    }
    state["segments"].append(segment)
    state["next_segment_id"] = segment_id + 1
    state["active_segment_id"] = segment_id
    # Deliberately do not replay prior source documents in this opt-in mode.
    state["documents_seen"] = 0
    state["document_offset"] = 0
    return segment


def update_fast_segment_end(state: dict) -> None:
    active_id = state.get("active_segment_id")
    if active_id is None:
        return
    for segment in state["segments"]:
        if segment["segment_id"] == active_id:
            segment["train_end_token"] = int(state["split_tokens"]["train"])
            return


def close_fast_segment(state: dict, reason: str) -> None:
    update_fast_segment_end(state)
    for segment in state.get("segments", []):
        if segment.get("segment_id") == state.get("active_segment_id"):
            segment["active"] = False
            segment["closed_reason"] = reason
    state["active_segment_id"] = None


def bounded_document_batches(rows, max_docs: int, max_utf8_bytes: int):
    """Yield ordered, bounded text windows; one oversized document stands alone."""
    batch: list[str] = []
    batch_bytes = 0
    for row in rows:
        text = row.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text_bytes = len(text.encode("utf-8"))
        if batch and (len(batch) >= max_docs or batch_bytes + text_bytes > max_utf8_bytes):
            yield batch
            batch, batch_bytes = [], 0
        batch.append(text)
        batch_bytes += text_bytes
        if len(batch) >= max_docs or text_bytes > max_utf8_bytes:
            yield batch
            batch, batch_bytes = [], 0
    if batch:
        yield batch


def _process_tokenizer_init(tokenizer_name: str) -> None:
    """Spawn-worker initializer: only tiktoken is loaded in tokenizer workers."""
    global _PROCESS_ENCODING
    import tiktoken
    _PROCESS_ENCODING = tiktoken.get_encoding(tokenizer_name)


def _process_encode_ordinary(text: str) -> array:
    return array("H", _PROCESS_ENCODING.encode_ordinary(text))


class TokenizerBackend:
    """Persistent ordered tokenizer backend with explicit worker lifecycle."""
    def __init__(self, backend: str, enc, tokenizer_name: str, threads: int, *, allow_process: bool = True):
        self.backend = backend
        self.enc = enc
        self.tokenizer_name = tokenizer_name
        self.threads = threads
        self.allow_process = allow_process
        self.executor = None
        self.pool = None
        self.closed_cleanly = False
        self._previous_handlers = {}

    def start(self):
        backend = self.backend
        if backend == "processpool" and not self.allow_process:
            backend = "threadpool"
        self.effective_backend = backend
        if backend == "threadpool":
            self.executor = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="fineweb-tokenize")
        elif backend == "processpool":
            context = multiprocessing.get_context("spawn")
            self.pool = context.Pool(self.threads, initializer=_process_tokenizer_init, initargs=(self.tokenizer_name,))
        self._install_signal_cleanup()
        return self

    def _install_signal_cleanup(self) -> None:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"tokenizer backend interrupted by signal {signum}")
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupted)
            except (ValueError, OSError):
                pass

    def encode(self, texts: list[str]) -> list[list[int] | array]:
        if self.effective_backend == "serial":
            return [self.enc.encode_ordinary(text) for text in texts]
        if self.effective_backend == "threadpool":
            return list(self.executor.map(self.enc.encode_ordinary, texts))
        chunksize = max(1, len(texts) // max(1, self.threads * 4))
        return self.pool.map(_process_encode_ordinary, texts, chunksize=chunksize)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return [list(tokens) + [self.enc.eot_token] for tokens in self.encode(texts)]

    def close(self, failed: bool = False) -> None:
        if self.closed_cleanly:
            return
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=failed)
            if self.pool is not None:
                if failed:
                    self.pool.terminate()
                else:
                    self.pool.close()
                self.pool.join()
            self.closed_cleanly = not failed
        finally:
            for signum, previous in self._previous_handlers.items():
                signal.signal(signum, previous)
            self._previous_handlers.clear()


def benchmark_tokenization(args) -> None:
    """Measure one fixed ordered replay window without creating staging/output files."""
    if min(args.tokenizer_threads, args.tokenizer_batch_docs, args.tokenizer_batch_bytes) < 1:
        raise SystemExit("tokenizer settings must be positive")
    state = {"documents_seen": 0, "document_offset": 0}
    if args.staging_dir and not args.fast_benchmark:
        state_path = Path(args.staging_dir) / "state.json"
        if state_path.is_file():
            state = load_json(state_path)
    enc, ds = encoder(args, seed=args.seed)
    rows = iter(ds)
    for _ in range(state["documents_seen"]):
        next(rows)
    texts = next(bounded_document_batches(rows, args.tokenizer_batch_docs, args.tokenizer_batch_bytes))
    print(f"benchmark window: docs={len(texts)} utf8_bytes={sum(len(text.encode('utf-8')) for text in texts)} max_docs={args.tokenizer_batch_docs} max_bytes={args.tokenizer_batch_bytes} in_flight_batches=1")
    backends = benchmark_backends(args.benchmark_backends)
    baseline_tokens_per_second = None
    for backend in backends:
        threads = 1 if backend == "serial" else 8
        startup_start = time.perf_counter()
        tokenizer_backend = TokenizerBackend(backend, enc, args.tokenizer, threads, allow_process=not args.smoke).start()
        failed = True
        try:
            # First call includes process spawn/worker initialization and is
            # intentionally excluded from steady-state throughput.
            warmup = tokenizer_backend.tokenize(texts)
            startup_seconds = time.perf_counter() - startup_start
            tokens_per_batch = sum(len(item) for item in warmup) - state["document_offset"]
            start = time.perf_counter()
            for _ in range(args.benchmark_batches):
                tokenizer_backend.tokenize(texts)
            elapsed = max(time.perf_counter() - start, 1e-12)
            tokens = tokens_per_batch * args.benchmark_batches
            tokens_per_second = tokens / elapsed
            if backend == "serial":
                baseline_tokens_per_second = tokens_per_second
            speedup = tokens_per_second / max(baseline_tokens_per_second or tokens_per_second, 1e-12)
            failed = False
        finally:
            tokenizer_backend.close(failed=failed)
        print(f"benchmark backend={backend} threads={threads} startup_s={startup_seconds:.2f} steady_batches={args.benchmark_batches} docs_per_s={len(texts) * args.benchmark_batches / elapsed:.2f} steady_tokens_per_s={tokens_per_second:.2f} speedup_vs_serial={speedup:.2f} lifecycle_clean={tokenizer_backend.closed_cleanly} docs={len(texts)} tokens={tokens}")


def benchmark_backends(value: str) -> list[str]:
    backends = [item.strip() for item in value.split(",") if item.strip()]
    if not backends or any(item not in {"serial", "threadpool", "processpool"} for item in backends):
        raise SystemExit("--benchmark-backends must use serial,threadpool,processpool")
    return ["serial"] + [item for item in backends if item != "serial"]


def encoder(args, seed: int | None = None):
    if args.smoke:
        class SmokeEncoder:
            eot_token = 50256
            def encode_ordinary(self, text): return list(text.encode("utf-8"))
        return SmokeEncoder(), iter({"text": x} for x in ("safe prep ", "resumes without truncation ", "fineweb smoke ") * 10000)
    import tiktoken
    enc = tiktoken.get_encoding(args.tokenizer)
    ds = load_streaming_dataset(args)
    shuffle_seed = args.seed if seed is None else seed
    return enc, (ds.shuffle(seed=shuffle_seed, buffer_size=10_000) if args.streaming else ds.shuffle(seed=shuffle_seed))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir"); p.add_argument("--target-tokens", type=int)
    p.add_argument("--val-tokens", type=int, default=20_000_000); p.add_argument("--shard-tokens", type=int, default=50_000_000)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu"); p.add_argument("--name", default="sample-10BT")
    p.add_argument("--split", default="train"); p.add_argument("--revision", default="main"); p.add_argument("--tokenizer", default="gpt2"); p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=1337); p.add_argument("--staging-dir"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--probe", action="store_true", help="verify HF builder/streaming metadata only; writes no token output")
    p.add_argument("--probe-commit", help="optional pinned FineWeb commit for a paginated hf:// data glob during --probe")
    p.add_argument("--tokenizer-threads", type=int, default=DEFAULT_TOKENIZER_THREADS, help="persistent thread/process worker count (default: 8)")
    p.add_argument("--tokenizer-backend", choices=["serial", "threadpool", "processpool"], default=DEFAULT_TOKENIZER_BACKEND, help="persistent ordered tokenizer backend (default: processpool)")
    p.add_argument("--tokenizer-batch-docs", type=int, default=DEFAULT_BATCH_DOCS, help="max ordered documents per tokenization batch (default: 128)")
    p.add_argument("--tokenizer-batch-bytes", type=int, default=DEFAULT_BATCH_BYTES, help="max UTF-8 bytes per tokenization batch (default: 2MiB)")
    p.add_argument("--benchmark", action="store_true", help="tokenize one bounded replay window only; writes no token output")
    p.add_argument("--fast-benchmark", action="store_true", help="benchmark a fresh sample-100BT window without replay or writes")
    p.add_argument("--benchmark-backends", default="serial,processpool", help="comma-separated benchmark backends; threadpool is experimental/unsafe")
    p.add_argument("--benchmark-batches", type=int, default=8, help="steady-state bounded batches per backend after one warmup")
    p.add_argument("--fast-continuation", action="store_true", help="append fresh source segments after durable shards; never replay source cursor")
    p.add_argument("--hard-exit", action=argparse.BooleanOptionalAction, default=os.environ.get("PREP_HARD_EXIT", "0") == "1", help="use os._exit(0) after successful durable completion")
    args = p.parse_args()
    if args.probe:
        probe_dataset(args)
        return args.hard_exit
    if args.benchmark or args.fast_benchmark:
        if args.benchmark_batches < 1:
            p.error("--benchmark-batches must be positive")
        if args.fast_benchmark and args.name != "sample-100BT":
            p.error("--fast-benchmark requires --name sample-100BT")
        benchmark_tokenization(args)
        return args.hard_exit
    if args.output_dir is None or args.target_tokens is None:
        p.error("--output-dir and --target-tokens are required unless --probe is used")
    if min(args.target_tokens, args.val_tokens, args.shard_tokens, args.tokenizer_threads, args.tokenizer_batch_docs, args.tokenizer_batch_bytes) < 1: p.error("token counts and tokenizer settings must be positive")
    if args.fast_continuation and args.name != "sample-100BT":
        p.error("--fast-continuation requires --name sample-100BT")
    final = Path(args.output_dir).resolve()
    if final.exists() and complete_manifest(final): raise SystemExit(f"refusing to replace valid complete output: {final}")
    if final.exists(): raise SystemExit(f"refusing existing incomplete output; move it aside first: {final}")
    stage = Path(args.staging_dir).resolve() if args.staging_dir else final.with_name(final.name + ".staging." + uuid.uuid4().hex)
    if stage.parent != final.parent: raise SystemExit("staging directory must share output parent/filesystem")
    stage.mkdir(parents=True, exist_ok=True); (stage / "shards").mkdir(exist_ok=True)
    state_path = stage / "state.json"
    if state_path.exists(): state = load_json(state_path)
    else:
        state = {"version": 2, "target_tokens": args.target_tokens, "val_tokens": args.val_tokens, "shard_tokens": args.shard_tokens, "documents_seen": 0, "document_offset": 0, "committed_tokens": 0, "split_tokens": {"val": 0, "train": 0}, "shards": {"val": [], "train": []}, "commit_sequence": 0}
        atomic_json(state_path, state)
    if any(state[k] != getattr(args, k) for k in ("target_tokens", "val_tokens", "shard_tokens")): raise SystemExit("resume arguments disagree with durable state")
    state = recover_checkpoint(stage, state)
    active_segment = None
    if args.fast_continuation:
        if state["split_tokens"]["val"] != args.val_tokens:
            raise SystemExit("--fast-continuation requires an already-complete validation split")
        active_segment = begin_fast_continuation(state, args)
    atomic_json(state_path, state)
    print(f"staging directory (use to resume): {stage}", flush=True)
    status(stage, "writing", state)
    enc, ds = encoder(args, seed=active_segment["seed"] if active_segment else None)
    # Deterministically replay streaming input to the durable document/token offset.
    it = iter(ds)
    if not args.fast_continuation:
        for _ in range(state["documents_seen"]): next(it)
    work = copy.deepcopy(state)
    stop_after = int(os.environ.get("PREP_TEST_STOP_AFTER_COMMITS", "0"))
    batches = bounded_document_batches(it, args.tokenizer_batch_docs, args.tokenizer_batch_bytes)
    tokenizer_backend = TokenizerBackend(args.tokenizer_backend, enc, args.tokenizer, args.tokenizer_threads, allow_process=not args.smoke).start()
    failed = True
    try:
        while work["committed_tokens"] < args.target_tokens + args.val_tokens:
            texts = next(batches)
            # Reader -> one bounded persistent-backend call -> ordered writer.
            # There is one in-flight batch and no per-document durable state.
            for tokens in tokenizer_backend.tokenize(texts):
                offset = work["document_offset"]
                while offset < len(tokens) and work["committed_tokens"] < args.target_tokens + args.val_tokens:
                    split = "val" if work["split_tokens"]["val"] < args.val_tokens else "train"
                    limit = args.val_tokens if split == "val" else args.target_tokens
                    shard_index = len(work["shards"][split])
                    part = stage / "shards" / f"{split}-{shard_index:06d}.bin.part"
                    used = part.stat().st_size // DTYPE_BYTES if part.exists() else 0
                    take = min(len(tokens) - offset, limit - work["split_tokens"][split], args.shard_tokens - used)
                    with part.open("ab") as f:
                        array("H", tokens[offset:offset + take]).tofile(f)
                    offset += take; work["split_tokens"][split] += take; work["committed_tokens"] += take; work["document_offset"] = offset
                    if used + take == args.shard_tokens or work["split_tokens"][split] == limit:
                        done = part.with_suffix("")
                        work["shards"][split].append({"path": str(done.relative_to(stage)), "tokens": used + take})
                        work["commit_sequence"] += 1
                        if args.fast_continuation:
                            update_fast_segment_end(work)
                        commit_shard(part, done, used + take, work)
                        state = copy.deepcopy(work)
                        atomic_json(state_path, state); status(stage, "writing", state)
                        if stop_after and state["commit_sequence"] >= stop_after:
                            raise SystemExit("test interruption after durable shard boundary")
                if offset == len(tokens):
                    work["documents_seen"] += 1; work["document_offset"] = 0
                if work["committed_tokens"] >= args.target_tokens + args.val_tokens:
                    break
        failed = False
    finally:
        tokenizer_backend.close(failed=failed)
    status(stage, "verification", state)
    files = {}
    for split, expected in (("train", args.target_tokens), ("val", args.val_tokens)):
        target = stage / f"{split}.bin"; partial = target.with_suffix(".bin.part")
        with partial.open("wb") as out:
            for shard in state["shards"][split]:
                src = stage / shard["path"]
                done = load_json(src.with_suffix(src.suffix + ".done"))
                if done["tokens"] != shard["tokens"] or src.stat().st_size != done["bytes"]: raise RuntimeError(f"invalid committed shard {src}")
                shutil.copyfileobj(src.open("rb"), out, 4 * 1024 * 1024)
            out.flush(); os.fsync(out.fileno())
        os.replace(partial, target)
        if target.stat().st_size != expected * DTYPE_BYTES: raise RuntimeError(f"wrong {split} size")
        files[split] = {"path": target.name, "bytes": target.stat().st_size, "tokens": expected, "sha256": sha256(target), "dtype": "uint16"}
    if args.fast_continuation:
        close_fast_segment(state, "completed")
        atomic_json(state_path, state)
    manifest = {"format": "finewebedu-gpt2-uint16-v1", "complete": True, "target_tokens": args.target_tokens, "val_tokens": args.val_tokens, "tokenizer": args.tokenizer, "dataset": args.dataset, "name": args.name, "revision": args.revision, "split": args.split, "files": files}
    if args.fast_continuation or "segments" in state:
        manifest["segments"] = state["segments"]
        manifest["segment_policy"] = "mixed FineWeb-Edu segments may overlap; post-interruption fast continuation is not byte-identical"
    # The preceding loop is the sole checksum pass during generation.  Do not
    # re-read 40+ GB merely to compare hashes we just generated.
    if not validate(stage, manifest, checksums=False): raise RuntimeError("final validation failed")
    atomic_json(stage / "manifest.json", manifest)
    if stage.stat().st_dev != final.parent.stat().st_dev: raise RuntimeError("refusing cross-filesystem promotion")
    os.replace(stage, final)
    print(f"promoted verified output: {final}", flush=True)
    return args.hard_exit


if __name__ == "__main__":
    try:
        hard_exit = main()
    except SystemExit as exc:
        code = exc.code
        if code not in (None, 0):
            print(str(code), file=sys.stderr, flush=True)
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(code if isinstance(code, int) else 1)
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush(); sys.stderr.flush()
        if hard_exit:
            # Y400's extension stack has crashed during interpreter teardown
            # after successful streaming/pool shutdown and durable writes.
            os._exit(0)
