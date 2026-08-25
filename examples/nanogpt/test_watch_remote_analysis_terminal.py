import json
from pathlib import Path

from examples.nanogpt.watch_remote_analysis_terminal import (
    acquire_exclusive_lock,
    callback_already_delivered,
    watch_identity,
)


def test_watch_identity_is_stable_and_run_specific() -> None:
    identity = watch_identity("PRO6", 42, "/status", "/result")
    assert identity == watch_identity("PRO6", 42, "/status", "/result")
    assert identity != watch_identity("PRO6", 43, "/status", "/result")


def test_terminal_delivery_requires_matching_terminal_state() -> None:
    signature = ["finished", 0, "abc"]
    delivered = {
        "state": "finished",
        "terminal_signature": signature,
        "callback_delivered_at_unix": 1.0,
    }
    assert callback_already_delivered(delivered)
    assert callback_already_delivered(delivered, signature)
    assert not callback_already_delivered(delivered, ["finished", 0, "def"])
    assert not callback_already_delivered({"state": "running"})


def test_exclusive_lock_rejects_duplicate_watcher(tmp_path: Path) -> None:
    lock_path = tmp_path / "watch.lock"
    first = acquire_exclusive_lock(lock_path)
    assert first is not None
    try:
        assert acquire_exclusive_lock(lock_path) is None
    finally:
        first.close()


def test_terminal_receipt_round_trip_shape(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    payload = {
        "state": "finished",
        "terminal_signature": ["finished", 0, "abc"],
        "callback_delivered_at_unix": 1.0,
    }
    receipt.write_text(json.dumps(payload))
    assert callback_already_delivered(json.loads(receipt.read_text()))
