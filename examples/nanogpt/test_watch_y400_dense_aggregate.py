from __future__ import annotations

import json

from examples.nanogpt import watch_y400_dense_aggregate as watcher


def test_send_allows_slow_bridge_ack_and_mentions_agent(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class Response:
        def read(self) -> bytes:
            return b'{"ok":true}'

    class Opener:
        def open(self, request, *, timeout):
            observed["timeout"] = timeout
            observed["body"] = json.loads(request.data)
            return Response()

    monkeypatch.setattr(watcher.urllib.request, "build_opener", lambda *_args: Opener())
    assert watcher.send("chat", "PROGRESS: run 20%") is True
    assert observed["timeout"] == 300
    assert observed["body"] == {
        "chat_id": "chat",
        "text": "@Codex PROGRESS: run 20%",
    }


def test_failed_milestone_delivery_remains_pending_after_threshold() -> None:
    assert watcher.milestone_crossings(330, 340, 677, {20}) == [50]
    assert watcher.milestone_crossings(340, 350, 677, {20}) == [50]
    assert watcher.milestone_crossings(350, 350, 677, {20}) == [50]


def test_owned_or_unreached_milestones_are_not_replayed() -> None:
    assert watcher.milestone_crossings(330, 340, 677, {20, 50}) == []
    assert watcher.milestone_crossings(300, 320, 677, {20}) == []
    assert watcher.milestone_crossings(350, 349, 677, {20}) == []
