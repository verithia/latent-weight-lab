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
    assert observed["timeout"] == 90
    assert observed["body"] == {
        "chat_id": "chat",
        "text": "@Codex PROGRESS: run 20%",
    }
