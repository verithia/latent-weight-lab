from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def test_launcher_creates_independent_live_session(tmp_path: Path) -> None:
    stdout = tmp_path / "child.log"
    receipt = tmp_path / "receipt.json"
    launcher = Path(__file__).with_name("launch_local_detached.py")
    subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--stdout",
            str(stdout),
            "--receipt",
            str(receipt),
            "--cwd",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        check=True,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    pid = int(payload["pid"])
    try:
        os.kill(pid, 0)
        assert os.getsid(pid) == pid
        assert payload["start_new_session"] is True
    finally:
        os.killpg(pid, signal.SIGTERM)
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
