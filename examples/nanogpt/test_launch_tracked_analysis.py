from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_tracked_analysis_records_success(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    status = tmp_path / "status.json"
    log = tmp_path / "run.log"
    child = tmp_path / "child.py"
    child.write_text(
        "from pathlib import Path\n"
        f"Path({str(result)!r}).write_text('{{\"ok\": true}}\\n')\n"
        "print('done')\n"
    )
    subprocess.run(
        [
            sys.executable,
            "examples/nanogpt/launch_tracked_analysis.py",
            "--status",
            str(status),
            "--result",
            str(result),
            "--log",
            str(log),
            "--run-label",
            "synthetic",
            "--cwd",
            str(Path.cwd()),
            "--",
            sys.executable,
            str(child),
        ],
        check=True,
    )
    payload = json.loads(status.read_text())
    assert payload["state"] == "finished"
    assert payload["exit_code"] == 0
    assert payload["result_exists"] is True
    assert payload["result_sha256"]
    assert log.read_text().strip() == "done"
