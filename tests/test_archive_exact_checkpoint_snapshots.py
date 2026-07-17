from __future__ import annotations

import json

from examples.nanogpt.archive_exact_checkpoint_snapshots import archive_checkpoint


def test_archives_only_the_requested_published_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "ckpt.meta.json").write_text(json.dumps({"next_iter": 12}))
    archive = tmp_path / "snapshots"

    assert not archive_checkpoint(tmp_path, archive, 11)
    assert archive_checkpoint(tmp_path, archive, 12)
    assert (archive / "ckpt_iter000012.pt").read_bytes() == b"checkpoint"
    assert json.loads((archive / "ckpt_iter000012.meta.json").read_text())["analysis_snapshot"]["published_next_iter"] == 12
