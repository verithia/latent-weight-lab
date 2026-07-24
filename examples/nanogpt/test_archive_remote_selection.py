from pathlib import Path

import pytest

from examples.nanogpt.archive_remote_selection import (
    local_manifest,
    partition_names,
    rsync_command,
    validate_names,
)


def test_validate_names_requires_safe_unique_direct_children() -> None:
    assert validate_names(["first", "second"]) == ["first", "second"]
    for invalid in ([], ["."], [".."], ["nested/child"], ["same", "same"]):
        with pytest.raises(ValueError):
            validate_names(invalid)


def test_local_manifest_preserves_selected_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "ckpt.pt").write_bytes(b"first checkpoint")
    (second / "ckpt.meta.json").write_bytes(b'{"next_iter": 1}\n')

    manifest = local_manifest(tmp_path)

    assert sorted(manifest) == ["first/ckpt.pt", "second/ckpt.meta.json"]
    assert manifest["first/ckpt.pt"]["bytes"] == 16


def test_rsync_command_selects_only_requested_directories(tmp_path: Path) -> None:
    command = rsync_command(
        "Y400",
        "/remote/root",
        tmp_path,
        ["first", "second"],
    )

    assert "--include=/first/" in command
    assert "--include=/first/***" in command
    assert "--include=/second/" in command
    assert "--include=/second/***" in command
    assert "--exclude=*" in command
    assert "Y400:/remote/root/" in command


def test_partition_names_limits_and_balances_parallel_transfers() -> None:
    assert partition_names(["a", "b", "c"], 1) == [["a", "b", "c"]]
    assert partition_names(["a", "b", "c"], 2) == [["a", "c"], ["b"]]
    assert partition_names(["a", "b", "c"], 8) == [["a"], ["b"], ["c"]]
