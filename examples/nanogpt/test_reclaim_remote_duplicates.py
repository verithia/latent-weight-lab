import copy

import pytest

from examples.nanogpt.reclaim_remote_duplicates import (
    expected_observation,
    manifest_sha256,
    same_storage_namespace,
    validate_manifest,
)


def manifest() -> dict:
    return {
        "schema_version": 1,
        "primary": {"host": "Y400", "root": "/workspace/runs"},
        "authority": {"host": "Y800", "root": "/workspace/runs"},
        "entries": [{"path": "run/ckpt.pt", "bytes": 10, "sha256": "a" * 64}],
        "expected_total_bytes": 10,
    }


def test_validate_manifest_and_expected_observation() -> None:
    payload = manifest()
    validate_manifest(payload)
    assert expected_observation(payload["entries"]) == {
        "run/ckpt.pt": {"exists": True, "bytes": 10, "sha256": "a" * 64}
    }
    assert len(manifest_sha256(payload)) == 64


def test_same_storage_namespace_requires_matching_mount_and_inode() -> None:
    identity = {
        "filesystem_type": "ceph",
        "mount_source": "monitors:/volume/share",
        "mount_root": "/volume/share",
        "inode": 123,
    }
    assert same_storage_namespace(identity, copy.deepcopy(identity))
    distinct = copy.deepcopy(identity)
    distinct["inode"] = 124
    assert not same_storage_namespace(identity, distinct)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["entries"][0].update(path="../ckpt.pt"),
        lambda payload: payload.update(expected_total_bytes=9),
        lambda payload: payload["entries"][0].update(sha256="invalid"),
        lambda payload: payload["entries"].append(copy.deepcopy(payload["entries"][0])),
        lambda payload: payload.update(policy={"do_not_execute": True}),
    ],
)
def test_validate_manifest_rejects_unsafe_or_inconsistent_entries(mutation) -> None:
    payload = manifest()
    mutation(payload)
    with pytest.raises(ValueError):
        validate_manifest(payload)
