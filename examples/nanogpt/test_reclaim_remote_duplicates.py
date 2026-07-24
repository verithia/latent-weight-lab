import copy

import pytest

from examples.nanogpt.reclaim_remote_duplicates import (
    expected_observation,
    manifest_sha256,
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


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["entries"][0].update(path="../ckpt.pt"),
        lambda payload: payload.update(expected_total_bytes=9),
        lambda payload: payload["entries"][0].update(sha256="invalid"),
        lambda payload: payload["entries"].append(copy.deepcopy(payload["entries"][0])),
    ],
)
def test_validate_manifest_rejects_unsafe_or_inconsistent_entries(mutation) -> None:
    payload = manifest()
    mutation(payload)
    with pytest.raises(ValueError):
        validate_manifest(payload)
