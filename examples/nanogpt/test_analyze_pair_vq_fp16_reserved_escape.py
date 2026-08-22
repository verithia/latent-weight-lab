import numpy as np

from examples.nanogpt.analyze_pair_vq_fp16_reserved_escape import (
    decode_item,
    encode_item,
    evaluate_candidate,
    select,
)
from examples.nanogpt.analyze_pair_vq_fp16_highbyte_dictionary import top_k_table


def _item(words: np.ndarray, name: str = "m", scope: str = "c_fc") -> dict:
    words = np.asarray(words, dtype=np.uint16).reshape(-1)
    return {
        "name": name,
        "scope": scope,
        "layer": 0,
        "shape": (1, words.size),
        "words": words,
        "low": (words & 255).astype(np.uint8),
        "high": (words >> 8).astype(np.uint8),
    }


def test_reserved_escape_round_trip_has_no_bitmap() -> None:
    words = np.array([0x1234, 0x12FF, 0xAB01, 0xCD02, 0xEF03] * 17, dtype=np.uint16)
    item = _item(words)
    table = top_k_table(np.bincount(item["high"], minlength=256), 3)
    encoded, account = encode_item(
        item,
        k=4,
        block_words=19,
        granularity="scope",
        region_tables={"c_fc": table},
    )
    recovered = decode_item(encoded, count=words.size, block_words=19, granularity="scope")
    assert np.array_equal(recovered, words)
    assert "exception_bitmap_bytes" not in account
    assert account["exceptions"] > 0


def test_scope_adaptive_widths_are_accounted_exactly() -> None:
    a = _item(np.array([0x1234, 0x1335, 0x1436, 0x1537] * 32, dtype=np.uint16), "a", "c_fc")
    b = _item(np.array([0x8034, 0x9135, 0xA236, 0xB337] * 32, dtype=np.uint16), "b", "c_proj")
    row = evaluate_candidate(
        [a, b], k_fc=8, k_proj=16, block_words=32, granularity="matrix"
    )
    assert row["exact_roundtrip"]
    expected_indices = (a["words"].size * 3 + 7) // 8 + (b["words"].size * 4 + 7) // 8
    assert row["components"]["index_bytes"] == expected_indices


def test_selection_is_deterministic() -> None:
    row = {
        "name": "x",
        "exact_roundtrip": True,
        "total_bytes": 99,
        "block_words": 4096,
        "granularity": "scope",
        "k_fc": 8,
        "k_proj": 16,
    }
    primary, control = select([row], 99)
    assert primary is row
    assert control is row
