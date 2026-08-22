import numpy as np

from examples.nanogpt.analyze_pair_vq_fp16_highbyte_dictionary import (
    _selection,
    decode_item,
    encode_item,
    evaluate_candidate,
    top_k_table,
)


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


def test_top_k_ties_use_ascending_value() -> None:
    counts = np.zeros(256, dtype=np.int64)
    counts[[7, 3, 9]] = [4, 4, 2]
    assert top_k_table(counts, 4).tolist() == [3, 7, 9, 0]


def test_global_dictionary_round_trip_and_accounting() -> None:
    words = np.array([0x1234, 0x12FF, 0xAB01, 0x1200, 0xCD02] * 13, dtype=np.uint16)
    item = _item(words)
    table = top_k_table(np.bincount(item["high"], minlength=256), 4)
    encoded, account = encode_item(
        item,
        k=4,
        block_words=16,
        granularity="global",
        region_tables={"global": table},
    )
    recovered = decode_item(encoded, count=words.size, block_words=16, granularity="global")
    assert np.array_equal(recovered, words)
    assert account["low_literal_bytes"] == words.size
    assert account["index_bytes"] == (words.size * 2 + 7) // 8
    assert account["exception_bitmap_bytes"] == (words.size + 7) // 8


def test_block_dictionary_round_trip_with_exceptions() -> None:
    high = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 7, dtype=np.uint16)
    words = (high << 8) | np.arange(high.size, dtype=np.uint16)
    item = _item(words)
    encoded, account = encode_item(
        item,
        k=4,
        block_words=9,
        granularity="block",
        region_tables={},
    )
    recovered = decode_item(encoded, count=words.size, block_words=9, granularity="block")
    assert np.array_equal(recovered, words)
    assert account["exceptions"] > 0


def test_candidate_and_selection_are_deterministic() -> None:
    a = _item(np.array([0x1234, 0x1235, 0x5636, 0x7837] * 16, dtype=np.uint16), "a")
    b = _item(np.array([0x1234, 0x9A35, 0xBC36, 0xDE37] * 16, dtype=np.uint16), "b", "c_proj")
    row = evaluate_candidate([a, b], k=4, block_words=16, granularity="matrix")
    assert row["exact_roundtrip"]
    primary, control = _selection([row], row["total_bytes"])
    assert primary is row
    assert control is row
