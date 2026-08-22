import inspect

import numpy as np
import pytest

from examples.nanogpt.analyze_pair_vq_fp16_momentum_entropy import (
    CODECS,
    conditional_lower_mantissa_entropy,
    entropy_from_counts,
    prefix_entropy,
    run_audit,
    word_counts,
)


def test_result_boolean_literal_is_python() -> None:
    source = inspect.getsource(run_audit)
    assert '"gpu_used": False' in source
    assert " false" not in source
    assert " true" not in source
    assert " null" not in source


def test_entropy_known_distributions() -> None:
    assert entropy_from_counts(np.array([8, 0])) == 0.0
    assert entropy_from_counts(np.array([4, 4])) == pytest.approx(1.0)
    counts = word_counts(np.arange(65536, dtype=np.uint16))
    assert entropy_from_counts(counts) == pytest.approx(16.0)
    assert prefix_entropy(counts, 8) == pytest.approx(14.0)
    assert conditional_lower_mantissa_entropy(counts, 8) == pytest.approx(2.0)


@pytest.mark.parametrize("codec", sorted(CODECS))
@pytest.mark.parametrize("shape", [(7, 19), (8, 512)])
def test_registered_codecs_are_bit_exact(codec: str, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(20260822)
    words = rng.integers(0, 65536, size=np.prod(shape), dtype=np.uint16)
    encode, decode = CODECS[codec]
    recovered = decode(encode(words, shape), shape)
    assert recovered.dtype == np.uint16
    assert np.array_equal(recovered, words)


def test_codecs_preserve_fp16_special_words() -> None:
    words = np.array(
        [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0x7C00, 0xFC00, 0x7E00],
        dtype=np.uint16,
    )
    matrix = np.tile(words, 17).reshape(9, 17)
    for encode, decode in CODECS.values():
        assert np.array_equal(decode(encode(matrix, matrix.shape), matrix.shape), matrix.reshape(-1))
