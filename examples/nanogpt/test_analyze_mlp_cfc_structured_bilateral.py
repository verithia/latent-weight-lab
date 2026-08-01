import torch

from examples.nanogpt.analyze_mlp_cfc_structured_bilateral import (
    _axis_view,
    _from_axis_view,
    stage_chunks,
)


def test_axis_views_round_trip() -> None:
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    for axis in ("output", "input"):
        restored = _from_axis_view(_axis_view(values, axis), axis)
        torch.testing.assert_close(restored, values)


def test_stage_chunks_preserve_budget_and_native_limit() -> None:
    assert stage_chunks(0) == []
    assert stage_chunks(56) == [56]
    assert stage_chunks(88) == [64, 24]
    assert stage_chunks(128) == [64, 64]
    for stages in (32, 56, 64, 72, 80, 88, 96, 128):
        chunks = stage_chunks(stages)
        assert sum(chunks) == stages
        assert all(0 < chunk <= 64 for chunk in chunks)


def test_registered_allocations_have_equal_coordinates() -> None:
    allocations = ((88, 0), (80, 32), (72, 64), (64, 96), (56, 128))
    coordinates = {
        output * (3072 // 2) + input_ * (768 // 2)
        for output, input_ in allocations
    }
    assert coordinates == {135168}
