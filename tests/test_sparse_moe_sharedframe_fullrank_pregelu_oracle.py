import torch

from examples.nanogpt.analyze_sparse_moe_sharedframe_fullrank_pregelu_oracle import (
    SharedFrameFullRankPreGelu,
    coordinate_count,
    procedural_signs,
)


def _module(*, learn_frame: bool) -> SharedFrameFullRankPreGelu:
    write, _ = torch.linalg.qr(torch.randn(8, 4, generator=torch.Generator().manual_seed(7)))
    return SharedFrameFullRankPreGelu(
        write_basis=write,
        hidden_width=16,
        padded_width=16,
        tensor_layers=2,
        experts=2,
        input_frame_seed=11,
        procedural_map_seed=13,
        learn_frame=learn_frame,
        device="cpu",
    )


def test_registered_coordinate_accounting() -> None:
    assert coordinate_count(
        input_width=768, write_rank=448, hidden_width=1536,
        tensor_layers=12, experts=8,
    ) == 1_124_352
    assert 226_492_416 / 1_124_352 > 200.0


def test_procedural_signs_repeat_exactly() -> None:
    left = procedural_signs(
        tensor_layers=2, experts=3, padded_width=16, base_seed=17
    )
    right = procedural_signs(
        tensor_layers=2, experts=3, padded_width=16, base_seed=17
    )
    assert torch.equal(left, right)
    assert set(left.unique().tolist()) == {-1, 1}


def test_candidate_and_control_have_identical_initial_function() -> None:
    candidate, control = _module(learn_frame=True), _module(learn_frame=False)
    x = torch.randn(2, 5, 8)
    direction = torch.randn_like(x)
    candidate_pair = candidate.function_and_jvp(x, direction, layer=1)
    control_pair = control.function_and_jvp(x, direction, layer=1)
    assert torch.equal(candidate.raw_frame, control.raw_frame)
    assert torch.equal(candidate.procedural_signs, control.procedural_signs)
    assert all(torch.equal(left, right) for left, right in zip(candidate_pair, control_pair))
    assert all(torch.count_nonzero(value) == 0 for value in candidate_pair)


def test_candidate_frame_receives_gradient_after_write_modulation_moves() -> None:
    module = _module(learn_frame=True)
    with torch.no_grad():
        module.output_modulation.fill_(0.25)
    x = torch.randn(2, 5, 8)
    direction = torch.randn_like(x)
    output, jvp = module.function_and_jvp(x, direction, layer=0)
    loss = output.square().mean() + jvp.square().mean()
    loss.backward()
    assert module.raw_frame.grad is not None
    assert torch.isfinite(module.raw_frame.grad).all()
    assert module.raw_frame.grad.abs().sum() > 0


def test_square_frame_is_algebraically_full_rank() -> None:
    module = _module(learn_frame=True)
    assert torch.linalg.matrix_rank(module.input_frame()) == 8
    assert module.counted_coordinates() == coordinate_count(
        input_width=8, write_rank=4, hidden_width=16,
        tensor_layers=2, experts=2,
    )
