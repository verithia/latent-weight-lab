import torch

from examples.nanogpt.analyze_sparse_moe_nodeprivate_fullframe_ceiling import (
    NodePrivateFullFrame,
    absolute_output_pass,
)
from examples.nanogpt.analyze_sparse_moe_sharedframe_fullrank_pregelu_oracle import (
    SharedFrameFullRankPreGelu,
)


def _parent() -> SharedFrameFullRankPreGelu:
    torch.manual_seed(3)
    module = SharedFrameFullRankPreGelu(
        write_basis=torch.randn(8, 4),
        hidden_width=16,
        padded_width=16,
        tensor_layers=2,
        experts=2,
        input_frame_seed=11,
        procedural_map_seed=12,
        learn_frame=True,
        device="cpu",
    )
    with torch.no_grad():
        module.hidden_bias.normal_()
        module.output_modulation.normal_()
    return module


def test_candidate_control_have_identical_initial_function() -> None:
    parent = _parent()
    candidate = NodePrivateFullFrame(
        parent, layers=[0, 1], private_frame=True, device="cpu"
    )
    control = NodePrivateFullFrame(
        parent, layers=[0, 1], private_frame=False, device="cpu"
    )
    x = torch.randn(2, 5, 8)
    direction = torch.randn_like(x)
    for layer in (0, 1):
        left = candidate.function_and_jvp(x, direction, layer=layer)
        right = control.function_and_jvp(x, direction, layer=layer)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_private_frame_change_is_node_local() -> None:
    parent = _parent()
    candidate = NodePrivateFullFrame(
        parent, layers=[0, 1], private_frame=True, device="cpu"
    )
    x = torch.randn(1, 5, 8)
    direction = torch.randn_like(x)
    before_other = candidate.function_and_jvp(
        x, direction, layer=0, expert=1
    )[0]
    with torch.no_grad():
        candidate.raw_frames["0"][0, 0, 0] += 1.0
    after_other = candidate.function_and_jvp(
        x, direction, layer=0, expert=1
    )[0]
    torch.testing.assert_close(before_other, after_other, atol=0, rtol=0)


def test_absolute_output_gate() -> None:
    row = {
        "mixture_recovery_mean": 0.9,
        "mixture_recovery_minimum_layer": 0.8,
        "minimum_expert_recovery": 0.6,
    }
    gates = {
        "heldout_mixture_recovery_mean_min_each_bank": 0.8,
        "heldout_mixture_recovery_every_layer_min_each_bank": 0.7,
        "heldout_expert_recovery_min_each_bank": 0.5,
    }
    assert absolute_output_pass(row, gates)
