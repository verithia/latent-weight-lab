from __future__ import annotations

import math

import torch

from examples.nanogpt.analyze_mlp_moving_activation_frame_oracle import (
    moving_frame_transport_metrics,
)


def test_moving_frame_transport_recovers_principal_plane_motion() -> None:
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(7, 8, generator=generator)
    start = torch.eye(8)[:, :2]
    end = torch.zeros(8, 2)
    angles = (0.35, 0.7)
    transform = torch.eye(8)
    for start_axis, complement_axis, angle in (
        (0, 2, angles[0]),
        (1, 3, angles[1]),
    ):
        cosine = math.cos(angle)
        sine = math.sin(angle)
        transform[start_axis, start_axis] = cosine
        transform[complement_axis, complement_axis] = cosine
        transform[complement_axis, start_axis] = sine
        transform[start_axis, complement_axis] = -sine
        end[start_axis, start_axis] = cosine
        end[complement_axis, start_axis] = sine
    target = source @ transform.T
    result = moving_frame_transport_metrics(source, target, start, end)
    assert result["forward_transport_recovery"] > 0.999999
    assert (
        result["forward_transport_recovery"]
        > result["reverse_transport_recovery"]
    )
    assert result["principal_mapping_error_fro"] < 1e-10


def test_moving_frame_transport_validates_basis_shape() -> None:
    source = torch.eye(4)
    target = source.clone()
    start = torch.eye(4)[:, :2]
    end = torch.eye(4)[:, :1]
    try:
        moving_frame_transport_metrics(source, target, start, end)
    except ValueError as error:
        assert "activation bases" in str(error)
    else:
        raise AssertionError("shape mismatch must fail")
