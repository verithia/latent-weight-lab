from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    activation_effect_metrics,
    aggregate_results,
    exact_muon_update,
)
from examples.nanogpt.muon import Muon


def test_exact_muon_update_matches_optimizer_step() -> None:
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(8, 4, generator=generator)
    gradient = torch.randn(8, 4, generator=generator)
    buffer = torch.randn(8, 4, generator=generator)
    parameter = torch.nn.Parameter(weight.clone())
    optimizer = Muon(
        [parameter],
        lr=0.003,
        momentum=0.9,
        weight_decay=0.1,
        ns_steps=4,
    )
    optimizer.state[parameter]["momentum_buffer"] = buffer.clone()
    parameter.grad = gradient.clone()
    expected, _direction, diagnostics = exact_muon_update(
        weight,
        gradient,
        buffer,
        learning_rate=0.003,
        momentum=0.9,
        weight_decay=0.1,
        ns_steps=4,
    )
    optimizer.step()
    torch.testing.assert_close(
        parameter.detach() - weight, expected, rtol=2e-6, atol=2e-6
    )
    assert diagnostics["polar_scale"] == 2**0.5


def test_activation_effect_identity_is_exact() -> None:
    generator = torch.Generator().manual_seed(23)
    mlp_input = torch.randn(19, 4, generator=generator)
    pre_gelu = torch.randn(19, 8, generator=generator)
    cproj = torch.randn(4, 8, generator=generator)
    update = 0.01 * torch.randn(8, 4, generator=generator)
    metrics = activation_effect_metrics(
        mlp_input,
        pre_gelu,
        cproj,
        update,
        update,
        device="cpu",
        chunk_size=5,
    )
    for point in ("post_gelu", "mlp_output"):
        assert abs(metrics[point]["cosine"] - 1.0) < 1e-6
        assert abs(metrics[point]["fixed_scale_recovery"] - 1.0) < 1e-6
        assert abs(metrics[point]["positive_line_recovery"] - 1.0) < 1e-6


def _row(
    candidate: str,
    window: str,
    *,
    recovery: float,
    descent: float,
) -> dict[str, float | int | str]:
    return {
        "candidate": candidate,
        "window": window,
        "layer": 0,
        "weight_target_energy": 1.0,
        "weight_positive_line_recovery": recovery,
        "post_gelu_target_energy": 1.0,
        "post_gelu_positive_line_recovery": recovery,
        "mlp_output_target_energy": 1.0,
        "mlp_output_positive_line_recovery": recovery,
        "predicted_ce_decrease": descent,
    }


def test_aggregate_requires_every_registered_gate() -> None:
    recoveries = {
        "dense_exact": 1.0,
        "fresh_expansion64": 0.20,
        "fresh_expansion88": 0.30,
        "random_expansion88": 0.15,
    }
    descents = {
        "dense_exact": 1.0,
        "fresh_expansion64": 0.20,
        "fresh_expansion88": 0.30,
        "random_expansion88": 0.15,
    }
    rows = [
        _row(
            candidate,
            window,
            recovery=recoveries[candidate],
            descent=descents[candidate],
        )
        for candidate in recoveries
        for window in ("fit", "validation_a", "validation_b")
    ]
    finite_rows = [
        {
            "candidate": candidate,
            "window": window,
            "loss": loss,
        }
        for window in ("validation_a", "validation_b")
        for candidate, loss in (
            ("baseline", 5.0),
            ("dense_exact", 4.8),
            ("fresh_expansion64", 4.9),
            ("fresh_expansion88", 4.85),
            ("random_expansion88", 4.95),
        )
    ]
    result = aggregate_results(rows, finite_rows)
    assert result["decision"].startswith("SELECT_")
    next(
        row
        for row in finite_rows
        if row["window"] == "validation_b"
        and row["candidate"] == "fresh_expansion88"
    )["loss"] = 5.1
    result = aggregate_results(rows, finite_rows)
    assert result["decision"] == "REJECT_EXACT_CURRENT_SPARSE_CFC_MATCHER"
    assert not result["gates"]["finite_ce"]
