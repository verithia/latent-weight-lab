from __future__ import annotations

import json

import torch

from examples.nanogpt.analyze_pair_vq_shadow_endpoint import (
    install_variant,
    parse_training_log,
    summarize_refreshes,
)
from examples.nanogpt.muon_pair_vq import MuonPairVQLinear


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = MuonPairVQLinear(
            4,
            8,
            bias=False,
            stages=1,
            base_seed=1,
            weight_std=0.02,
            layer_id=0,
        )
        self.c_proj = MuonPairVQLinear(
            8,
            4,
            bias=False,
            stages=1,
            base_seed=2,
            weight_std=0.02,
            layer_id=0,
        )


def _row(step: int, c_fc: bool) -> dict[str, float | int]:
    return {
        "optimizer_step": step,
        "feedback_quantization_residual_energy": 1.0,
        "feedback_target_energy": 1000.0,
        "current_request_energy": 100.0,
        "projection_residual_energy": 80.0,
        "request_energy": 100.0,
        "feedback_codec_energy_recovery": 0.999,
        "conserved_requested_step_energy_recovery": 0.99,
        "requested_step_energy_recovery": 0.2,
        "feedback_energy": 10.0,
        "feedback_to_weight_energy_ratio": 0.1,
        "out_features": 8 if c_fc else 4,
        "in_features": 4 if c_fc else 8,
    }


def test_parse_and_summarize_refreshes(tmp_path) -> None:
    rows = [_row(0, index < 12) for index in range(24)]
    path = tmp_path / "run.log"
    path.write_text(
        "\n".join(
            ["muon_matched_givens_refresh " + json.dumps(row) for row in rows]
            + [
                "step 0: train loss 10.0, val loss 11.0",
                "perf iter=10 tokens_per_s=100.0 iter_ms=10.0 opt_ms=2.0 peak_mib=9.0",
            ]
        )
        + "\n"
    )
    parsed, evaluations, performance = parse_training_log(path)
    summary = summarize_refreshes(parsed)
    assert evaluations[0]["val"] == 11.0
    assert performance[0]["tokens_per_s"] == 100.0
    assert summary["refresh_count"] == 1
    assert abs(summary["minimum_codec_weighted_recovery"] - 0.999) < 1e-12
    assert abs(summary["minimum_current_weighted_recovery"] - 0.99) < 1e-12
    assert abs(summary["minimum_realized_weighted_recovery"] - 0.2) < 1e-12


def test_install_variant_is_side_selective() -> None:
    model = TinyModel()
    base = {
        "c_fc": model.c_fc.weight.detach().clone(),
        "c_proj": model.c_proj.weight.detach().clone(),
    }
    feedback = {
        "c_fc": torch.ones_like(model.c_fc.weight),
        "c_proj": 2.0 * torch.ones_like(model.c_proj.weight),
    }
    install_variant(model, base, feedback, "c_fc_shadow")
    torch.testing.assert_close(model.c_fc.weight, base["c_fc"] + 1.0)
    torch.testing.assert_close(model.c_proj.weight, base["c_proj"])
    install_variant(model, base, feedback, "full_shadow")
    torch.testing.assert_close(model.c_fc.weight, base["c_fc"] + 1.0)
    torch.testing.assert_close(model.c_proj.weight, base["c_proj"] + 2.0)
    install_variant(model, base, feedback, "native")
    torch.testing.assert_close(model.c_fc.weight, base["c_fc"])
    torch.testing.assert_close(model.c_proj.weight, base["c_proj"])
