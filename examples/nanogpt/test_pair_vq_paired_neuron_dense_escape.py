from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from examples.nanogpt.pair_vq_paired_neuron_dense_escape import (
    evaluate_layer,
    summarize_fraction,
)


def _toy_layer() -> dict[str, object]:
    hidden = torch.eye(4)
    zeros = torch.zeros(4, 2)
    c_proj_update = torch.tensor(
        [[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.5]]
    )
    samples = {
        "input": torch.zeros(4, 2),
        "preactivation": torch.zeros(4, 4),
        "hidden": hidden,
    }
    return evaluate_layer(
        identity="transformer.h.0",
        fit=samples,
        heldout=samples,
        c_fc_update=zeros,
        c_proj_update=c_proj_update,
        c_fc_gradient=zeros,
        c_proj_gradient=c_proj_update,
        c_proj_weight=torch.zeros(2, 4),
        fractions=(0.25, 1.0),
        maximum_actionable_fraction=0.25,
        device="cpu",
    )


def test_complete_neuron_order_and_full_fraction_are_exact() -> None:
    layer = _toy_layer()
    assert layer["actionable_selected_indices"] == [0]
    quarter = layer["candidates"]["0.25"]
    assert quarter["count"] == 1
    assert 0.79 < quarter["functional"]["cosine"] < 0.80
    full = layer["candidates"]["1.0"]
    assert full["functional"]["cosine"] == 1.0
    assert full["functional"]["relative_error"] == 0.0
    assert full["ambient"]["cosine"] == 1.0
    assert full["task_line_retention"] == 1.0


def test_dense_escape_byte_accounting_is_complete_pair_accounting() -> None:
    layer = _toy_layer()
    summary = summarize_fraction(
        layer_rows=[layer, layer], fraction=0.25, n_embd=2
    )
    assert summary["selected_values"] == 2 * 1 * 2 * 2
    assert summary["selected_weight_bytes_bf16"] == 16
    assert summary["selected_optimizer_bytes_fp32"] == 32
    assert summary["selected_weight_plus_optimizer_bytes"] == 48


def test_paired_neuron_dense_escape_plan_is_frozen_and_nonintervening() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/124m_pair_vq_paired_neuron_dense_escape_oracle_plan.json"
    )
    plan = json.loads(path.read_text())
    assert plan["schema_version"].endswith("_v1")
    assert plan["frozen_protocol"]["probe_steps"] == [180, 238]
    assert plan["frozen_protocol"]["model_updates_from_oracle"] == 0
    assert plan["dense_escape"]["fractions"] == [
        0.015625,
        0.03125,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
    ]
    assert plan["dense_escape"]["maximum_actionable_dense_fraction"] == 0.25
    assert plan["decision_rule"]["automatic_language_model_training"] is False
    assert plan["decision_rule"]["automatic_fraction_sweep"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "cef3603cac2a38f70646a3bbfe0aa74816c0f51ecbc12d424ec9c20a4f39a448"
    )
