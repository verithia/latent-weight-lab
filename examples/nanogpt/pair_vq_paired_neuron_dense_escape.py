"""Zero-update complete-neuron dense-escape ceiling for Pair-VQ MLPs."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import torch


def gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0))) + (
        values * torch.exp(-0.5 * values.square())
        / math.sqrt(2.0 * math.pi)
    )


def direction_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    reference64 = reference.detach().double()
    candidate64 = candidate.detach().double()
    reference_energy = float(reference64.square().sum())
    candidate_energy = float(candidate64.square().sum())
    error_energy = float((candidate64 - reference64).square().sum())
    inner = float((reference64 * candidate64).sum())
    cosine = inner / max(
        math.sqrt(reference_energy * candidate_energy), 1e-30
    )
    return {
        "reference_energy": reference_energy,
        "candidate_energy": candidate_energy,
        "error_energy": error_energy,
        "inner": inner,
        "relative_error": math.sqrt(
            error_energy / max(reference_energy, 1e-30)
        ),
        "cosine": cosine,
        "positive_line_recovery": max(cosine, 0.0) ** 2,
        "candidate_to_reference_energy": candidate_energy
        / max(reference_energy, 1e-30),
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    reference = sum(float(row["reference_energy"]) for row in rows)
    candidate = sum(float(row["candidate_energy"]) for row in rows)
    error = sum(float(row["error_energy"]) for row in rows)
    inner = sum(float(row["inner"]) for row in rows)
    cosine = inner / max(math.sqrt(reference * candidate), 1e-30)
    return {
        "reference_energy": reference,
        "candidate_energy": candidate,
        "error_energy": error,
        "inner": inner,
        "relative_error": math.sqrt(error / max(reference, 1e-30)),
        "cosine": cosine,
        "positive_line_recovery": max(cosine, 0.0) ** 2,
        "candidate_to_reference_energy": candidate / max(reference, 1e-30),
    }


def _functional_terms(
    *,
    inputs: torch.Tensor,
    preactivation: torch.Tensor,
    hidden: torch.Tensor,
    c_fc_update: torch.Tensor,
    c_proj_update: torch.Tensor,
    c_proj_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drive = (inputs @ c_fc_update.T) * gelu_derivative(preactivation)
    full = drive @ c_proj_weight.T + hidden @ c_proj_update.T
    return drive, hidden, full


def _fit_order(
    *,
    drive: torch.Tensor,
    hidden: torch.Tensor,
    full: torch.Tensor,
    c_proj_update: torch.Tensor,
    c_proj_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # <A_j, full> is additive over complete neurons.  Sorting this exact
    # contribution maximizes positive-line capture for each fixed cardinality.
    inner = (
        drive * (full @ c_proj_weight)
        + hidden * (full @ c_proj_update)
    ).sum(dim=0)
    order = torch.argsort(inner, descending=True, stable=True)
    return order, inner


def _selected_functional_action(
    *,
    drive: torch.Tensor,
    hidden: torch.Tensor,
    c_proj_update: torch.Tensor,
    c_proj_weight: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    return (
        drive.index_select(1, selected)
        @ c_proj_weight.index_select(1, selected).T
        + hidden.index_select(1, selected)
        @ c_proj_update.index_select(1, selected).T
    )


def _masked_ambient_metrics(
    *,
    c_fc_update: torch.Tensor,
    c_proj_update: torch.Tensor,
    selected: torch.Tensor,
) -> dict[str, float]:
    reference_energy = float(
        c_fc_update.double().square().sum()
        + c_proj_update.double().square().sum()
    )
    candidate_energy = float(
        c_fc_update.index_select(0, selected).double().square().sum()
        + c_proj_update.index_select(1, selected).double().square().sum()
    )
    cosine = math.sqrt(candidate_energy / max(reference_energy, 1e-30))
    return {
        "reference_energy": reference_energy,
        "candidate_energy": candidate_energy,
        "error_energy": max(reference_energy - candidate_energy, 0.0),
        "inner": candidate_energy,
        "relative_error": math.sqrt(
            max(reference_energy - candidate_energy, 0.0)
            / max(reference_energy, 1e-30)
        ),
        "cosine": cosine,
        "positive_line_recovery": cosine * cosine,
        "candidate_to_reference_energy": candidate_energy
        / max(reference_energy, 1e-30),
    }


def _task_line(
    *,
    c_fc_gradient: torch.Tensor,
    c_proj_gradient: torch.Tensor,
    c_fc_update: torch.Tensor,
    c_proj_update: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[float, float]:
    per_neuron = (
        (c_fc_gradient.double() * c_fc_update.double()).sum(dim=1)
        + (c_proj_gradient.double() * c_proj_update.double()).sum(dim=0)
    )
    dense = float(per_neuron.sum())
    candidate = float(per_neuron.index_select(0, selected).sum())
    return dense, candidate


@torch.no_grad()
def evaluate_layer(
    *,
    identity: str,
    fit: dict[str, torch.Tensor],
    heldout: dict[str, torch.Tensor],
    c_fc_update: torch.Tensor,
    c_proj_update: torch.Tensor,
    c_fc_gradient: torch.Tensor,
    c_proj_gradient: torch.Tensor,
    c_proj_weight: torch.Tensor,
    fractions: tuple[float, ...],
    maximum_actionable_fraction: float,
    device: str,
) -> dict[str, Any]:
    tensors = {
        "c_fc_update": c_fc_update,
        "c_proj_update": c_proj_update,
        "c_fc_gradient": c_fc_gradient,
        "c_proj_gradient": c_proj_gradient,
        "c_proj_weight": c_proj_weight,
    }
    moved = {
        name: value.detach().to(device=device, dtype=torch.float32)
        for name, value in tensors.items()
    }
    fit_moved = {
        name: value.detach().to(device=device, dtype=torch.float32)
        for name, value in fit.items()
    }
    heldout_moved = {
        name: value.detach().to(device=device, dtype=torch.float32)
        for name, value in heldout.items()
    }
    fit_drive, fit_hidden, fit_full = _functional_terms(
        inputs=fit_moved["input"],
        preactivation=fit_moved["preactivation"],
        hidden=fit_moved["hidden"],
        c_fc_update=moved["c_fc_update"],
        c_proj_update=moved["c_proj_update"],
        c_proj_weight=moved["c_proj_weight"],
    )
    heldout_drive, heldout_hidden, heldout_full = _functional_terms(
        inputs=heldout_moved["input"],
        preactivation=heldout_moved["preactivation"],
        hidden=heldout_moved["hidden"],
        c_fc_update=moved["c_fc_update"],
        c_proj_update=moved["c_proj_update"],
        c_proj_weight=moved["c_proj_weight"],
    )
    order, fit_inner = _fit_order(
        drive=fit_drive,
        hidden=fit_hidden,
        full=fit_full,
        c_proj_update=moved["c_proj_update"],
        c_proj_weight=moved["c_proj_weight"],
    )
    hidden_width = int(moved["c_fc_update"].shape[0])
    if moved["c_proj_update"].shape[1] != hidden_width:
        raise ValueError("c_fc row and c_proj column inventories disagree")
    candidates: dict[str, Any] = {}
    for fraction in fractions:
        count = min(hidden_width, max(1, int(round(hidden_width * fraction))))
        selected = order[:count]
        candidate_action = _selected_functional_action(
            drive=heldout_drive,
            hidden=heldout_hidden,
            c_proj_update=moved["c_proj_update"],
            c_proj_weight=moved["c_proj_weight"],
            selected=selected,
        )
        dense_line, candidate_line = _task_line(
            c_fc_gradient=moved["c_fc_gradient"],
            c_proj_gradient=moved["c_proj_gradient"],
            c_fc_update=moved["c_fc_update"],
            c_proj_update=moved["c_proj_update"],
            selected=selected,
        )
        candidates[str(fraction)] = {
            "count": count,
            "fraction": count / hidden_width,
            "functional": direction_metrics(heldout_full, candidate_action),
            "ambient": _masked_ambient_metrics(
                c_fc_update=moved["c_fc_update"],
                c_proj_update=moved["c_proj_update"],
                selected=selected,
            ),
            "dense_task_line": dense_line,
            "candidate_task_line": candidate_line,
            "task_line_retention": candidate_line
            / (math.copysign(max(abs(dense_line), 1e-30), dense_line)),
        }
    actionable_count = min(
        hidden_width,
        max(1, int(round(hidden_width * maximum_actionable_fraction))),
    )
    actionable = order[:actionable_count].detach().cpu().tolist()
    order_bytes = order.detach().cpu().to(torch.int32).numpy().tobytes()
    return {
        "identity": identity,
        "hidden_width": hidden_width,
        "fit_full_functional_energy": float(fit_full.double().square().sum()),
        "heldout_full_functional_energy": float(
            heldout_full.double().square().sum()
        ),
        "fit_positive_contribution_fraction": float(
            (fit_inner > 0).float().mean()
        ),
        "order_sha256": hashlib.sha256(order_bytes).hexdigest(),
        "actionable_selected_indices": actionable,
        "candidates": candidates,
    }


def summarize_fraction(
    *, layer_rows: list[dict[str, Any]], fraction: float, n_embd: int
) -> dict[str, Any]:
    key = str(fraction)
    rows = [row["candidates"][key] for row in layer_rows]
    functional = aggregate_metrics([row["functional"] for row in rows])
    ambient = aggregate_metrics([row["ambient"] for row in rows])
    dense_line = sum(float(row["dense_task_line"]) for row in rows)
    candidate_line = sum(float(row["candidate_task_line"]) for row in rows)
    count_per_layer = int(rows[0]["count"])
    selected_values = len(rows) * count_per_layer * 2 * int(n_embd)
    return {
        "fraction": float(rows[0]["fraction"]),
        "count_per_layer": count_per_layer,
        "selected_values": selected_values,
        "selected_weight_bytes_bf16": selected_values * 2,
        "selected_optimizer_bytes_fp32": selected_values * 4,
        "selected_weight_plus_optimizer_bytes": selected_values * 6,
        "functional": functional,
        "ambient": ambient,
        "task_line_retention": candidate_line
        / math.copysign(max(abs(dense_line), 1e-30), dense_line),
        "minimum_layer_functional_cosine": min(
            float(row["functional"]["cosine"]) for row in rows
        ),
        "minimum_layer_task_line_retention": min(
            float(row["task_line_retention"]) for row in rows
        ),
        "layers": [
            {
                "identity": layer["identity"],
                "functional": row["functional"],
                "ambient": row["ambient"],
                "task_line_retention": row["task_line_retention"],
            }
            for layer, row in zip(layer_rows, rows, strict=True)
        ],
    }
