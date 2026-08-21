"""Paired dense-path optimizer-transition oracle for full-MLP Pair-VQ.

The dense parent remains the only training path.  A nonintervening compact
momentum tracker consumes the identical clipped dense MLP gradients under the
current teacher-forced Pair-VQ assignments.  Registered probes compare dense
and compact combined momentum, Muon polar updates, and isolated production
retractions without retaining any candidate dense state.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.dense_pair_vq_shadow import (
    DensePairVQShadowObserver,
    atomic_json,
    sha256_file,
)
from examples.nanogpt.muon import muon_update
from examples.nanogpt.muon_pair_vq import MuonPairVQLinear


PLAN_SCHEMA = "mai_124m_pair_vq_optimizer_transition_oracle_plan_v1"
RESULT_SCHEMA = "mai_pair_vq_optimizer_transition_oracle_result_v1"


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def direction_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    reference64 = reference.detach().double()
    candidate64 = candidate.detach().double()
    reference_energy = float(reference64.square().sum())
    candidate_energy = float(candidate64.square().sum())
    error_energy = float((candidate64 - reference64).square().sum())
    inner = float((candidate64 * reference64).sum())
    return {
        "reference_energy": reference_energy,
        "candidate_energy": candidate_energy,
        "error_energy": error_energy,
        "relative_error": math.sqrt(
            error_energy / max(reference_energy, 1e-30)
        ),
        "cosine": inner
        / max(math.sqrt(reference_energy * candidate_energy), 1e-30),
        "positive_line_energy_recovery": 1.0
        - (
            reference_energy
            - max(inner, 0.0) ** 2 / max(candidate_energy, 1e-30)
        )
        / max(reference_energy, 1e-30),
    }


@torch.no_grad()
def update_compact_momentum(
    module: MuonPairVQLinear,
    compact_momentum: torch.Tensor,
    gradient: torch.Tensor,
    *,
    momentum: float,
) -> torch.Tensor:
    """Apply the exact production code-conditioned momentum transition."""
    gradient_pairs = gradient.detach().float().reshape(-1, 2)
    expanded = torch.zeros_like(gradient_pairs)
    for stage in range(module.stages):
        codes = module.codes[stage].long()
        accum = torch.zeros_like(module.codebooks[stage])
        accum.index_add_(0, codes, gradient_pairs)
        counts = torch.bincount(codes, minlength=module.codebook_size)
        live = counts > 0
        means = torch.zeros_like(module.codebooks[stage])
        means[live] = accum[live] / counts[live, None]
        compact_momentum[stage].mul_(float(momentum)).add_(means)
        expanded.add_(compact_momentum[stage].index_select(0, codes))
    return expanded.div_(module.stages).reshape_as(gradient)


def _aggregate_direction(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float]:
    reference_energy = sum(float(row[key]["reference_energy"]) for row in rows)
    candidate_energy = sum(float(row[key]["candidate_energy"]) for row in rows)
    error_energy = sum(float(row[key]["error_energy"]) for row in rows)
    inner = sum(
        math.sqrt(
            max(float(row[key]["reference_energy"]), 0.0)
            * max(float(row[key]["candidate_energy"]), 0.0)
        )
        * float(row[key]["cosine"])
        for row in rows
    )
    return {
        "reference_energy": reference_energy,
        "candidate_energy": candidate_energy,
        "error_energy": error_energy,
        "relative_error": math.sqrt(
            error_energy / max(reference_energy, 1e-30)
        ),
        "cosine": inner
        / max(math.sqrt(reference_energy * candidate_energy), 1e-30),
        "positive_line_energy_recovery": 1.0
        - (
            reference_energy
            - max(inner, 0.0) ** 2 / max(candidate_energy, 1e-30)
        )
        / max(reference_energy, 1e-30),
        "worst_matrix_cosine": min(float(row[key]["cosine"]) for row in rows),
        "worst_matrix_relative_error": max(
            float(row[key]["relative_error"]) for row in rows
        ),
    }


def _aggregate_retraction(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float]:
    requested = sum(float(row[key]["requested_delta_energy"]) for row in rows)
    achieved = sum(float(row[key]["achieved_delta_energy"]) for row in rows)
    error = sum(float(row[key]["retraction_error_energy"]) for row in rows)
    inner = sum(float(row[key]["requested_achieved_inner"]) for row in rows)
    return {
        "requested_delta_energy": requested,
        "achieved_delta_energy": achieved,
        "retraction_error_energy": error,
        "requested_step_energy_recovery": 1.0 - error / max(requested, 1e-30),
        "requested_update_cosine": inner
        / max(math.sqrt(requested * achieved), 1e-30),
        "worst_matrix_requested_step_energy_recovery": min(
            float(row[key]["requested_step_energy_recovery"]) for row in rows
        ),
        "worst_matrix_requested_update_cosine": min(
            float(row[key]["requested_update_cosine"]) for row in rows
        ),
    }


class PairVQOptimizerTransitionOracle:
    """Compare dense and production compact Muon state on one dense path."""

    def __init__(
        self,
        model,
        observer: DensePairVQShadowObserver,
        optimizer,
        *,
        plan_path: Path,
        plan_sha256: str,
        result_path: Path,
    ) -> None:
        if sha256_file(plan_path) != plan_sha256:
            raise ValueError("optimizer-transition plan identity mismatch")
        self.plan_path = plan_path
        self.plan_sha256 = plan_sha256
        self.plan = json.loads(plan_path.read_text())
        if self.plan.get("schema_version") != PLAN_SCHEMA:
            raise ValueError("optimizer-transition plan schema mismatch")
        self.model = model
        self.observer = observer
        self.result_path = result_path
        self.update_indices = set(
            int(step)
            for step in self.plan["frozen_protocol"][
                "optimizer_update_indices"
            ]
        )
        self.reported_steps = dict(
            zip(
                self.plan["frozen_protocol"]["optimizer_update_indices"],
                self.plan["frozen_protocol"][
                    "reported_post_update_state_steps"
                ],
                strict=True,
            )
        )
        children = list(getattr(optimizer, "optimizers", [optimizer]))
        self._dense_owner: dict[str, tuple[Any, dict[str, Any]]] = {}
        for name, module in observer._dense_modules.items():
            matches = []
            for child in children:
                for group in child.param_groups:
                    if any(parameter is module.weight for parameter in group["params"]):
                        matches.append((child, group))
            if len(matches) != 1:
                raise ValueError(
                    f"expected one dense optimizer owner for {name}, found {len(matches)}"
                )
            self._dense_owner[name] = matches[0]
        self._compact_momentum = {
            name: torch.zeros_like(module.codebooks)
            for name, module in observer._shadow_modules.items()
        }
        self.records: list[dict[str, Any]] = []
        self._run_identity_sha256: str | None = None

    @staticmethod
    @torch.no_grad()
    def _retract(
        module: MuonPairVQLinear,
        dense_start: torch.Tensor,
        requested_delta: torch.Tensor,
        *,
        refresh_codes: bool,
    ) -> tuple[dict[str, float], torch.Tensor]:
        candidate = copy.deepcopy(module)
        candidate.forward_visible_feedback = True
        levels = torch.zeros(
            candidate.feedback_level_shape,
            device=dense_start.device,
            dtype=torch.float32,
        )
        codes = torch.zeros(
            candidate.feedback_code_shape,
            device=dense_start.device,
            dtype=torch.uint8,
        )
        center = None
        if candidate.feedback_center_shape is not None:
            center = torch.zeros(
                candidate.feedback_center_shape,
                device=dense_start.device,
                dtype=torch.float32,
            )
        initial_residual = dense_start.float() - candidate.weight.float()
        candidate.fit_feedback_(
            initial_residual.reshape(-1, candidate.vector_length),
            levels,
            codes,
            center,
        )
        start_virtual = candidate.weight.float() + candidate.decode_feedback(
            levels, codes, center
        ).reshape_as(candidate.weight)
        target = start_virtual + requested_delta.float()
        candidate.project_requested_weight_(target, refresh_codes=refresh_codes)
        residual = target - candidate.weight.float()
        candidate.fit_feedback_(
            residual.reshape(-1, candidate.vector_length),
            levels,
            codes,
            center,
        )
        achieved_virtual = candidate.weight.float() + candidate.decode_feedback(
            levels, codes, center
        ).reshape_as(candidate.weight)
        achieved_delta = achieved_virtual - start_virtual
        requested_energy = float(requested_delta.double().square().sum())
        achieved_energy = float(achieved_delta.double().square().sum())
        error_energy = float(
            (achieved_delta.double() - requested_delta.double()).square().sum()
        )
        inner = float((achieved_delta.double() * requested_delta.double()).sum())
        metrics = {
            "requested_delta_energy": requested_energy,
            "achieved_delta_energy": achieved_energy,
            "retraction_error_energy": error_energy,
            "requested_achieved_inner": inner,
            "requested_step_energy_recovery": 1.0
            - error_energy / max(requested_energy, 1e-30),
            "requested_update_cosine": inner
            / max(math.sqrt(requested_energy * achieved_energy), 1e-30),
            "initial_virtual_weight_energy_recovery": 1.0
            - float((start_virtual.double() - dense_start.double()).square().sum())
            / max(float(dense_start.double().square().sum()), 1e-30),
        }
        del candidate
        return metrics, achieved_delta

    def _aggregate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for label, selected in (
            ("all", rows),
            ("c_fc", [row for row in rows if row["side"] == "c_fc"]),
            ("c_proj", [row for row in rows if row["side"] == "c_proj"]),
        ):
            momentum_error = sum(
                float(row["decomposition"]["momentum_polar_error_energy"])
                for row in selected
            )
            retraction_error = sum(
                float(row["decomposition"]["retraction_error_energy"])
                for row in selected
            )
            interaction = sum(
                float(row["decomposition"]["interaction_energy"])
                for row in selected
            )
            final_error = sum(
                float(row["final_compact_vs_dense_request_error_energy"])
                for row in selected
            )
            denominator = max(final_error, 1e-30)
            payload[label] = {
                "combined_prepolar": _aggregate_direction(
                    selected, "combined_prepolar"
                ),
                "polar_update": _aggregate_direction(selected, "polar_update"),
                "requested_delta": _aggregate_direction(
                    selected, "requested_delta"
                ),
                "dense_retraction": _aggregate_retraction(
                    selected, "dense_retraction"
                ),
                "compact_retraction": _aggregate_retraction(
                    selected, "compact_retraction"
                ),
                "decomposition": {
                    "momentum_polar_error_energy": momentum_error,
                    "retraction_error_energy": retraction_error,
                    "interaction_energy": interaction,
                    "final_compact_vs_dense_request_error_energy": final_error,
                    "momentum_polar_fraction": momentum_error / denominator,
                    "retraction_fraction": retraction_error / denominator,
                    "interaction_fraction": interaction / denominator,
                    "decomposition_closure_relative_error": abs(
                        final_error
                        - momentum_error
                        - retraction_error
                        - interaction
                    )
                    / denominator,
                },
            }
        return payload

    @torch.no_grad()
    def before_step(
        self, *, optimizer_update_index: int, run_identity_sha256: str
    ) -> dict[str, Any] | None:
        self._run_identity_sha256 = run_identity_sha256
        probe = int(optimizer_update_index) in self.update_indices
        rows = []
        for name in sorted(self.observer._dense_modules):
            dense = self.observer._dense_modules[name]
            shadow = self.observer._shadow_modules[name]
            gradient = dense.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"missing or non-finite dense gradient for {name}")
            gradient = gradient.detach().float()
            owner, group = self._dense_owner[name]
            momentum = float(group["momentum"])
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            state = owner.state[dense.weight]
            dense_buffer = state.get("momentum_buffer")
            if dense_buffer is None:
                dense_buffer = torch.zeros_like(dense.weight)
            dense_next = dense_buffer.float() * momentum + gradient
            dense_combined = gradient + momentum * dense_next
            expanded = update_compact_momentum(
                shadow,
                self._compact_momentum[name],
                gradient,
                momentum=momentum,
            )
            compact_combined = gradient + momentum * expanded
            if not probe:
                continue
            dense_update = muon_update(dense_combined, steps=ns_steps).float()
            compact_update = muon_update(compact_combined, steps=ns_steps).float()
            decay_delta = dense.weight.detach().float() * (-lr * weight_decay)
            dense_delta = decay_delta - lr * dense_update
            compact_delta = decay_delta - lr * compact_update
            refresh = (
                int(optimizer_update_index) % shadow.code_refresh_interval == 0
            )
            dense_retraction, dense_achieved = self._retract(
                shadow,
                dense.weight.detach().float(),
                dense_delta,
                refresh_codes=refresh,
            )
            compact_retraction, compact_achieved = self._retract(
                shadow,
                dense.weight.detach().float(),
                compact_delta,
                refresh_codes=refresh,
            )
            momentum_error = compact_delta.double() - dense_delta.double()
            retraction_error = compact_achieved.double() - compact_delta.double()
            final_error = compact_achieved.double() - dense_delta.double()
            rows.append(
                {
                    "module": name,
                    "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                    "layer": int(name.split(".")[2]),
                    "combined_prepolar": direction_metrics(
                        dense_combined, compact_combined
                    ),
                    "polar_update": direction_metrics(
                        dense_update, compact_update
                    ),
                    "requested_delta": direction_metrics(
                        dense_delta, compact_delta
                    ),
                    "dense_retraction": dense_retraction,
                    "compact_retraction": compact_retraction,
                    "dense_achieved_vs_dense_request_error_energy": float(
                        (dense_achieved.double() - dense_delta.double())
                        .square()
                        .sum()
                    ),
                    "final_compact_vs_dense_request_error_energy": float(
                        final_error.square().sum()
                    ),
                    "decomposition": {
                        "momentum_polar_error_energy": float(
                            momentum_error.square().sum()
                        ),
                        "retraction_error_energy": float(
                            retraction_error.square().sum()
                        ),
                        "interaction_energy": float(
                            2.0 * (momentum_error * retraction_error).sum()
                        ),
                    },
                }
            )
        if not probe:
            return None
        record = {
            "optimizer_update_index": int(optimizer_update_index),
            "reported_post_update_state_step": int(
                self.reported_steps[int(optimizer_update_index)]
            ),
            "aggregate": self._aggregate(rows),
            "matrices": rows,
        }
        self.records.append(record)
        self._write(status="running")
        return {
            "optimizer_update_index": int(optimizer_update_index),
            "reported_post_update_state_step": record[
                "reported_post_update_state_step"
            ],
            "aggregate": record["aggregate"]["all"],
        }

    def _gate(self) -> dict[str, Any]:
        late_steps = set(
            int(step)
            for step in self.plan["frozen_protocol"][
                "primary_late_post_update_state_steps"
            ]
        )
        late = [
            row
            for row in self.records
            if row["reported_post_update_state_step"] in late_steps
        ]
        if len(late) != len(late_steps):
            return {"ready": False, "classification": None}
        momentum_fractions = [
            float(row["aggregate"]["all"]["decomposition"]["momentum_polar_fraction"])
            for row in late
        ]
        retraction_fractions = [
            float(row["aggregate"]["all"]["decomposition"]["retraction_fraction"])
            for row in late
        ]
        momentum_exceeds_retraction = [
            float(row["aggregate"]["all"]["decomposition"]["momentum_polar_error_energy"])
            > float(row["aggregate"]["all"]["decomposition"]["retraction_error_energy"])
            for row in late
        ]
        retraction_exceeds_momentum = [not value for value in momentum_exceeds_retraction]
        momentum_primary = all(value >= 0.70 for value in momentum_fractions) and all(
            momentum_exceeds_retraction
        )
        retraction_primary = all(value >= 0.70 for value in retraction_fractions) and all(
            retraction_exceeds_momentum
        )
        classification = "MIXED_OR_UNRESOLVED"
        if momentum_primary:
            classification = "COMPACT_MOMENTUM_POLAR_PRIMARY"
        elif retraction_primary:
            classification = "COMPACT_RETRACTION_PRIMARY"
        return {
            "ready": True,
            "classification": classification,
            "minimum_late_momentum_polar_fraction": min(momentum_fractions),
            "minimum_late_retraction_fraction": min(retraction_fractions),
            "momentum_exceeds_retraction_at_every_late_probe": all(
                momentum_exceeds_retraction
            ),
            "retraction_exceeds_momentum_at_every_late_probe": all(
                retraction_exceeds_momentum
            ),
            "maximum_late_decomposition_closure_relative_error": max(
                float(
                    row["aggregate"]["all"]["decomposition"][
                        "decomposition_closure_relative_error"
                    ]
                )
                for row in late
            ),
            "all_metrics_finite": _all_finite(late),
        }

    def _write(
        self,
        *,
        status: str,
        dense_terminal_losses: dict[str, float] | None = None,
        fixed_eval_indices_sha256: str | None = None,
    ) -> None:
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": status,
            "plan": {"path": str(self.plan_path), "sha256": self.plan_sha256},
            "run_identity_sha256": self._run_identity_sha256,
            "fixed_eval_indices_sha256": fixed_eval_indices_sha256,
            "dense_terminal_losses": dense_terminal_losses,
            "records": self.records,
            "gate": self._gate(),
        }
        if not _all_finite(payload):
            raise RuntimeError("optimizer-transition payload contains non-finite values")
        atomic_json(self.result_path, payload)

    def finalize(
        self,
        *,
        dense_terminal_losses: dict[str, float],
        fixed_eval_indices_sha256: str,
    ) -> dict[str, Any]:
        self._write(
            status="finished",
            dense_terminal_losses=dense_terminal_losses,
            fixed_eval_indices_sha256=fixed_eval_indices_sha256,
        )
        return {"status": "finished", "gate": self._gate()}
