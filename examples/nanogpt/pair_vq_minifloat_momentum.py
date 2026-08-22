"""Nonintervening exponent-preserving momentum audit for Pair-VQ MLPs.

The production Pair-VQ path owns an FP16 ambient Muon-momentum buffer.  This
observer carries independent, deterministic E5M* shadows, compares their
causal next requests after the production polar map, and never updates model
parameters or optimizer state.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.dense_pair_vq_lowbit_momentum import (
    _aggregate_direction,
    _all_finite,
)
from examples.nanogpt.dense_pair_vq_optimizer_transition import direction_metrics
from examples.nanogpt.dense_pair_vq_shadow import atomic_json, sha256_file
from examples.nanogpt.muon import muon_update
from examples.nanogpt.muon_pair_vq import MuonPairVQ, MuonPairVQLinear


PLAN_SCHEMA = "mai_124m_pair_vq_minifloat_momentum_precision_frontier_plan_v3"
RESULT_SCHEMA = "mai_pair_vq_minifloat_momentum_precision_frontier_result_v1"
VALID_STAGES = {"stage_ab_deterministic_replay"}


@torch.no_grad()
def round_fp16_mantissa(
    values: torch.Tensor,
    *,
    mantissa_bits: int,
) -> torch.Tensor:
    """Round finite values to FP16 E5M* with ties-to-even and saturation.

    The returned tensor is FP16.  It has the exact IEEE FP16 sign/exponent
    layout and only ``mantissa_bits`` retained fraction bits.  The operation
    first rounds the input to FP16, matching the registered production state.
    """
    if not 0 <= int(mantissa_bits) <= 10:
        raise ValueError("mantissa_bits must be in [0, 10]")
    source = values.detach().to(torch.float16).contiguous()
    if not torch.isfinite(source).all():
        raise ValueError("minifloat momentum source must be finite")
    if int(mantissa_bits) == 10:
        return source.clone()

    dropped = 10 - int(mantissa_bits)
    raw = source.view(torch.int16).to(torch.int32) & 0xFFFF
    sign = raw & 0x8000
    magnitude = raw & 0x7FFF
    remainder_mask = (1 << dropped) - 1
    half = 1 << (dropped - 1)
    truncated = magnitude >> dropped
    remainder = magnitude & remainder_mask
    increment = (remainder > half) | (
        (remainder == half) & ((truncated & 1) != 0)
    )
    rounded = (truncated + increment.to(torch.int32)) << dropped
    # Finite values that round through 0x7c00 saturate at the largest finite
    # value representable with the selected mantissa width.
    maximum_finite = 0x7C00 - (1 << dropped)
    rounded = rounded.clamp_max(maximum_finite)
    encoded = sign | rounded
    signed = torch.where(encoded >= 0x8000, encoded - 0x10000, encoded)
    decoded = signed.to(torch.int16).view(torch.float16)
    if not torch.isfinite(decoded).all():
        raise RuntimeError("minifloat rounding produced a non-finite value")
    return decoded


@torch.no_grad()
def advance_minifloat_momentum(
    previous: torch.Tensor | None,
    gradient: torch.Tensor,
    *,
    momentum: float,
    mantissa_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one causal E5M* history without a hidden dense residual."""
    if previous is None:
        previous_decoded = torch.zeros_like(gradient, dtype=torch.float32)
    else:
        if previous.dtype != torch.float16:
            raise ValueError("encoded minifloat momentum must use FP16 container")
        previous_decoded = previous.float()
    target = previous_decoded.mul(float(momentum)).add(gradient.detach().float())
    encoded = round_fp16_mantissa(target, mantissa_bits=mantissa_bits)
    return encoded, encoded.float()


class PairVQMinifloatMomentumOracle:
    """Audit nested E5M* histories beside a production FP16 Pair-VQ owner."""

    def __init__(
        self,
        model,
        optimizer,
        *,
        plan_path: Path,
        plan_sha256: str,
        result_path: Path,
        stage: str,
    ) -> None:
        if stage not in VALID_STAGES:
            raise ValueError(f"unsupported minifloat audit stage: {stage}")
        if sha256_file(plan_path) != plan_sha256:
            raise ValueError("minifloat momentum plan identity mismatch")
        self.plan_path = plan_path
        self.plan_sha256 = plan_sha256
        self.plan = json.loads(plan_path.read_text())
        if self.plan.get("schema_version") != PLAN_SCHEMA:
            raise ValueError("minifloat momentum plan schema mismatch")
        decision = self.plan["decision_gate"]
        for field in (
            "automatic_endpoint",
            "automatic_scale_up",
            "automatic_horizon_transfer",
            "automatic_sweep",
        ):
            if bool(decision[field]):
                raise ValueError("minifloat plan cannot pre-authorize follow-up work")
        protocol = self.plan["frozen_protocol"]
        if int(protocol["parameter_updates_by_candidates"]) != 0:
            raise ValueError("minifloat candidates must be nonintervening")

        pair_optimizers = [
            child
            for child in getattr(optimizer, "optimizers", [optimizer])
            if isinstance(child, MuonPairVQ)
        ]
        if len(pair_optimizers) != 1:
            raise ValueError(
                f"expected one MuonPairVQ optimizer, found {len(pair_optimizers)}"
            )
        self.owner = pair_optimizers[0]
        if not self.owner.fp16_ambient_momentum:
            raise ValueError("minifloat audit requires production FP16 momentum")
        names = {
            id(module): name
            for name, module in model.named_modules()
            if isinstance(module, MuonPairVQLinear)
        }
        self.modules: dict[str, MuonPairVQLinear] = {}
        for module in self.owner.modules_by_id.values():
            name = names.get(id(module))
            if name is None:
                raise ValueError("Pair-VQ optimizer module is absent from model")
            self.modules[name] = module
        elements = sum(module.element_count for module in self.modules.values())
        if elements != int(protocol["momentum_elements"]):
            raise ValueError(
                f"minifloat momentum inventory mismatch: {elements} != "
                f"{protocol['momentum_elements']}"
            )

        formats = self.plan["formats"]
        self.candidate_order = list(formats["candidate_order"])
        self.candidates = dict(formats["candidates"])
        if self.candidate_order != list(self.candidates):
            raise ValueError("candidate order must match registered candidate mapping")
        self.stage = stage
        stage_protocol = protocol[stage]
        self.update_indices = {
            int(value) for value in stage_protocol["probe_update_indices"]
        }
        self.futility_indices = {
            int(value)
            for value in stage_protocol["early_futility_update_indices"]
        }
        self.probe_only = False
        self.result_path = result_path
        self.records: list[dict[str, Any]] = []
        self._codec_seconds = {name: 0.0 for name in self.candidate_order}
        self._run_identity_sha256: str | None = None
        self._states: dict[str, dict[str, torch.Tensor | None]] = {
            candidate: {} for candidate in self.candidate_order
        }
        for name, module in self.modules.items():
            owner_state = self.owner.state[module.weight]
            resident = owner_state.get("ambient_momentum")
            for candidate in self.candidate_order:
                if resident is None:
                    encoded = None
                else:
                    encoded = round_fp16_mantissa(
                        resident,
                        mantissa_bits=int(
                            self.candidates[candidate]["mantissa_bits"]
                        ),
                    )
                self._states[candidate][name] = encoded

    @staticmethod
    def _side(name: str) -> str:
        return "c_fc" if name.endswith(".c_fc") else "c_proj"

    def _aggregate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for candidate in self.candidate_order:
            candidate_rows = [row for row in rows if row["candidate"] == candidate]
            payload[candidate] = {}
            for label, selected in (
                ("all", candidate_rows),
                ("c_fc", [row for row in candidate_rows if row["side"] == "c_fc"]),
                (
                    "c_proj",
                    [row for row in candidate_rows if row["side"] == "c_proj"],
                ),
            ):
                payload[candidate][label] = {
                    key: _aggregate_direction(selected, key)
                    for key in ("momentum_state", "combined_prepolar", "polar_update")
                }
        return payload

    @torch.no_grad()
    def before_step(
        self,
        *,
        optimizer_update_index: int,
        run_identity_sha256: str,
    ) -> dict[str, Any] | None:
        self._run_identity_sha256 = run_identity_sha256
        probe = int(optimizer_update_index) in self.update_indices
        rows: list[dict[str, Any]] = []
        for name in sorted(self.modules):
            module = self.modules[name]
            gradient = module.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"missing or non-finite Pair-VQ gradient for {name}")
            gradient = gradient.detach().float()
            matches = [
                group
                for group in self.owner.param_groups
                if any(parameter is module.weight for parameter in group["params"])
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one owner group for {name}")
            group = matches[0]
            momentum = float(group["momentum"])
            ns_steps = int(group["ns_steps"])
            resident = self.owner.state[module.weight].get("ambient_momentum")
            reference_previous = (
                torch.zeros_like(gradient) if resident is None else resident.float()
            )
            reference_next = (
                reference_previous.mul(momentum).add(gradient).to(torch.float16)
            ).float()
            reference_combined = gradient + momentum * reference_next
            reference_update = None
            if probe:
                reference_update = muon_update(
                    reference_combined, steps=ns_steps
                ).float()
            for candidate in self.candidate_order:
                started = time.perf_counter()
                encoded, decoded = advance_minifloat_momentum(
                    self._states[candidate][name],
                    gradient,
                    momentum=momentum,
                    mantissa_bits=int(self.candidates[candidate]["mantissa_bits"]),
                )
                if gradient.is_cuda:
                    torch.cuda.synchronize(gradient.device)
                self._codec_seconds[candidate] += time.perf_counter() - started
                self._states[candidate][name] = encoded
                if not probe:
                    continue
                combined = gradient + momentum * decoded
                update = muon_update(combined, steps=ns_steps).float()
                rows.append(
                    {
                        "candidate": candidate,
                        "module": name,
                        "side": self._side(name),
                        "layer": int(name.split(".")[2]),
                        "momentum_state": direction_metrics(reference_next, decoded),
                        "combined_prepolar": direction_metrics(
                            reference_combined, combined
                        ),
                        "polar_update": direction_metrics(reference_update, update),
                    }
                )
        if not probe:
            return None
        report_step = int(optimizer_update_index) + 1
        record = {
            "optimizer_update_index": int(optimizer_update_index),
            "reported_post_update_state_step": report_step,
            "aggregate": self._aggregate(rows),
            "matrices": rows,
        }
        self.records.append(record)
        self._write(status="running")
        return {
            "stage": self.stage,
            "optimizer_update_index": int(optimizer_update_index),
            "reported_post_update_state_step": report_step,
            "all_polar": {
                candidate: record["aggregate"][candidate]["all"]["polar_update"]
                for candidate in self.candidate_order
            },
        }

    def _gate(self) -> dict[str, Any]:
        expected = self.update_indices
        observed = {int(row["optimizer_update_index"]) for row in self.records}
        partial_futility = observed == self.futility_indices
        if observed != expected and not partial_futility:
            return {
                "ready": False,
                "stage": self.stage,
                "selected": None,
                "observed_update_indices": sorted(observed),
            }
        thresholds = self.plan["decision_gate"][
            "requirements_at_every_registered_probe"
        ]
        control_thresholds = self.plan["decision_gate"][
            "fp16_control_requirements"
        ]
        decisions: dict[str, Any] = {}
        passing: list[str] = []
        for candidate in self.candidate_order:
            all_polar = [
                row["aggregate"][candidate]["all"]["polar_update"]
                for row in self.records
            ]
            all_prepolar = [
                row["aggregate"][candidate]["all"]["combined_prepolar"]
                for row in self.records
            ]
            summary = {
                "minimum_all_postpolar_cosine": min(
                    float(item["cosine"]) for item in all_polar
                ),
                "minimum_every_matrix_postpolar_cosine": min(
                    float(item["worst_matrix_cosine"]) for item in all_polar
                ),
                "minimum_all_postpolar_positive_line_energy_recovery": min(
                    float(item["positive_line_energy_recovery"])
                    for item in all_polar
                ),
                "minimum_all_prepolar_cosine": min(
                    float(item["cosine"]) for item in all_prepolar
                ),
            }
            checks = {
                key: summary[key] >= float(value)
                for key, value in thresholds.items()
                if key != "all_metrics_finite"
            }
            checks["all_metrics_finite"] = _all_finite(summary)
            if candidate == "e5m10_fp16_control":
                for key, value in control_thresholds.items():
                    checks[f"control_{key}"] = summary[key] >= float(value)
            passed = all(checks.values())
            decisions[candidate] = {
                "passed": passed,
                "checks": checks,
                "summary": summary,
            }
            if passed:
                passing.append(candidate)
        selected = None
        if passing:
            selected = min(
                passing,
                key=lambda candidate: (
                    int(self.candidates[candidate]["total_bits"]),
                    -float(
                        decisions[candidate]["summary"][
                            "minimum_every_matrix_postpolar_cosine"
                        ]
                    ),
                ),
            )
        if partial_futility:
            compressed_passing = [
                candidate
                for candidate in passing
                if candidate != "e5m10_fp16_control"
            ]
            control_passed = "e5m10_fp16_control" in passing
            if control_passed and not compressed_passing:
                return {
                    "ready": True,
                    "stage": self.stage,
                    "classification": "EARLY_FUTILITY_FAIL",
                    "selected": None,
                    "candidates": decisions,
                    "terminal_replay_required": False,
                    "automatic_endpoint": False,
                }
            return {
                "ready": False,
                "stage": self.stage,
                "classification": (
                    "CONTINUE_TO_TERMINAL_REQUIRED"
                    if compressed_passing
                    else "INVALID_FP16_CONTROL"
                ),
                "selected": (
                    min(
                        compressed_passing,
                        key=lambda candidate: int(
                            self.candidates[candidate]["total_bits"]
                        ),
                    )
                    if compressed_passing
                    else None
                ),
                "candidates": decisions,
                "terminal_replay_required": bool(compressed_passing),
                "automatic_endpoint": False,
            }
        return {
            "ready": True,
            "stage": self.stage,
            "classification": "PASS" if selected else "FAIL",
            "selected": selected,
            "candidates": decisions,
            "automatic_endpoint": False,
        }

    def _write(self, *, status: str) -> None:
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": status,
            "stage": self.stage,
            "plan": {"path": str(self.plan_path), "sha256": self.plan_sha256},
            "run_identity_sha256": self._run_identity_sha256,
            "theoretical_persistent_storage": self.plan[
                "theoretical_persistent_storage"
            ],
            "simulated_state_bytes": {
                candidate: sum(
                    0 if state is None else state.numel() * state.element_size()
                    for state in states.values()
                )
                for candidate, states in self._states.items()
            },
            "codec_seconds": self._codec_seconds,
            "records": self.records,
            "gate": self._gate(),
        }
        if not _all_finite(payload):
            raise RuntimeError("minifloat momentum payload contains non-finite values")
        atomic_json(self.result_path, payload)

    def finalize(self) -> dict[str, Any]:
        self._write(status="finished")
        return {"status": "finished", "stage": self.stage, "gate": self._gate()}
