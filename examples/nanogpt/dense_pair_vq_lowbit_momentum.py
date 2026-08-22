"""Nonintervening low-bit ambient Muon-momentum oracle for Pair-VQ MLPs.

The dense model is the only training path.  Each candidate stores only int8
simulation codes plus FP16 block scales, decodes that compact state, applies
the causal momentum recurrence, and re-encodes it.  The registered storage
figures use packed two-/four-bit codes; int8 tensors are only a portable
simulation container and are reported separately.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.dense_pair_vq_optimizer_transition import (
    direction_metrics,
)
from examples.nanogpt.dense_pair_vq_shadow import (
    DensePairVQShadowObserver,
    atomic_json,
    sha256_file,
)
from examples.nanogpt.muon import muon_update


PLAN_SCHEMA = "mai_124m_pair_vq_lowbit_momentum_tangent_oracle_plan_v1"
RESULT_SCHEMA = "mai_pair_vq_lowbit_momentum_tangent_oracle_result_v1"


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


@dataclass
class EncodedBlocks:
    codes: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    elements: int
    codec: str
    block_size: int

    @property
    def simulated_bytes(self) -> int:
        return int(
            self.codes.numel() * self.codes.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


@torch.no_grad()
def encode_blocks(
    values: torch.Tensor,
    *,
    codec: str,
    block_size: int,
    ternary_threshold_rms: float,
) -> tuple[EncodedBlocks, torch.Tensor]:
    """Encode one tensor and return its deterministic FP32 decode."""
    flat = values.detach().float().reshape(-1)
    elements = int(flat.numel())
    blocks = math.ceil(elements / int(block_size))
    padding = blocks * int(block_size) - elements
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    shaped = flat.reshape(blocks, int(block_size))
    if codec == "ternary2":
        rms = shaped.square().mean(dim=-1, keepdim=True).sqrt()
        active = shaped.abs() >= float(ternary_threshold_rms) * rms
        codes = (torch.sign(shaped) * active).to(torch.int8)
        scales32 = (
            (shaped.abs() * active).sum(dim=-1, keepdim=True)
            / active.sum(dim=-1, keepdim=True).clamp_min(1)
        )
    elif codec == "symmetric_int4":
        scales32 = (
            shaped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 7.0
        )
        codes = torch.round(shaped / scales32).clamp(-7, 7).to(torch.int8)
    else:
        raise ValueError(f"unsupported low-bit momentum codec: {codec}")
    scales = scales32.to(torch.float16)
    decoded = (codes.float() * scales.float()).reshape(-1)[:elements]
    encoded = EncodedBlocks(
        codes=codes,
        scales=scales,
        shape=tuple(values.shape),
        elements=elements,
        codec=codec,
        block_size=int(block_size),
    )
    return encoded, decoded.reshape_as(values)


@torch.no_grad()
def decode_blocks(state: EncodedBlocks) -> torch.Tensor:
    decoded = (state.codes.float() * state.scales.float()).reshape(-1)
    return decoded[: state.elements].reshape(state.shape)


@torch.no_grad()
def advance_lowbit_momentum(
    previous: tuple[EncodedBlocks, EncodedBlocks | None] | None,
    gradient: torch.Tensor,
    *,
    momentum: float,
    primary_codec: str,
    residual_codec: str | None,
    block_size: int,
    ternary_threshold_rms: float,
) -> tuple[tuple[EncodedBlocks, EncodedBlocks | None], torch.Tensor]:
    """Apply one causal compact-state recurrence without dense persistence."""
    if previous is None:
        previous_decoded = torch.zeros_like(gradient, dtype=torch.float32)
    else:
        previous_decoded = decode_blocks(previous[0])
        if previous[1] is not None:
            previous_decoded.add_(decode_blocks(previous[1]))
    target = previous_decoded.mul(float(momentum)).add(gradient.detach().float())
    primary, primary_decoded = encode_blocks(
        target,
        codec=primary_codec,
        block_size=block_size,
        ternary_threshold_rms=ternary_threshold_rms,
    )
    residual = None
    decoded = primary_decoded
    if residual_codec is not None:
        residual, residual_decoded = encode_blocks(
            target - primary_decoded,
            codec=residual_codec,
            block_size=block_size,
            ternary_threshold_rms=ternary_threshold_rms,
        )
        decoded = primary_decoded + residual_decoded
    return (primary, residual), decoded


def _aggregate_direction(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
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
        "relative_error": math.sqrt(error_energy / max(reference_energy, 1e-30)),
        "cosine": inner / max(math.sqrt(reference_energy * candidate_energy), 1e-30),
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


class PairVQLowBitMomentumOracle:
    """Track compact ambient momentum codecs along one matched dense replay."""

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
        del model
        if sha256_file(plan_path) != plan_sha256:
            raise ValueError("low-bit momentum plan identity mismatch")
        self.plan_path = plan_path
        self.plan_sha256 = plan_sha256
        self.plan = json.loads(plan_path.read_text())
        if self.plan.get("schema_version") != PLAN_SCHEMA:
            raise ValueError("low-bit momentum plan schema mismatch")
        protocol = self.plan["frozen_protocol"]
        if int(protocol["parameter_updates_by_candidates"]) != 0:
            raise ValueError("low-bit momentum oracle must be nonintervening")
        decision = self.plan["decision_gate"]
        if any(
            bool(decision[field])
            for field in (
                "automatic_endpoint",
                "automatic_scale_up",
                "automatic_horizon_transfer",
                "automatic_sweep",
            )
        ):
            raise ValueError("low-bit momentum plan cannot pre-authorize follow-up work")
        self.observer = observer
        self.result_path = result_path
        self.block_size = int(protocol["block_size"])
        self.candidate_order = list(protocol["candidate_order"])
        self.candidates = dict(protocol["candidates"])
        if self.candidate_order != list(self.candidates):
            raise ValueError("candidate order must match frozen candidate mapping")
        self.update_indices = {
            int(step) for step in protocol["optimizer_update_indices"]
        }
        self.reported_steps = dict(
            zip(
                protocol["optimizer_update_indices"],
                protocol["reported_post_update_state_steps"],
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
        elements = sum(
            int(module.weight.numel()) for module in observer._dense_modules.values()
        )
        if elements != int(protocol["momentum_elements"]):
            raise ValueError(
                f"momentum element inventory mismatch: {elements} != "
                f"{protocol['momentum_elements']}"
            )
        self._states: dict[
            str, dict[str, tuple[EncodedBlocks, EncodedBlocks | None] | None]
        ] = {
            candidate: {name: None for name in observer._dense_modules}
            for candidate in self.candidate_order
        }
        self._codec_seconds = {candidate: 0.0 for candidate in self.candidate_order}
        self.records: list[dict[str, Any]] = []
        self._run_identity_sha256: str | None = None

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
        self, *, optimizer_update_index: int, run_identity_sha256: str
    ) -> dict[str, Any] | None:
        self._run_identity_sha256 = run_identity_sha256
        probe = int(optimizer_update_index) in self.update_indices
        rows: list[dict[str, Any]] = []
        cuda_timings: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for name in sorted(self.observer._dense_modules):
            dense = self.observer._dense_modules[name]
            gradient = dense.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"missing or non-finite dense gradient for {name}")
            gradient = gradient.detach().float()
            owner, group = self._dense_owner[name]
            momentum = float(group["momentum"])
            ns_steps = int(group["ns_steps"])
            dense_buffer = owner.state[dense.weight].get("momentum_buffer")
            if dense_buffer is None:
                dense_buffer = torch.zeros_like(dense.weight)
            dense_next = dense_buffer.float() * momentum + gradient
            dense_combined = gradient + momentum * dense_next
            dense_update = None
            if probe:
                dense_update = muon_update(dense_combined, steps=ns_steps).float()
            for candidate in self.candidate_order:
                spec = self.candidates[candidate]
                started = None
                start_event = None
                end_event = None
                if gradient.is_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    started = time.perf_counter()
                state, compact_next = advance_lowbit_momentum(
                    self._states[candidate][name],
                    gradient,
                    momentum=momentum,
                    primary_codec=str(spec["primary_codec"]),
                    residual_codec=(
                        None
                        if spec.get("residual_codec") is None
                        else str(spec["residual_codec"])
                    ),
                    block_size=self.block_size,
                    ternary_threshold_rms=float(
                        spec.get("ternary_threshold_rms", 0.5)
                    ),
                )
                if gradient.is_cuda:
                    assert start_event is not None and end_event is not None
                    end_event.record()
                    cuda_timings.append((candidate, start_event, end_event))
                else:
                    assert started is not None
                    self._codec_seconds[candidate] += time.perf_counter() - started
                self._states[candidate][name] = state
                if not probe:
                    continue
                compact_combined = gradient + momentum * compact_next
                compact_update = muon_update(
                    compact_combined, steps=ns_steps
                ).float()
                rows.append(
                    {
                        "candidate": candidate,
                        "module": name,
                        "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                        "layer": int(name.split(".")[2]),
                        "momentum_state": direction_metrics(
                            dense_next, compact_next
                        ),
                        "combined_prepolar": direction_metrics(
                            dense_combined, compact_combined
                        ),
                        "polar_update": direction_metrics(
                            dense_update, compact_update
                        ),
                    }
                )
        if cuda_timings:
            torch.cuda.synchronize()
            for candidate, start_event, end_event in cuda_timings:
                self._codec_seconds[candidate] += (
                    float(start_event.elapsed_time(end_event)) / 1000.0
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
            "all_polar": {
                candidate: record["aggregate"][candidate]["all"]["polar_update"]
                for candidate in self.candidate_order
            },
        }

    def _simulated_bytes(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for candidate in self.candidate_order:
            total = 0
            for state in self._states[candidate].values():
                if state is None:
                    continue
                total += state[0].simulated_bytes
                if state[1] is not None:
                    total += state[1].simulated_bytes
            result[candidate] = total
        return result

    def _gate(self) -> dict[str, Any]:
        late_steps = {
            int(step)
            for step in self.plan["frozen_protocol"][
                "primary_late_post_update_state_steps"
            ]
        }
        late = [
            row
            for row in self.records
            if int(row["reported_post_update_state_step"]) in late_steps
        ]
        if len(late) != len(late_steps):
            return {"ready": False, "selected": None, "candidates": {}}
        thresholds = self.plan["decision_gate"][
            "late_probe_requirements_for_one_candidate"
        ]
        storage = self.plan["theoretical_persistent_storage"]
        decisions: dict[str, Any] = {}
        passing: list[str] = []
        for candidate in self.candidate_order:
            all_polar = [
                row["aggregate"][candidate]["all"]["polar_update"] for row in late
            ]
            all_prepolar = [
                row["aggregate"][candidate]["all"]["combined_prepolar"]
                for row in late
            ]
            side_polar = [
                row["aggregate"][candidate][side]["polar_update"]
                for row in late
                for side in ("c_fc", "c_proj")
            ]
            summary = {
                "minimum_all_polar_cosine": min(float(x["cosine"]) for x in all_polar),
                "minimum_all_polar_positive_line_energy_recovery": min(
                    float(x["positive_line_energy_recovery"]) for x in all_polar
                ),
                "minimum_side_polar_cosine": min(float(x["cosine"]) for x in side_polar),
                "minimum_worst_matrix_polar_cosine": min(
                    float(x["worst_matrix_cosine"]) for x in all_polar
                ),
                "minimum_all_prepolar_cosine": min(
                    float(x["cosine"]) for x in all_prepolar
                ),
                "fraction_of_dense_fp32_momentum_bytes": float(
                    storage[candidate]["fraction_of_dense_fp32"]
                ),
            }
            checks = {
                field: summary[field] >= float(threshold)
                for field, threshold in thresholds.items()
                if field
                not in (
                    "all_metrics_finite",
                    "maximum_fraction_of_dense_fp32_momentum_bytes",
                )
            }
            checks["maximum_fraction_of_dense_fp32_momentum_bytes"] = (
                summary["fraction_of_dense_fp32_momentum_bytes"]
                <= float(
                    thresholds[
                        "maximum_fraction_of_dense_fp32_momentum_bytes"
                    ]
                )
            )
            checks["all_metrics_finite"] = _all_finite(summary)
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
                key=lambda name: (
                    int(storage[name]["total_bytes"]),
                    -float(
                        decisions[name]["summary"]["minimum_all_polar_cosine"]
                    ),
                ),
            )
        return {
            "ready": True,
            "classification": "PASS" if selected is not None else "FAIL",
            "selected": selected,
            "candidates": decisions,
            "automatic_endpoint": False,
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
            "theoretical_persistent_storage": self.plan[
                "theoretical_persistent_storage"
            ],
            "actual_int8_simulation_bytes": self._simulated_bytes(),
            "codec_seconds": self._codec_seconds,
            "records": self.records,
            "gate": self._gate(),
        }
        if not _all_finite(payload):
            raise RuntimeError("low-bit momentum payload contains non-finite values")
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
