"""Nonintervening polar-sensitive low-rank momentum capacity oracle.

The production Pair-VQ optimizer remains the only owner.  At registered
gradients this observer asks whether a packed minifloat momentum plus a small
FP16 low-rank correction can reproduce the owner's Muon polar direction.  It
is deliberately a fixed-state capacity oracle, not a causal optimizer.
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
)
from examples.nanogpt.dense_pair_vq_optimizer_transition import (
    direction_metrics,
)
from examples.nanogpt.dense_pair_vq_shadow import atomic_json, sha256_file
from examples.nanogpt.muon import muon_update
from examples.nanogpt.muon_pair_vq import MuonPairVQ, MuonPairVQLinear
from examples.nanogpt.pair_vq_minifloat_momentum import (
    _all_finite,
    round_fp16_mantissa,
)


PLAN_SCHEMA = "mai_124m_pair_vq_polar_sensitive_lowrank_momentum_plan_v1"
RESULT_SCHEMA = "mai_pair_vq_polar_sensitive_lowrank_momentum_result_v1"


def _side(name: str) -> str:
    return "c_fc" if name.endswith(".c_fc") else "c_proj"


def _layer(name: str) -> int:
    return int(name.split(".")[2])


def _seed_for(
    *, optimizer_update_index: int, module_name: str, basis_index: int
) -> int:
    side_index = 0 if _side(module_name) == "c_fc" else 1
    return (
        20261120
        + 1000 * int(optimizer_update_index)
        + 100 * _layer(module_name)
        + 10 * side_index
        + int(basis_index)
    )


def deterministic_svd_lowrank(
    matrix: torch.Tensor,
    *,
    q: int,
    niter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an RNG-isolated randomized SVD in deterministic seed scope."""
    q = min(int(q), min(matrix.shape))
    devices = [matrix.device] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if matrix.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        left, singular, right = torch.svd_lowrank(
            matrix.detach().float(), q=q, niter=int(niter)
        )
    return left, singular, right


def _direction_loss(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    reference = reference.float()
    candidate = candidate.float()
    inner = (reference * candidate).sum()
    denominator = reference.square().sum().sqrt() * candidate.square().sum().sqrt()
    return 1.0 - inner / denominator.clamp_min(1e-30)


def polar_sensitivity(
    *,
    reference_update: torch.Tensor,
    gradient: torch.Tensor,
    base_state: torch.Tensor,
    momentum: float,
    ns_steps: int,
) -> torch.Tensor:
    """Differentiate post-polar angular error with respect to momentum state."""
    with torch.enable_grad():
        perturbation = torch.zeros_like(base_state, requires_grad=True)
        candidate = muon_update(
            gradient + float(momentum) * (base_state + perturbation),
            steps=int(ns_steps),
        )
        loss = _direction_loss(reference_update, candidate)
        (gradient_state,) = torch.autograd.grad(loss, perturbation)
    return -gradient_state.detach().float()


def capacity_objective(
    *,
    reference_state: torch.Tensor,
    reference_update: torch.Tensor,
    gradient: torch.Tensor,
    candidate_state: torch.Tensor,
    momentum: float,
    ns_steps: int,
    state_weight: float,
) -> torch.Tensor:
    candidate_update = muon_update(
        gradient + float(momentum) * candidate_state,
        steps=int(ns_steps),
    ).float()
    polar_error = (candidate_update - reference_update).square().sum()
    polar_error = polar_error / reference_update.square().sum().clamp_min(1e-30)
    state_error = (candidate_state - reference_state).square().sum()
    state_error = state_error / reference_state.square().sum().clamp_min(1e-30)
    return polar_error + float(state_weight) * state_error


def fit_lowrank_core(
    *,
    reference_state: torch.Tensor,
    reference_update: torch.Tensor,
    gradient: torch.Tensor,
    base_state: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    momentum: float,
    ns_steps: int,
    maximum_iterations: int,
    history_size: int,
    tolerance_grad: float,
    tolerance_change: float,
    state_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit only the small basis core; return a detached ambient correction."""
    residual = reference_state - base_state
    initial_core = (left.T @ residual) @ right
    core = torch.nn.Parameter(initial_core.detach().clone())
    optimizer = torch.optim.LBFGS(
        [core],
        lr=1.0,
        max_iter=int(maximum_iterations),
        history_size=int(history_size),
        tolerance_grad=float(tolerance_grad),
        tolerance_change=float(tolerance_change),
        line_search_fn="strong_wolfe",
    )
    closure_count = 0
    started = time.perf_counter()

    def form_candidate(active_core: torch.Tensor) -> torch.Tensor:
        return base_state + (left @ active_core) @ right.T

    with torch.enable_grad():
        initial_objective = capacity_objective(
            reference_state=reference_state,
            reference_update=reference_update,
            gradient=gradient,
            candidate_state=form_candidate(core),
            momentum=momentum,
            ns_steps=ns_steps,
            state_weight=state_weight,
        ).detach()

        def closure() -> torch.Tensor:
            nonlocal closure_count
            closure_count += 1
            optimizer.zero_grad(set_to_none=True)
            objective = capacity_objective(
                reference_state=reference_state,
                reference_update=reference_update,
                gradient=gradient,
                candidate_state=form_candidate(core),
                momentum=momentum,
                ns_steps=ns_steps,
                state_weight=state_weight,
            )
            objective.backward()
            return objective

        optimizer.step(closure)
        fitted_objective = capacity_objective(
            reference_state=reference_state,
            reference_update=reference_update,
            gradient=gradient,
            candidate_state=form_candidate(core),
            momentum=momentum,
            ns_steps=ns_steps,
            state_weight=state_weight,
        ).detach()
    reverted = bool(float(fitted_objective) > float(initial_objective) + 1e-10)
    selected_core = initial_core if reverted else core.detach()
    correction = (left @ selected_core) @ right.T
    elapsed = time.perf_counter() - started
    selected_objective = initial_objective if reverted else fitted_objective
    return correction.detach(), {
        "initial_objective": float(initial_objective),
        "fitted_objective": float(fitted_objective),
        "selected_objective": float(selected_objective),
        "objective_monotonic": bool(
            float(selected_objective) <= float(initial_objective) + 1e-10
        ),
        "fit_reverted": reverted,
        "lbfgs_closure_count": int(closure_count),
        "fit_seconds": float(elapsed),
    }


def postpolar_error_spectrum(
    error: torch.Tensor,
    *,
    ranks: list[int],
    q: int,
    niter: int,
    seed: int,
) -> dict[str, Any]:
    _, singular, _ = deterministic_svd_lowrank(
        error, q=q, niter=niter, seed=seed
    )
    energy = float(error.detach().double().square().sum())
    singular_energy = singular.detach().double().square()
    return {
        "total_error_energy": energy,
        "top_singular_values": [float(value) for value in singular[: max(ranks)]],
        "captured_energy_fraction": {
            str(rank): float(singular_energy[:rank].sum()) / max(energy, 1e-30)
            for rank in ranks
        },
    }


class PairVQPolarLowRankMomentumOracle:
    """Probe fixed-state low-rank capacity beside a production FP16 owner."""

    def __init__(
        self,
        model,
        optimizer,
        *,
        plan_path: Path,
        plan_sha256: str,
        result_path: Path,
        stop_on_gate: bool,
    ) -> None:
        if sha256_file(plan_path) != plan_sha256:
            raise ValueError("polar low-rank momentum plan identity mismatch")
        self.plan_path = plan_path
        self.plan_sha256 = plan_sha256
        self.plan = json.loads(plan_path.read_text())
        if self.plan.get("schema_version") != PLAN_SCHEMA:
            raise ValueError("polar low-rank momentum plan schema mismatch")
        decision = self.plan["decision_gate"]
        for field in (
            "automatic_endpoint",
            "automatic_scale_up",
            "automatic_horizon_transfer",
            "automatic_sweep",
        ):
            if bool(decision[field]):
                raise ValueError("polar low-rank plan cannot authorize follow-up work")
        protocol = self.plan["frozen_protocol"]
        if int(protocol["candidate_parameter_updates"]) != 0:
            raise ValueError("polar low-rank candidates must be nonintervening")
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
            raise ValueError("polar low-rank audit requires FP16 ambient momentum")
        names = {
            id(module): name
            for name, module in model.named_modules()
            if isinstance(module, MuonPairVQLinear)
        }
        self.modules = {
            names[id(module)]: module
            for module in self.owner.modules_by_id.values()
            if id(module) in names
        }
        if len(self.modules) != len(self.owner.modules_by_id):
            raise ValueError("Pair-VQ optimizer module inventory mismatch")
        elements = sum(module.element_count for module in self.modules.values())
        if elements != int(protocol["momentum_elements"]):
            raise ValueError("polar low-rank momentum element inventory mismatch")

        self.result_path = result_path
        self.stop_on_gate = bool(stop_on_gate)
        self.update_indices = {
            int(value) for value in protocol["probe_optimizer_update_indices"]
        }
        self.ranks = [int(value) for value in protocol["ranks"]]
        self.svd = dict(protocol["randomized_svd"])
        self.fit = dict(protocol["core_fit"])
        self.storage = self.plan["theoretical_persistent_storage"]
        self.records: list[dict[str, Any]] = []
        self._run_identity_sha256: str | None = None
        self._total_fit_seconds = 0.0

    @property
    def probe_only(self) -> bool:
        return self.stop_on_gate and bool(self._gate().get("ready"))

    def _owner_group(self, module: MuonPairVQLinear) -> dict[str, Any]:
        matches = [
            group
            for group in self.owner.param_groups
            if any(parameter is module.weight for parameter in group["params"])
        ]
        if len(matches) != 1:
            raise ValueError("expected exactly one Pair-VQ optimizer group")
        return matches[0]

    def _candidate_name(self, base: str, basis: str, rank: int) -> str:
        return f"{base}_{basis}_r{int(rank)}"

    def _candidate_storage(self, base: str, rank: int) -> dict[str, Any]:
        row = self.storage[base][f"rank{int(rank)}"]
        return {
            "base": base,
            "rank": int(rank),
            "factor_bytes": int(row["factor_bytes"]),
            "total_training_bytes": int(row["total_training_bytes"]),
            "training_reduction_factor": float(row["training_reduction_factor"]),
            "transient_expand_flops_per_update": int(
                row["transient_expand_flops_per_update"]
            ),
        }

    def _row(
        self,
        *,
        candidate: str,
        module_name: str,
        reference_state: torch.Tensor,
        reference_combined: torch.Tensor,
        reference_update: torch.Tensor,
        candidate_state: torch.Tensor,
        gradient: torch.Tensor,
        momentum: float,
        ns_steps: int,
        fit: dict[str, Any] | None,
        storage: dict[str, Any] | None,
    ) -> dict[str, Any]:
        combined = gradient + float(momentum) * candidate_state
        update = muon_update(combined, steps=int(ns_steps)).float()
        return {
            "candidate": candidate,
            "module": module_name,
            "side": _side(module_name),
            "layer": _layer(module_name),
            "momentum_state": direction_metrics(reference_state, candidate_state),
            "combined_prepolar": direction_metrics(reference_combined, combined),
            "polar_update": direction_metrics(reference_update, update),
            "fit": fit,
            "storage": storage,
        }

    def _evaluate_base(
        self,
        *,
        base_name: str,
        mantissa_bits: int,
        optimizer_update_index: int,
        include_controls: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        spectra: list[dict[str, Any]] = []
        q = int(self.svd["q"])
        niter = int(self.svd["niter"])
        for module_name in sorted(self.modules):
            module = self.modules[module_name]
            gradient = module.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"missing or non-finite Pair-VQ gradient: {module_name}")
            gradient = gradient.detach().float()
            group = self._owner_group(module)
            momentum = float(group["momentum"])
            ns_steps = int(group["ns_steps"])
            resident = self.owner.state[module.weight].get("ambient_momentum")
            previous = (
                torch.zeros_like(gradient) if resident is None else resident.float()
            )
            reference_state = previous.mul(momentum).add(gradient).to(torch.float16).float()
            reference_combined = gradient + momentum * reference_state
            reference_update = muon_update(
                reference_combined, steps=ns_steps
            ).float()
            base_state = round_fp16_mantissa(
                reference_state, mantissa_bits=int(mantissa_bits)
            ).float()
            base_candidate = f"{base_name}_rank0"
            base_row = self._row(
                candidate=base_candidate,
                module_name=module_name,
                reference_state=reference_state,
                reference_combined=reference_combined,
                reference_update=reference_update,
                candidate_state=base_state,
                gradient=gradient,
                momentum=momentum,
                ns_steps=ns_steps,
                fit={"objective_monotonic": True, "role": "rank_zero_control"},
                storage=None,
            )
            rows.append(base_row)
            spectra.append(
                {
                    "base": base_name,
                    "module": module_name,
                    **postpolar_error_spectrum(
                        reference_update
                        - muon_update(
                            gradient + momentum * base_state, steps=ns_steps
                        ).float(),
                        ranks=self.ranks,
                        q=q,
                        niter=niter,
                        seed=_seed_for(
                            optimizer_update_index=optimizer_update_index,
                            module_name=module_name,
                            basis_index=2,
                        ),
                    ),
                }
            )
            if include_controls:
                rows.append(
                    self._row(
                        candidate="fp16_full_residual_control",
                        module_name=module_name,
                        reference_state=reference_state,
                        reference_combined=reference_combined,
                        reference_update=reference_update,
                        candidate_state=reference_state,
                        gradient=gradient,
                        momentum=momentum,
                        ns_steps=ns_steps,
                        fit={"objective_monotonic": True, "role": "exact_control"},
                        storage=None,
                    )
                )
            residual = reference_state - base_state
            bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            residual_left, _, residual_right = deterministic_svd_lowrank(
                residual,
                q=q,
                niter=niter,
                seed=_seed_for(
                    optimizer_update_index=optimizer_update_index,
                    module_name=module_name,
                    basis_index=0,
                ),
            )
            bases["residual_svd"] = (residual_left, residual_right)
            sensitivity = polar_sensitivity(
                reference_update=reference_update,
                gradient=gradient,
                base_state=base_state,
                momentum=momentum,
                ns_steps=ns_steps,
            )
            polar_left, _, polar_right = deterministic_svd_lowrank(
                sensitivity,
                q=q,
                niter=niter,
                seed=_seed_for(
                    optimizer_update_index=optimizer_update_index,
                    module_name=module_name,
                    basis_index=1,
                ),
            )
            bases["polar_gradient"] = (polar_left, polar_right)
            for basis_name, (left_max, right_max) in bases.items():
                for rank in self.ranks:
                    left = left_max[:, :rank]
                    right = right_max[:, :rank]
                    correction, fit = fit_lowrank_core(
                        reference_state=reference_state,
                        reference_update=reference_update,
                        gradient=gradient,
                        base_state=base_state,
                        left=left,
                        right=right,
                        momentum=momentum,
                        ns_steps=ns_steps,
                        maximum_iterations=int(self.fit["maximum_iterations"]),
                        history_size=int(self.fit["history_size"]),
                        tolerance_grad=float(self.fit["tolerance_grad"]),
                        tolerance_change=float(self.fit["tolerance_change"]),
                        state_weight=0.01,
                    )
                    self._total_fit_seconds += float(fit["fit_seconds"])
                    rows.append(
                        self._row(
                            candidate=self._candidate_name(
                                base_name, basis_name, rank
                            ),
                            module_name=module_name,
                            reference_state=reference_state,
                            reference_combined=reference_combined,
                            reference_update=reference_update,
                            candidate_state=base_state + correction,
                            gradient=gradient,
                            momentum=momentum,
                            ns_steps=ns_steps,
                            fit=fit,
                            storage=self._candidate_storage(base_name, rank),
                        )
                    )
        return rows, spectra

    def _aggregate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for candidate in sorted({row["candidate"] for row in rows}):
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
            payload[candidate]["objective_monotonic"] = all(
                row["fit"] is None
                or bool(row["fit"].get("objective_monotonic", True))
                for row in candidate_rows
            )
            storage_rows = [row["storage"] for row in candidate_rows if row["storage"]]
            payload[candidate]["storage"] = storage_rows[0] if storage_rows else None
        return payload

    def _candidate_passes(self, aggregate: dict[str, Any], candidate: str) -> bool:
        thresholds = self.plan["decision_gate"][
            "requirements_at_every_registered_probe"
        ]
        row = aggregate[candidate]
        post = row["all"]["polar_update"]
        pre = row["all"]["combined_prepolar"]
        state = row["all"]["momentum_state"]
        return bool(
            _all_finite(row)
            and post["cosine"] >= thresholds["minimum_all_postpolar_cosine"]
            and post["worst_matrix_cosine"]
            >= thresholds["minimum_every_matrix_postpolar_cosine"]
            and post["positive_line_energy_recovery"]
            >= thresholds[
                "minimum_all_postpolar_positive_line_energy_recovery"
            ]
            and pre["cosine"] >= thresholds["minimum_all_prepolar_cosine"]
            and state["cosine"]
            >= thresholds["minimum_all_momentum_state_cosine"]
            and row["objective_monotonic"]
        )

    def before_step(
        self, *, optimizer_update_index: int, run_identity_sha256: str
    ) -> dict[str, Any] | None:
        self._run_identity_sha256 = run_identity_sha256
        if int(optimizer_update_index) not in self.update_indices:
            return None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rows, spectra = self._evaluate_base(
            base_name="e5m8",
            mantissa_bits=8,
            optimizer_update_index=int(optimizer_update_index),
            include_controls=True,
        )
        aggregate = self._aggregate(rows)
        stage_a_candidates = [
            name
            for name in aggregate
            if name.startswith("e5m8_") and name != "e5m8_rank0"
        ]
        stage_a_passed = any(
            self._candidate_passes(aggregate, candidate)
            for candidate in stage_a_candidates
        )
        stage_b_evaluated = False
        if stage_a_passed:
            stage_b_rows, stage_b_spectra = self._evaluate_base(
                base_name="e5m6",
                mantissa_bits=6,
                optimizer_update_index=int(optimizer_update_index),
                include_controls=False,
            )
            rows.extend(stage_b_rows)
            spectra.extend(stage_b_spectra)
            aggregate = self._aggregate(rows)
            stage_b_evaluated = True
        elapsed = time.perf_counter() - started
        peak_mib = 0.0
        if torch.cuda.is_available():
            peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        record = {
            "optimizer_update_index": int(optimizer_update_index),
            "reported_post_update_state_step": int(optimizer_update_index) + 1,
            "stage_a_passed": bool(stage_a_passed),
            "stage_b_evaluated": bool(stage_b_evaluated),
            "probe_seconds": float(elapsed),
            "peak_mib": float(peak_mib),
            "aggregate": aggregate,
            "rows": rows,
            "postpolar_error_spectra": spectra,
        }
        self.records.append(record)
        self._write(status="running")
        passing = [
            candidate
            for candidate in aggregate
            if candidate not in {"fp16_full_residual_control", "e5m8_rank0", "e5m6_rank0"}
            and self._candidate_passes(aggregate, candidate)
        ]
        return {
            "optimizer_update_index": int(optimizer_update_index),
            "reported_post_update_state_step": int(optimizer_update_index) + 1,
            "stage_a_passed": bool(stage_a_passed),
            "stage_b_evaluated": bool(stage_b_evaluated),
            "passing_candidates": passing,
            "probe_seconds": float(elapsed),
            "gate": self._gate(),
        }

    def _gate(self) -> dict[str, Any]:
        if not self.records:
            return {"ready": False, "classification": "PENDING", "selected": None}
        controls_pass = all(
            self._candidate_passes(
                record["aggregate"], "fp16_full_residual_control"
            )
            for record in self.records
        )
        if not controls_pass:
            return {
                "ready": True,
                "classification": "INVALID_FP16_CONTROL",
                "selected": None,
            }
        if not self.records[0]["stage_a_passed"]:
            return {
                "ready": True,
                "classification": "EARLY_LOW_RANK_CAPACITY_FAIL",
                "selected": None,
            }
        observed = {int(record["optimizer_update_index"]) for record in self.records}
        if observed != self.update_indices:
            return {
                "ready": False,
                "classification": "CONTINUE_TO_UPDATE_8",
                "selected": None,
                "observed_update_indices": sorted(observed),
            }
        common = set(self.records[0]["aggregate"])
        for record in self.records[1:]:
            common &= set(record["aggregate"])
        excluded = {"fp16_full_residual_control", "e5m8_rank0", "e5m6_rank0"}
        passing = [
            candidate
            for candidate in sorted(common - excluded)
            if all(
                self._candidate_passes(record["aggregate"], candidate)
                for record in self.records
            )
        ]
        selected = None
        if passing:
            selected = min(
                passing,
                key=lambda candidate: (
                    int(self.records[0]["aggregate"][candidate]["storage"]["total_training_bytes"]),
                    -min(
                        float(record["aggregate"][candidate]["all"]["polar_update"]["worst_matrix_cosine"])
                        for record in self.records
                    ),
                ),
            )
        return {
            "ready": True,
            "classification": "PASS" if selected else "FAIL",
            "selected": selected,
            "passing_candidates": passing,
        }

    def _write(self, *, status: str) -> None:
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": status,
            "plan": {"path": str(self.plan_path), "sha256": self.plan_sha256},
            "run_identity_sha256": self._run_identity_sha256,
            "candidate_parameter_updates": 0,
            "total_fit_seconds": float(self._total_fit_seconds),
            "records": self.records,
            "gate": self._gate(),
        }
        if not _all_finite(payload):
            raise RuntimeError("polar low-rank result contains non-finite values")
        atomic_json(self.result_path, payload)

    def finalize(self) -> dict[str, Any]:
        self._write(status="finished")
        return {"status": "finished", "gate": self._gate()}
