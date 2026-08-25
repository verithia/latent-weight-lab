"""Held-out functional-gradient oracle for forward-visible Pair-VQ carry.

The dense training path is never modified.  It supplies a synchronized
reference matrix and reference task gradient.  Candidate state is restricted
to the production Pair-VQ projection plus its existing compact feedback
codec.  The oracle attributes the effect of a one-pass virtual center and an
exactly centered two-pass antithetic realization.
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from examples.nanogpt.muon import muon_update
from examples.nanogpt.dense_pair_vq_shadow import (
    DensePairVQShadowObserver,
    atomic_json,
    sha256_file,
)
from examples.nanogpt.pair_vq_tangent_atlas import (
    AtlasProtocol,
    direction_metrics as atlas_direction_metrics,
    evaluate_matrix as evaluate_atlas_matrix,
)


PLAN_SCHEMA = "mai_124m_pair_vq_antithetic_functional_gradient_oracle_plan_v1"
POLAR_PLAN_SCHEMA = "mai_124m_pair_vq_same_momentum_polar_amplification_oracle_plan_v1"
REGULARIZED_POLAR_PLAN_SCHEMA = (
    "mai_124m_pair_vq_early_stopped_muon_polar_stability_oracle_plan_v1"
)
CODEC_STABILITY_PLAN_SCHEMA = (
    "mai_124m_pair_vq_codec_neighbor_path_stability_oracle_plan_v1"
)
SPECTRAL_DAMPING_PLAN_SCHEMA = (
    "mai_124m_pair_vq_norm_preserving_spectral_damping_oracle_plan_v1"
)
TANGENT_ATLAS_PLAN_SCHEMA = (
    "mai_124m_pair_vq_full_rank_tangent_atlas_oracle_plan_v1"
)
RESULT_SCHEMA = "mai_pair_vq_antithetic_functional_gradient_oracle_result_v1"
POLAR_RESULT_SCHEMA = "mai_pair_vq_same_momentum_polar_amplification_oracle_result_v1"
REGULARIZED_POLAR_RESULT_SCHEMA = (
    "mai_pair_vq_early_stopped_muon_polar_stability_oracle_result_v1"
)
CODEC_STABILITY_RESULT_SCHEMA = (
    "mai_pair_vq_codec_neighbor_path_stability_oracle_result_v1"
)
SPECTRAL_DAMPING_RESULT_SCHEMA = (
    "mai_pair_vq_norm_preserving_spectral_damping_oracle_result_v1"
)
TANGENT_ATLAS_RESULT_SCHEMA = (
    "mai_pair_vq_full_rank_tangent_atlas_oracle_result_v1"
)


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


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.double()
    right64 = right.double()
    denominator = (left64.square().sum() * right64.square().sum()).sqrt()
    return float((left64 * right64).sum() / denominator.clamp_min(1e-30))


def gradient_comparison(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ValueError("gradient maps do not contain identical matrices")
    rows = []
    total_reference = 0.0
    total_candidate = 0.0
    total_error = 0.0
    total_inner = 0.0
    for name in sorted(reference):
        dense = reference[name].double()
        proposed = candidate[name].double()
        reference_energy = float(dense.square().sum())
        candidate_energy = float(proposed.square().sum())
        error_energy = float((proposed - dense).square().sum())
        inner = float((proposed * dense).sum())
        cosine = inner / max(
            math.sqrt(reference_energy * candidate_energy), 1e-30
        )
        rows.append(
            {
                "module": name,
                "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                "reference_energy": reference_energy,
                "candidate_energy": candidate_energy,
                "error_energy": error_energy,
                "relative_error": math.sqrt(
                    error_energy / max(reference_energy, 1e-30)
                ),
                "cosine": cosine,
                "positive_line_recovery": max(cosine, 0.0) ** 2,
            }
        )
        total_reference += reference_energy
        total_candidate += candidate_energy
        total_error += error_energy
        total_inner += inner
    aggregate_cosine = total_inner / max(
        math.sqrt(total_reference * total_candidate), 1e-30
    )
    return {
        "aggregate": {
            "reference_energy": total_reference,
            "candidate_energy": total_candidate,
            "error_energy": total_error,
            "relative_error": math.sqrt(
                total_error / max(total_reference, 1e-30)
            ),
            "cosine": aggregate_cosine,
            "positive_line_recovery": max(aggregate_cosine, 0.0) ** 2,
        },
        "matrices": rows,
    }


def gradient_cross_cosine(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> float:
    if set(left) != set(right):
        raise ValueError("gradient maps do not contain identical matrices")
    inner = sum(
        float((left[name].double() * right[name].double()).sum())
        for name in left
    )
    left_energy = sum(
        float(value.double().square().sum()) for value in left.values()
    )
    right_energy = sum(
        float(value.double().square().sum()) for value in right.values()
    )
    return inner / max(math.sqrt(left_energy * right_energy), 1e-30)


def antithetic_average(
    minus: dict[str, torch.Tensor], plus: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(minus) != set(plus):
        raise ValueError("antithetic gradient maps do not match")
    return {name: (minus[name] + plus[name]) * 0.5 for name in minus}


def _direction_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    reference64 = reference.detach().double()
    candidate64 = candidate.detach().double()
    reference_energy = float(reference64.square().sum())
    candidate_energy = float(candidate64.square().sum())
    error_energy = float((candidate64 - reference64).square().sum())
    inner = float((reference64 * candidate64).sum())
    return {
        "reference_energy": reference_energy,
        "candidate_energy": candidate_energy,
        "error_energy": error_energy,
        "inner": inner,
        "relative_error": math.sqrt(
            error_energy / max(reference_energy, 1e-30)
        ),
        "cosine": inner
        / max(math.sqrt(reference_energy * candidate_energy), 1e-30),
    }


def _aggregate_directions(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float]:
    reference_energy = sum(float(row[key]["reference_energy"]) for row in rows)
    candidate_energy = sum(float(row[key]["candidate_energy"]) for row in rows)
    error_energy = sum(float(row[key]["error_energy"]) for row in rows)
    inner = sum(float(row[key]["inner"]) for row in rows)
    return {
        "reference_energy": reference_energy,
        "candidate_energy": candidate_energy,
        "error_energy": error_energy,
        "relative_error": math.sqrt(
            error_energy / max(reference_energy, 1e-30)
        ),
        "cosine": inner
        / max(math.sqrt(reference_energy * candidate_energy), 1e-30),
        "worst_matrix_relative_error": max(
            float(row[key]["relative_error"]) for row in rows
        ),
        "worst_matrix_cosine": min(float(row[key]["cosine"]) for row in rows),
    }


class PairVQFunctionalGradientOracle:
    """Score compact functional gradients without changing dense training."""

    def __init__(
        self,
        model: nn.Module,
        observer: DensePairVQShadowObserver,
        *,
        optimizer: Any | None = None,
        plan_path: Path,
        plan_sha256: str,
        result_path: Path,
        data_dir: Path,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        if sha256_file(plan_path) != plan_sha256:
            raise ValueError("functional-oracle plan identity mismatch")
        plan = json.loads(plan_path.read_text())
        if plan.get("schema_version") not in {
            PLAN_SCHEMA,
            POLAR_PLAN_SCHEMA,
            REGULARIZED_POLAR_PLAN_SCHEMA,
            CODEC_STABILITY_PLAN_SCHEMA,
            SPECTRAL_DAMPING_PLAN_SCHEMA,
            TANGENT_ATLAS_PLAN_SCHEMA,
        }:
            raise ValueError("unexpected functional-oracle plan schema")
        self.polar_amplification_enabled = (
            plan.get("schema_version")
            in {
                POLAR_PLAN_SCHEMA,
                REGULARIZED_POLAR_PLAN_SCHEMA,
                CODEC_STABILITY_PLAN_SCHEMA,
                SPECTRAL_DAMPING_PLAN_SCHEMA,
            }
        )
        self.regularized_polar_enabled = (
            plan.get("schema_version") == REGULARIZED_POLAR_PLAN_SCHEMA
        )
        self.codec_stability_enabled = (
            plan.get("schema_version") == CODEC_STABILITY_PLAN_SCHEMA
        )
        self.spectral_damping_enabled = (
            plan.get("schema_version") == SPECTRAL_DAMPING_PLAN_SCHEMA
        )
        self.tangent_atlas_enabled = (
            plan.get("schema_version") == TANGENT_ATLAS_PLAN_SCHEMA
        )
        if (self.polar_amplification_enabled or self.tangent_atlas_enabled) and optimizer is None:
            raise ValueError(
                "same-momentum request analysis requires the dense optimizer"
            )
        frozen = plan["frozen_protocol"]
        if str(observer.source_config_path) != str(
            frozen["compact_source_config"]
        ):
            raise ValueError("functional oracle compact source changed")
        self.model = model
        self.observer = observer
        self.plan = plan
        self.plan_path = plan_path
        self.plan_sha256 = plan_sha256
        self.result_path = result_path
        self.device = device
        self.dtype = dtype
        self.probe_steps = {int(step) for step in frozen["probe_steps"]}
        self.primary_late_steps = {
            int(step) for step in frozen["primary_late_steps"]
        }
        self.records: list[dict[str, Any]] = []
        self._active_probe_step = 0
        self._isotropic_seed_base = int(
            frozen.get("isotropic_seed_base", 0)
        )
        self._dense_optimizer_owner: dict[str, tuple[Any, dict[str, Any]]] = {}
        if self.polar_amplification_enabled or self.tangent_atlas_enabled:
            children = list(getattr(optimizer, "optimizers", [optimizer]))
            for name, module in observer._dense_modules.items():
                matches = []
                for child in children:
                    for group in child.param_groups:
                        if any(
                            parameter is module.weight
                            for parameter in group["params"]
                        ):
                            matches.append((child, group))
                if len(matches) != 1:
                    raise ValueError(
                        f"expected one dense optimizer owner for {name}, "
                        f"found {len(matches)}"
                    )
                self._dense_optimizer_owner[name] = matches[0]
        self._feedback_state: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in observer._shadow_modules.items():
            if not module.error_feedback:
                raise ValueError(
                    f"functional oracle requires compact feedback for {name}"
                )
            state = {
                "levels": torch.zeros(
                    module.feedback_level_shape,
                    device=device,
                    dtype=torch.float32,
                ),
                "codes": torch.zeros(
                    module.feedback_code_shape,
                    device=device,
                    dtype=torch.uint8,
                ),
            }
            if module.feedback_center_shape is not None:
                state["center"] = torch.zeros(
                    module.feedback_center_shape,
                    device=device,
                    dtype=torch.float32,
                )
            self._feedback_state[name] = state

        self._windows: dict[str, list[torch.Tensor]] = {}
        self._window_indices: dict[str, set[int]] = {}
        for split in ("fit_window", "heldout_window"):
            batches, indices = self._make_window(
                data_dir=data_dir, protocol=frozen[split]
            )
            key = "fit" if split == "fit_window" else "heldout"
            self._windows[key] = batches
            self._window_indices[key] = indices
        overlap = self._window_indices["fit"] & self._window_indices["heldout"]
        if overlap:
            raise ValueError("functional-oracle fit and held-out windows overlap")
        self.indices_disjoint = True
        self.last_residual_metrics = self.update_compact_residual()

        self._atlas_protocols: dict[str, AtlasProtocol] = {}
        if self.tangent_atlas_enabled:
            atlas = plan["atlas"]
            for atoms in (1, 2):
                self._atlas_protocols[f"S{atoms}"] = AtlasProtocol(
                    width=int(atlas["width"]),
                    stages=int(atlas["stages"]),
                    atoms=atoms,
                    fit_steps=int(atlas["fit_steps"]),
                    fit_learning_rate=float(atlas["fit_learning_rate"]),
                    fit_weight_decay=float(atlas["fit_weight_decay"]),
                    fit_gradient_clip=float(atlas["fit_gradient_clip"]),
                    cg_iterations=int(atlas["cg_iterations"]),
                    cg_tolerance=float(atlas["cg_tolerance"]),
                    cg_ridge=float(atlas["cg_ridge"]),
                    seed=int(atlas["seed"]),
                )

    @staticmethod
    def _make_window(
        *, data_dir: Path, protocol: dict[str, Any]
    ) -> tuple[list[torch.Tensor], set[int]]:
        block_size = int(protocol["block_size"])
        values = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(protocol["seed"]))
        indices = torch.randint(
            len(values) - block_size - 1,
            (int(protocol["batches"]), int(protocol["batch_size"])),
            generator=generator,
        )
        batches = []
        for row in indices:
            batch = np.stack(
                [
                    np.array(
                        values[int(index) : int(index) + block_size + 1],
                        dtype=np.int64,
                        copy=True,
                    )
                    for index in row
                ]
            )
            batches.append(torch.from_numpy(batch))
        return batches, {int(value) for value in indices.flatten()}

    @property
    def persistent_training_bytes(self) -> int:
        return sum(
            module.persistent_codec_bytes
            + module.compact_momentum_bytes
            + module.compact_feedback_bytes
            for module in self.observer._shadow_modules.values()
        )

    def _decode_residual(self, name: str) -> torch.Tensor:
        module = self.observer._shadow_modules[name]
        state = self._feedback_state[name]
        return module.decode_feedback(
            state["levels"], state["codes"], state.get("center")
        ).reshape_as(module.weight)

    @torch.no_grad()
    def update_compact_residual(self) -> dict[str, Any]:
        rows = []
        for name, module in self.observer._shadow_modules.items():
            dense = self.observer._dense_modules[name].weight.detach().float()
            residual = dense - module.weight.detach().float()
            state = self._feedback_state[name]
            changes = module.fit_feedback_(
                residual.reshape(-1, module.vector_length),
                state["levels"],
                state["codes"],
                state.get("center"),
            )
            decoded = self._decode_residual(name)
            virtual = module.weight.detach().float() + decoded
            error = float((virtual - dense).square().sum())
            dense_energy = float(dense.square().sum())
            residual_energy = float(residual.square().sum())
            residual_error = float((decoded - residual).square().sum())
            rows.append(
                {
                    "module": name,
                    "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                    "feedback_code_changes": changes,
                    "dense_weight_energy": dense_energy,
                    "virtual_error_energy": error,
                    "virtual_weight_energy_recovery": 1.0
                    - error / max(dense_energy, 1e-30),
                    "residual_energy": residual_energy,
                    "residual_codec_energy_recovery": 1.0
                    - residual_error / max(residual_energy, 1e-30),
                }
            )
        total_error = sum(float(row["virtual_error_energy"]) for row in rows)
        total_energy = sum(float(row["dense_weight_energy"]) for row in rows)
        return {
            "weighted_virtual_weight_energy_recovery": 1.0
            - total_error / max(total_energy, 1e-30),
            "worst_matrix_virtual_weight_energy_recovery": min(
                float(row["virtual_weight_energy_recovery"]) for row in rows
            ),
            "feedback_code_changes": sum(
                int(row["feedback_code_changes"]) for row in rows
            ),
            "matrices": rows,
        }

    def _isotropic_delta(
        self, name: str, residual: torch.Tensor
    ) -> torch.Tensor:
        """Return a deterministic matched-energy Rademacher control."""
        layer = int(name.split(".")[2])
        side = 0 if name.endswith(".c_fc") else 1
        seed = (
            self._isotropic_seed_base
            + 1000 * int(self._active_probe_step)
            + 2 * layer
            + side
        )
        generator = torch.Generator(device=residual.device)
        generator.manual_seed(seed)
        signs = torch.randint(
            0,
            2,
            residual.shape,
            generator=generator,
            device=residual.device,
            dtype=torch.int8,
        ).float()
        signs.mul_(2.0).sub_(1.0)
        signs.mul_(
            residual.float().norm()
            / math.sqrt(max(int(residual.numel()), 1))
        )
        return signs

    @contextlib.contextmanager
    def _installed_variant(self, variant: str) -> Iterator[None]:
        if variant not in {
            "dense",
            "native",
            "center",
            "plus",
            "isotropic_minus",
            "isotropic_plus",
        }:
            raise ValueError(f"unknown functional-oracle variant: {variant}")
        backups = {
            name: module.weight.detach().clone()
            for name, module in self.observer._dense_modules.items()
        }
        with torch.no_grad():
            if variant != "dense":
                for name, dense in self.observer._dense_modules.items():
                    compact = self.observer._shadow_modules[name].weight
                    residual = self._decode_residual(name)
                    center = compact + residual
                    if variant in {"native", "center", "plus"}:
                        multiplier = {
                            "native": 0.0,
                            "center": 1.0,
                            "plus": 2.0,
                        }[variant]
                        dense.weight.copy_(compact + multiplier * residual)
                    else:
                        delta = self._isotropic_delta(name, residual)
                        sign = -1.0 if variant == "isotropic_minus" else 1.0
                        dense.weight.copy_(center + sign * delta)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, module in self.observer._dense_modules.items():
                    module.weight.copy_(backups[name])

    def _capture_gradients(
        self, *, split: str, variant: str
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        losses = []
        self.model.zero_grad(set_to_none=True)
        with self._installed_variant(variant):
            for tokens in self._windows[split]:
                batch = tokens.to(self.device, non_blocking=False)
                inputs = batch[:, :-1].contiguous()
                targets = batch[:, 1:].contiguous()
                device_type = "cuda" if "cuda" in self.device else "cpu"
                amp = (
                    torch.amp.autocast(device_type=device_type, dtype=self.dtype)
                    if device_type == "cuda"
                    else contextlib.nullcontext()
                )
                with amp:
                    _logits, loss = self.model(inputs, targets)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("non-finite functional-oracle loss")
                (loss / len(self._windows[split])).backward()
                losses.append(float(loss.detach()))
            gradients = {}
            for name, module in self.observer._dense_modules.items():
                gradient = module.weight.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError(
                        f"missing or non-finite functional gradient for {name}"
                    )
                gradients[name] = gradient.detach().float().cpu().clone()
        self.model.zero_grad(set_to_none=True)
        return gradients, losses

    @torch.no_grad()
    def _capture_logit_stability(self, *, split: str) -> dict[str, Any]:
        """Measure center-to-neighbor last-token KL on one frozen window."""
        neighbors = (
            "native",
            "plus",
            "isotropic_minus",
            "isotropic_plus",
        )
        totals = {variant: 0.0 for variant in neighbors}
        batches = self._windows[split]
        device_type = "cuda" if "cuda" in self.device else "cpu"
        for tokens in batches:
            inputs = tokens[:, :-1].contiguous().to(
                self.device, non_blocking=False
            )
            amp = (
                torch.amp.autocast(device_type=device_type, dtype=self.dtype)
                if device_type == "cuda"
                else contextlib.nullcontext()
            )
            with self._installed_variant("center"):
                with amp:
                    center_logits, _loss = self.model(inputs)
                center_log_probs = F.log_softmax(
                    center_logits[:, -1, :].float(), dim=-1
                )
                center_probs = center_log_probs.exp()
            for variant in neighbors:
                with self._installed_variant(variant):
                    with amp:
                        neighbor_logits, _loss = self.model(inputs)
                    neighbor_log_probs = F.log_softmax(
                        neighbor_logits[:, -1, :].float(), dim=-1
                    )
                divergence = F.kl_div(
                    neighbor_log_probs,
                    center_probs,
                    reduction="batchmean",
                )
                if not torch.isfinite(divergence):
                    raise RuntimeError("non-finite codec-neighbor logit KL")
                totals[variant] += float(divergence)
        means = {
            variant: value / max(len(batches), 1)
            for variant, value in totals.items()
        }
        actual = 0.5 * (means["native"] + means["plus"])
        isotropic = 0.5 * (
            means["isotropic_minus"] + means["isotropic_plus"]
        )
        return {
            "center_to_neighbor_kl": means,
            "actual_symmetric_mean": actual,
            "isotropic_symmetric_mean": isotropic,
            "actual_to_isotropic_ratio": actual / max(isotropic, 1e-30),
        }

    @torch.no_grad()
    def _neighbor_energy_metrics(self) -> dict[str, Any]:
        rows = []
        for name in sorted(self.observer._dense_modules):
            residual = self._decode_residual(name).float()
            isotropic = self._isotropic_delta(name, residual)
            actual_energy = float(residual.double().square().sum())
            isotropic_energy = float(isotropic.double().square().sum())
            mismatch = abs(isotropic_energy - actual_energy) / max(
                actual_energy, 1e-30
            )
            rows.append(
                {
                    "module": name,
                    "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                    "actual_energy": actual_energy,
                    "isotropic_energy": isotropic_energy,
                    "relative_energy_mismatch": mismatch,
                }
            )
        return {
            "maximum_relative_energy_mismatch": max(
                float(row["relative_energy_mismatch"]) for row in rows
            ),
            "matrices": rows,
        }

    @staticmethod
    def _actual_worse_matrix_fraction(
        actual: dict[str, Any], isotropic: dict[str, Any]
    ) -> float:
        actual_rows = {row["module"]: row for row in actual["matrices"]}
        isotropic_rows = {
            row["module"]: row for row in isotropic["matrices"]
        }
        if set(actual_rows) != set(isotropic_rows):
            raise ValueError("codec-neighbor matrix inventories do not match")
        return sum(
            float(actual_rows[name]["relative_error"])
            > float(isotropic_rows[name]["relative_error"])
            for name in actual_rows
        ) / max(len(actual_rows), 1)

    def _weight_center_metrics(self) -> dict[str, Any]:
        return self.last_residual_metrics

    @staticmethod
    def _norm_preserving_spectral_damping_frontier(
        request: torch.Tensor,
        *,
        relative_rms_ridges: tuple[float, ...],
        native_steps: int,
    ) -> dict[str, torch.Tensor]:
        """Apply scale-free ridge damping to the native NS spectral response.

        Exact SVD is intentionally confined to this nonintervening oracle. A
        selected filter still requires a separate fast-approximation gate.
        """
        if native_steps != 5:
            raise ValueError("spectral damping oracle requires native NS5")
        if not relative_rms_ridges or any(
            not math.isfinite(value) or value <= 0.0
            for value in relative_rms_ridges
        ):
            raise ValueError("relative RMS ridge values must be positive")
        original_shape = request.shape
        matrix = request.float().reshape(request.shape[0], -1)
        transposed = matrix.shape[0] > matrix.shape[1]
        if transposed:
            matrix = matrix.T
        normalized = matrix / (matrix.norm() + 1e-7)
        left, singular, right_t = torch.linalg.svd(
            normalized, full_matrices=False
        )
        response = singular
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(native_steps):
            response = a * response + b * response.pow(3) + c * response.pow(5)
        rms = singular.square().mean().sqrt()
        native = muon_update(request, steps=native_steps).float()
        native_norm = native.norm()
        candidates: dict[str, torch.Tensor] = {}
        for ridge in relative_rms_ridges:
            tau = float(ridge) * rms
            damping = singular / (singular.square() + tau.square()).sqrt()
            candidate = (left * (response * damping).unsqueeze(0)) @ right_t
            if transposed:
                candidate = candidate.T
            candidate = candidate.reshape(original_shape)
            candidate.mul_(native_norm / candidate.norm().clamp_min(1e-30))
            candidates[f"{ridge:.8g}"] = candidate
        return candidates

    @torch.no_grad()
    def _spectral_damping_comparison(
        self,
        reference: dict[str, torch.Tensor],
        candidate: dict[str, torch.Tensor],
        native_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        frontier = self.plan["spectral_damping_frontier"]
        ridges = tuple(float(value) for value in frontier["relative_rms_ridge"])
        native_steps = int(frontier["native_ns_steps"])
        if native_steps != 5 or ridges != (0.03125, 0.0625, 0.125, 0.25):
            raise ValueError("spectral damping frontier changed")
        native_by_name = {str(row["module"]): row for row in native_rows}
        per_ridge: dict[str, list[dict[str, Any]]] = {
            f"{ridge:.8g}": [] for ridge in ridges
        }
        for name in sorted(reference):
            module = self.observer._dense_modules[name]
            owner, group = self._dense_optimizer_owner[name]
            momentum = float(group["momentum"])
            state = owner.state[module.weight]
            buffer = state.get("momentum_buffer")
            if buffer is None:
                buffer = torch.zeros_like(module.weight)
            dense_gradient = reference[name].to(
                device=module.weight.device, dtype=torch.float32
            )
            compact_gradient = candidate[name].to(
                device=module.weight.device, dtype=torch.float32
            )
            dense_request = dense_gradient + momentum * (
                buffer.float() * momentum + dense_gradient
            )
            compact_request = compact_gradient + momentum * (
                buffer.float() * momentum + compact_gradient
            )
            native_dense = muon_update(dense_request, steps=native_steps).float()
            native_task = _direction_metrics(dense_gradient, native_dense)
            dense_frontier = self._norm_preserving_spectral_damping_frontier(
                dense_request,
                relative_rms_ridges=ridges,
                native_steps=native_steps,
            )
            compact_frontier = self._norm_preserving_spectral_damping_frontier(
                compact_request,
                relative_rms_ridges=ridges,
                native_steps=native_steps,
            )
            for key in per_ridge:
                dense_update = dense_frontier[key]
                compact_update = compact_frontier[key]
                candidate_polar = _direction_metrics(dense_update, compact_update)
                dense_native = _direction_metrics(native_dense, dense_update)
                dense_task = _direction_metrics(dense_gradient, dense_update)
                native_row = native_by_name[name]
                per_ridge[key].append(
                    {
                        "module": name,
                        "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                        "candidate_polar": candidate_polar,
                        "dense_native": dense_native,
                        "dense_task": dense_task,
                        "native_dense_task": native_task,
                        "dense_task_alignment_retention": (
                            dense_task["cosine"]
                            / max(native_task["cosine"], 1e-30)
                        ),
                        "relative_error_closure_vs_native": (
                            1.0
                            - candidate_polar["relative_error"]
                            / max(
                                float(native_row["polar"]["relative_error"]),
                                1e-30,
                            )
                        ),
                    }
                )
            del dense_frontier, compact_frontier
            del dense_gradient, compact_gradient, dense_request, compact_request
            del native_dense

        payload: dict[str, Any] = {}
        for key, rows in per_ridge.items():
            payload[key] = {}
            for label, selected in (
                ("all", rows),
                ("c_fc", [row for row in rows if row["side"] == "c_fc"]),
                ("c_proj", [row for row in rows if row["side"] == "c_proj"]),
            ):
                selected_names = {str(row["module"]) for row in selected}
                selected_native = [
                    row
                    for row in native_rows
                    if str(row["module"]) in selected_names
                ]
                candidate_polar = _aggregate_directions(
                    selected, "candidate_polar"
                )
                dense_native = _aggregate_directions(selected, "dense_native")
                dense_task = _aggregate_directions(selected, "dense_task")
                matrix_closures = [
                    float(row["relative_error_closure_vs_native"])
                    for row in selected
                ]
                payload[key][label] = {
                    "candidate_polar": candidate_polar,
                    "dense_native": dense_native,
                    "dense_native_norm_ratio": math.sqrt(
                        dense_native["candidate_energy"]
                        / max(dense_native["reference_energy"], 1e-30)
                    ),
                    "minimum_matrix_dense_native_cosine": min(
                        float(row["dense_native"]["cosine"])
                        for row in selected
                    ),
                    "dense_task": dense_task,
                    "minimum_matrix_dense_task_alignment_retention": min(
                        float(row["dense_task_alignment_retention"])
                        for row in selected
                    ),
                    "relative_error_amplification": (
                        candidate_polar["relative_error"]
                        / max(
                            _aggregate_directions(selected_native, "prepolar")[
                                "relative_error"
                            ],
                            1e-30,
                        )
                    ),
                    "relative_error_closure_vs_native": (
                        1.0
                        - candidate_polar["relative_error"]
                        / max(
                            _aggregate_directions(selected_native, "polar")[
                                "relative_error"
                            ],
                            1e-30,
                        )
                    ),
                    "minimum_matrix_relative_error_closure_vs_native": min(
                        matrix_closures
                    ),
                    "matrix_regression_fraction": sum(
                        closure < 0.0 for closure in matrix_closures
                    )
                    / max(len(matrix_closures), 1),
                }
        return payload

    @torch.no_grad()
    def _same_momentum_polar_comparison(
        self,
        reference: dict[str, torch.Tensor],
        candidate: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Compare dense/compact gradients under one identical momentum history.

        Both branches start from the live dense Muon buffer.  Only the task
        gradient differs, so any extra error after ``muon_update`` is caused by
        polar normalization of the compact functional-gradient perturbation,
        not compact momentum transport or retraction.
        """
        if set(reference) != set(candidate):
            raise ValueError("polar comparison gradient maps do not match")
        rows = []
        for name in sorted(reference):
            module = self.observer._dense_modules[name]
            owner, group = self._dense_optimizer_owner[name]
            momentum = float(group["momentum"])
            ns_steps = int(group["ns_steps"])
            state = owner.state[module.weight]
            buffer = state.get("momentum_buffer")
            if buffer is None:
                buffer = torch.zeros_like(module.weight)
            dense_gradient = reference[name].to(
                device=module.weight.device, dtype=torch.float32
            )
            compact_gradient = candidate[name].to(
                device=module.weight.device, dtype=torch.float32
            )
            dense_next = buffer.float() * momentum + dense_gradient
            compact_next = buffer.float() * momentum + compact_gradient
            dense_request = dense_gradient + momentum * dense_next
            compact_request = compact_gradient + momentum * compact_next
            dense_update = muon_update(dense_request, steps=ns_steps).float()
            compact_update = muon_update(compact_request, steps=ns_steps).float()
            prepolar = _direction_metrics(dense_request, compact_request)
            polar = _direction_metrics(dense_update, compact_update)
            rows.append(
                {
                    "module": name,
                    "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                    "layer": int(name.split(".")[2]),
                    "prepolar": prepolar,
                    "polar": polar,
                    "relative_error_amplification": (
                        polar["relative_error"]
                        / max(prepolar["relative_error"], 1e-30)
                    ),
                }
            )
            del dense_gradient, compact_gradient
            del dense_next, compact_next, dense_request, compact_request
            del dense_update, compact_update

        payload: dict[str, Any] = {"matrices": rows}
        for label, selected in (
            ("all", rows),
            ("c_fc", [row for row in rows if row["side"] == "c_fc"]),
            ("c_proj", [row for row in rows if row["side"] == "c_proj"]),
        ):
            prepolar = _aggregate_directions(selected, "prepolar")
            polar = _aggregate_directions(selected, "polar")
            payload[label] = {
                "prepolar": prepolar,
                "polar": polar,
                "relative_error_amplification": (
                    polar["relative_error"]
                    / max(prepolar["relative_error"], 1e-30)
                ),
                "maximum_matrix_relative_error_amplification": max(
                    float(row["relative_error_amplification"])
                    for row in selected
                ),
            }
        if self.regularized_polar_enabled:
            candidate_steps = tuple(
                int(value)
                for value in self.plan["regularized_polar_frontier"][
                    "candidate_ns_steps"
                ]
            )
            native_steps = int(
                self.plan["regularized_polar_frontier"]["native_ns_steps"]
            )
            if native_steps != 5 or candidate_steps != (1, 2, 3, 4):
                raise ValueError("regularized-polar frontier changed")
            for row in rows:
                name = str(row["module"])
                module = self.observer._dense_modules[name]
                owner, group = self._dense_optimizer_owner[name]
                momentum = float(group["momentum"])
                state = owner.state[module.weight]
                buffer = state.get("momentum_buffer")
                if buffer is None:
                    buffer = torch.zeros_like(module.weight)
                dense_gradient = reference[name].to(
                    device=module.weight.device, dtype=torch.float32
                )
                compact_gradient = candidate[name].to(
                    device=module.weight.device, dtype=torch.float32
                )
                dense_request = dense_gradient + momentum * (
                    buffer.float() * momentum + dense_gradient
                )
                compact_request = compact_gradient + momentum * (
                    buffer.float() * momentum + compact_gradient
                )
                native_dense = muon_update(
                    dense_request, steps=native_steps
                ).float()
                native_task = _direction_metrics(dense_gradient, native_dense)
                regularized = {}
                for steps in candidate_steps:
                    dense_update = muon_update(dense_request, steps=steps).float()
                    compact_update = muon_update(
                        compact_request, steps=steps
                    ).float()
                    candidate_polar = _direction_metrics(
                        dense_update, compact_update
                    )
                    dense_native = _direction_metrics(native_dense, dense_update)
                    dense_task = _direction_metrics(dense_gradient, dense_update)
                    regularized[str(steps)] = {
                        "candidate_polar": candidate_polar,
                        "dense_native": dense_native,
                        "dense_task": dense_task,
                        "native_dense_task_cosine": native_task["cosine"],
                        "dense_task_alignment_retention": (
                            dense_task["cosine"]
                            / max(native_task["cosine"], 1e-30)
                        ),
                        "relative_error_amplification": (
                            candidate_polar["relative_error"]
                            / max(row["prepolar"]["relative_error"], 1e-30)
                        ),
                        "relative_error_closure_vs_native": (
                            1.0
                            - candidate_polar["relative_error"]
                            / max(row["polar"]["relative_error"], 1e-30)
                        ),
                    }
                    del dense_update, compact_update
                row["regularized_polar"] = regularized
                del dense_gradient, compact_gradient
                del dense_request, compact_request, native_dense

            payload["regularized_polar"] = {}
            for steps in candidate_steps:
                key = str(steps)
                payload["regularized_polar"][key] = {}
                for label, selected in (
                    ("all", rows),
                    ("c_fc", [row for row in rows if row["side"] == "c_fc"]),
                    (
                        "c_proj",
                        [row for row in rows if row["side"] == "c_proj"],
                    ),
                ):
                    candidate_rows = [
                        {
                            "candidate_polar": row["regularized_polar"][key][
                                "candidate_polar"
                            ],
                            "dense_native": row["regularized_polar"][key][
                                "dense_native"
                            ],
                            "dense_task": row["regularized_polar"][key][
                                "dense_task"
                            ],
                            "native_dense_task": {
                                **row["regularized_polar"][key]["dense_task"],
                                "cosine": row["regularized_polar"][key][
                                    "native_dense_task_cosine"
                                ],
                            },
                        }
                        for row in selected
                    ]
                    candidate_polar = _aggregate_directions(
                        candidate_rows, "candidate_polar"
                    )
                    dense_native = _aggregate_directions(
                        candidate_rows, "dense_native"
                    )
                    dense_task = _aggregate_directions(
                        candidate_rows, "dense_task"
                    )
                    matrix_closures = [
                        float(
                            row["regularized_polar"][key][
                                "relative_error_closure_vs_native"
                            ]
                        )
                        for row in selected
                    ]
                    payload["regularized_polar"][key][label] = {
                        "candidate_polar": candidate_polar,
                        "dense_native": dense_native,
                        "dense_native_norm_ratio": math.sqrt(
                            dense_native["candidate_energy"]
                            / max(dense_native["reference_energy"], 1e-30)
                        ),
                        "dense_task": dense_task,
                        "minimum_matrix_dense_task_alignment_retention": min(
                            float(
                                row["regularized_polar"][key][
                                    "dense_task_alignment_retention"
                                ]
                            )
                            for row in selected
                        ),
                        "relative_error_amplification": (
                            candidate_polar["relative_error"]
                            / max(
                                _aggregate_directions(selected, "prepolar")[
                                    "relative_error"
                                ],
                                1e-30,
                            )
                        ),
                        "relative_error_closure_vs_native": (
                            1.0
                            - candidate_polar["relative_error"]
                            / max(
                                _aggregate_directions(selected, "polar")[
                                    "relative_error"
                                ],
                                1e-30,
                            )
                        ),
                        "minimum_matrix_relative_error_closure_vs_native": min(
                            matrix_closures
                        ),
                        "matrix_regression_fraction": sum(
                            closure < 0.0 for closure in matrix_closures
                        )
                        / max(len(matrix_closures), 1),
                    }
        if (
            self.spectral_damping_enabled
            and self._active_probe_step in self.primary_late_steps
        ):
            payload["spectral_damping"] = self._spectral_damping_comparison(
                reference, candidate, rows
            )
        return payload

    @torch.no_grad()
    def _same_momentum_dense_updates(
        self, reference: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return the exact post-polar dense request used by the control."""
        updates: dict[str, torch.Tensor] = {}
        for name in sorted(reference):
            module = self.observer._dense_modules[name]
            owner, group = self._dense_optimizer_owner[name]
            momentum = float(group["momentum"])
            ns_steps = int(group["ns_steps"])
            buffer = owner.state[module.weight].get("momentum_buffer")
            if buffer is None:
                buffer = torch.zeros_like(module.weight)
            gradient = reference[name].to(
                device=module.weight.device, dtype=torch.float32
            )
            next_buffer = buffer.float() * momentum + gradient
            request = gradient + momentum * next_buffer
            updates[name] = muon_update(request, steps=ns_steps).float().cpu()
        return updates

    @torch.no_grad()
    def _capture_mlp_functional_samples(
        self, *, split: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Capture a frozen small activation metric for MLP update JVPs."""
        limit = int(self.plan["atlas"]["functional_sample_rows_per_layer"])
        samples: dict[str, dict[str, torch.Tensor]] = {}
        handles = []

        def pre_hook(name: str):
            def capture(_module: nn.Module, values: tuple[torch.Tensor, ...]) -> None:
                flat = values[0].detach().reshape(-1, values[0].shape[-1])
                samples.setdefault(name, {})["input"] = flat[:limit].float().cpu()
            return capture

        def output_hook(name: str):
            def capture(
                _module: nn.Module,
                _values: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                flat = output.detach().reshape(-1, output.shape[-1])
                samples.setdefault(name, {})["output"] = flat[:limit].float().cpu()
            return capture

        for name, module in self.observer._dense_modules.items():
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            if name.endswith(".c_fc"):
                handles.append(module.register_forward_hook(output_hook(name)))
        try:
            tokens = self._windows[split][0].to(self.device, non_blocking=False)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            device_type = "cuda" if "cuda" in self.device else "cpu"
            amp = (
                torch.amp.autocast(device_type=device_type, dtype=self.dtype)
                if device_type == "cuda"
                else contextlib.nullcontext()
            )
            with amp:
                self.model(inputs, targets)
        finally:
            for handle in handles:
                handle.remove()
        return samples

    @staticmethod
    def _gelu_derivative(values: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0))) + (
            values * torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
        )

    def _functional_update_metrics(
        self,
        *,
        name: str,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        samples: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        values = samples[name]
        inputs = values["input"].float()
        if name.endswith(".c_fc"):
            preactivation = values["output"].float()
            multiplier = self._gelu_derivative(preactivation)
            reference_hidden = (inputs @ reference.float().T) * multiplier
            candidate_hidden = (inputs @ candidate.float().T) * multiplier
            projection_name = name[: -len("c_fc")] + "c_proj"
            projection = self.observer._dense_modules[
                projection_name
            ].weight.detach().float().cpu()
            reference_action = reference_hidden @ projection.T
            candidate_action = candidate_hidden @ projection.T
        else:
            reference_action = inputs @ reference.float().T
            candidate_action = inputs @ candidate.float().T
        return atlas_direction_metrics(reference_action, candidate_action)

    def _evaluate_tangent_atlas(
        self,
        *,
        dense_updates: dict[str, torch.Tensor],
        dense_gradients: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        samples = self._capture_mlp_functional_samples(split="heldout")
        thresholds = self.plan["atlas_gate"]
        candidates: dict[str, Any] = {}
        for label in ("S1", "S2"):
            protocol = self._atlas_protocols[label]
            rows = []
            for name in sorted(dense_updates):
                dense = self.observer._dense_modules[name].weight.detach().float()
                compact = self.observer._shadow_modules[name].weight.detach().float()
                requested = dense_updates[name]
                projected, details = evaluate_atlas_matrix(
                    identity=name,
                    dense_weight=dense,
                    compact_weight=compact,
                    requested_update=requested,
                    protocol=protocol,
                    device=self.device,
                )
                tangent = atlas_direction_metrics(requested, projected)
                functional = self._functional_update_metrics(
                    name=name,
                    reference=requested,
                    candidate=projected,
                    samples=samples,
                )
                gradient = dense_gradients[name].double()
                dense_line = float((gradient * requested.double()).sum())
                candidate_line = float((gradient * projected.double()).sum())
                rows.append(
                    {
                        "module": name,
                        "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                        "value": details["value"],
                        "tangent": tangent,
                        "functional": functional,
                        "task_line_retention": candidate_line
                        / max(abs(dense_line), 1e-30)
                        * (1.0 if dense_line >= 0.0 else -1.0),
                        "coordinates": details["coordinates"],
                        "ambient": details["ambient"],
                        "compression_vs_dense_values": details[
                            "compression_vs_dense_values"
                        ],
                        "panels": details["panels"],
                    }
                )
            tangent_aggregate = self._aggregate_atlas_rows(rows, "tangent")
            functional_aggregate = self._aggregate_atlas_rows(rows, "functional")
            measurements = {
                "postpolar_cosine": tangent_aggregate["cosine"],
                "minimum_matrix_postpolar_cosine": min(
                    float(row["tangent"]["cosine"]) for row in rows
                ),
                "minimum_matrix_positive_line_recovery": min(
                    float(row["tangent"]["positive_line_recovery"]) for row in rows
                ),
                "minimum_matrix_value_cosine": min(
                    float(row["value"]["cosine"]) for row in rows
                ),
                "functional_jvp_cosine": functional_aggregate["cosine"],
                "minimum_matrix_functional_jvp_cosine": min(
                    float(row["functional"]["cosine"]) for row in rows
                ),
                "minimum_matrix_task_line_retention": min(
                    float(row["task_line_retention"]) for row in rows
                ),
            }
            checks = {
                key: value >= float(thresholds[key])
                for key, value in measurements.items()
            }
            candidates[label] = {
                "protocol": {
                    "atoms": protocol.atoms,
                    "stages": protocol.stages,
                    "coordinates_per_panel": protocol.coordinates_per_panel,
                },
                "measurements": measurements,
                "checks": checks,
                "passed": all(checks.values()) and _all_finite(measurements),
                "tangent": tangent_aggregate,
                "functional": functional_aggregate,
                "matrices": rows,
            }
            if label == "S1" and candidates[label]["passed"]:
                break
        return {"candidates": candidates}

    @staticmethod
    def _aggregate_atlas_rows(
        rows: list[dict[str, Any]], key: str
    ) -> dict[str, float]:
        reference = sum(float(row[key]["reference_energy"]) for row in rows)
        candidate = sum(float(row[key]["candidate_energy"]) for row in rows)
        error = sum(float(row[key]["error_energy"]) for row in rows)
        inner = sum(
            float(row[key]["cosine"])
            * math.sqrt(
                float(row[key]["reference_energy"])
                * float(row[key]["candidate_energy"])
            )
            for row in rows
        )
        cosine = inner / max(math.sqrt(reference * candidate), 1e-30)
        return {
            "reference_energy": reference,
            "candidate_energy": candidate,
            "error_energy": error,
            "relative_error": math.sqrt(error / max(reference, 1e-30)),
            "cosine": cosine,
            "positive_line_recovery": max(cosine, 0.0) ** 2,
        }

    def _summarize_tangent_atlas_gate(self) -> dict[str, Any]:
        late = [row for row in self.records if row["step"] in self.primary_late_steps]
        if len(late) != len(self.primary_late_steps):
            return {"ready": False, "passed": False}
        for label in ("S1", "S2"):
            if any(label not in row["tangent_atlas"]["candidates"] for row in late):
                continue
            passed = all(
                bool(row["tangent_atlas"]["candidates"][label]["passed"])
                for row in late
            )
            if passed:
                return {
                    "ready": True,
                    "passed": True,
                    "selected": label,
                    "classification": "FULL_RANK_TANGENT_ATLAS_PASSES",
                }
        return {
            "ready": True,
            "passed": False,
            "selected": None,
            "classification": "FULL_RANK_TANGENT_ATLAS_REJECTED",
        }

    @staticmethod
    def _matrix_regression_fraction(
        native: dict[str, Any], candidate: dict[str, Any]
    ) -> float:
        native_rows = {row["module"]: row for row in native["matrices"]}
        candidate_rows = {
            row["module"]: row for row in candidate["matrices"]
        }
        regressions = sum(
            float(candidate_rows[name]["relative_error"])
            > float(native_rows[name]["relative_error"])
            for name in native_rows
        )
        return regressions / max(len(native_rows), 1)

    def _summarize_gate(self) -> dict[str, Any]:
        late = [row for row in self.records if row["step"] in self.primary_late_steps]
        if len(late) != len(self.primary_late_steps):
            return {"ready": False, "passed": False, "selected": None}
        threshold = self.plan["frozen_gate"]
        candidate_results = {}
        for candidate in ("center", "antithetic"):
            closures = []
            cosines = []
            retentions = []
            regressions = []
            for row in late:
                heldout = row["splits"]["heldout"]["comparisons"]
                native_error = float(heldout["native"]["aggregate"]["relative_error"])
                candidate_error = float(
                    heldout[candidate]["aggregate"]["relative_error"]
                )
                closures.append(
                    1.0 - candidate_error / max(native_error, 1e-30)
                )
                cosines.append(float(heldout[candidate]["aggregate"]["cosine"]))
                dense_alignment = float(row["cross_window"]["dense"])
                candidate_alignment = float(row["cross_window"][candidate])
                retentions.append(
                    candidate_alignment / max(dense_alignment, 1e-30)
                )
                regressions.append(
                    self._matrix_regression_fraction(
                        heldout["native"], heldout[candidate]
                    )
                )
            virtual_recovery = min(
                float(row["virtual_weight"]["weighted_virtual_weight_energy_recovery"])
                for row in late
            )
            worst_virtual = min(
                float(row["virtual_weight"]["worst_matrix_virtual_weight_energy_recovery"])
                for row in late
            )
            outcomes = {
                "minimum_late_heldout_gradient_error_closure_vs_native": min(
                    closures
                ),
                "minimum_late_heldout_gradient_cosine": min(cosines),
                "minimum_late_fit_to_heldout_task_alignment_retention_vs_dense": min(
                    retentions
                ),
                "maximum_primary_matrix_regression_fraction": max(regressions),
                "minimum_virtual_weight_energy_recovery_weighted": virtual_recovery,
                "minimum_virtual_weight_energy_recovery_every_matrix": worst_virtual,
            }
            checks = {
                key: (
                    value <= float(threshold[key])
                    if key == "maximum_primary_matrix_regression_fraction"
                    else value >= float(threshold[key])
                )
                for key, value in outcomes.items()
            }
            candidate_results[candidate] = {
                "measurements": outcomes,
                "checks": checks,
                "passed": all(checks.values()),
            }
        selected = None
        if candidate_results["center"]["passed"]:
            selected = "center"
        elif candidate_results["antithetic"]["passed"]:
            selected = "antithetic"
        return {
            "ready": True,
            "all_metrics_finite": _all_finite(self.records),
            "fit_and_heldout_indices_disjoint": self.indices_disjoint,
            "candidates": candidate_results,
            "selected": selected,
            "passed": bool(selected) and _all_finite(self.records),
        }

    def _summarize_polar_gate(self) -> dict[str, Any]:
        late = [row for row in self.records if row["step"] in self.primary_late_steps]
        if len(late) != len(self.primary_late_steps):
            return {"ready": False, "passed": False, "classification": None}
        threshold = self.plan["polar_gate"]
        all_rows = [row["same_momentum_polar"]["all"] for row in late]
        outcomes = {
            "minimum_late_prepolar_cosine": min(
                float(row["prepolar"]["cosine"]) for row in all_rows
            ),
            "maximum_late_polar_cosine": max(
                float(row["polar"]["cosine"]) for row in all_rows
            ),
            "minimum_late_polar_relative_error": min(
                float(row["polar"]["relative_error"]) for row in all_rows
            ),
            "minimum_late_relative_error_amplification": min(
                float(row["relative_error_amplification"]) for row in all_rows
            ),
        }
        checks = {
            "minimum_late_prepolar_cosine": (
                outcomes["minimum_late_prepolar_cosine"]
                >= float(threshold["minimum_late_prepolar_cosine"])
            ),
            "maximum_late_polar_cosine": (
                outcomes["maximum_late_polar_cosine"]
                <= float(threshold["maximum_late_polar_cosine"])
            ),
            "minimum_late_polar_relative_error": (
                outcomes["minimum_late_polar_relative_error"]
                >= float(threshold["minimum_late_polar_relative_error"])
            ),
            "minimum_late_relative_error_amplification": (
                outcomes["minimum_late_relative_error_amplification"]
                >= float(
                    threshold["minimum_late_relative_error_amplification"]
                )
            ),
        }
        passed = all(checks.values()) and _all_finite(late)
        return {
            "ready": True,
            "passed": passed,
            "classification": (
                "MUON_POLAR_AMPLIFIES_CODEC_GRADIENT_ERROR"
                if passed
                else "MUON_POLAR_NOT_PRIMARY"
            ),
            "measurements": outcomes,
            "checks": checks,
        }

    def _combined_gate(self) -> dict[str, Any]:
        # The codec-neighbor plan deliberately replaces the legacy functional
        # and polar gates with one self-contained discriminator. Its frozen
        # gate contains neither legacy threshold family, so evaluating those
        # summaries at the terminal probe is both scientifically wrong and a
        # schema error.
        if self.codec_stability_enabled:
            return self._summarize_codec_stability_gate()
        if self.spectral_damping_enabled:
            return self._summarize_spectral_damping_gate()
        functional = self._summarize_gate()
        if not self.polar_amplification_enabled:
            return functional
        polar = self._summarize_polar_gate()
        if self.regularized_polar_enabled:
            regularized = self._summarize_regularized_polar_gate()
            return {
                "ready": bool(
                    functional["ready"]
                    and polar["ready"]
                    and regularized["ready"]
                ),
                "passed": bool(
                    functional["passed"]
                    and polar["passed"]
                    and regularized["passed"]
                ),
                "classification": regularized["classification"],
                "functional": functional,
                "polar": polar,
                "regularized_polar": regularized,
            }
        return {
            "ready": bool(functional["ready"] and polar["ready"]),
            "passed": bool(functional["passed"] and polar["passed"]),
            "classification": polar["classification"],
            "functional": functional,
            "polar": polar,
        }

    def _summarize_regularized_polar_gate(self) -> dict[str, Any]:
        late = [row for row in self.records if row["step"] in self.primary_late_steps]
        if len(late) != len(self.primary_late_steps):
            return {"ready": False, "passed": False, "classification": None}
        thresholds = self.plan["regularized_polar_gate"]
        candidates = tuple(
            int(value)
            for value in self.plan["regularized_polar_frontier"][
                "candidate_ns_steps"
            ]
        )
        outcomes: dict[str, Any] = {}
        selected = None
        for steps in candidates:
            rows = [
                record["same_momentum_polar"]["regularized_polar"][str(steps)][
                    "all"
                ]
                for record in late
            ]
            measurements = {
                "minimum_late_relative_error_closure_vs_native": min(
                    float(row["relative_error_closure_vs_native"]) for row in rows
                ),
                "maximum_late_relative_error_amplification": max(
                    float(row["relative_error_amplification"]) for row in rows
                ),
                "maximum_late_candidate_polar_relative_error": max(
                    float(row["candidate_polar"]["relative_error"]) for row in rows
                ),
                "minimum_late_dense_native_cosine": min(
                    float(row["dense_native"]["cosine"]) for row in rows
                ),
                "minimum_late_dense_native_norm_ratio": min(
                    float(row["dense_native_norm_ratio"]) for row in rows
                ),
                "maximum_late_dense_native_norm_ratio": max(
                    float(row["dense_native_norm_ratio"]) for row in rows
                ),
                "minimum_late_matrix_dense_task_alignment_retention": min(
                    float(row["minimum_matrix_dense_task_alignment_retention"])
                    for row in rows
                ),
                "maximum_late_matrix_regression_fraction": max(
                    float(row["matrix_regression_fraction"]) for row in rows
                ),
            }
            checks = {
                "minimum_late_relative_error_closure_vs_native": (
                    measurements["minimum_late_relative_error_closure_vs_native"]
                    >= float(
                        thresholds[
                            "minimum_late_relative_error_closure_vs_native"
                        ]
                    )
                ),
                "maximum_late_relative_error_amplification": (
                    measurements["maximum_late_relative_error_amplification"]
                    <= float(
                        thresholds["maximum_late_relative_error_amplification"]
                    )
                ),
                "maximum_late_candidate_polar_relative_error": (
                    measurements["maximum_late_candidate_polar_relative_error"]
                    <= float(
                        thresholds[
                            "maximum_late_candidate_polar_relative_error"
                        ]
                    )
                ),
                "minimum_late_dense_native_cosine": (
                    measurements["minimum_late_dense_native_cosine"]
                    >= float(thresholds["minimum_late_dense_native_cosine"])
                ),
                "minimum_late_dense_native_norm_ratio": (
                    measurements["minimum_late_dense_native_norm_ratio"]
                    >= float(thresholds["minimum_late_dense_native_norm_ratio"])
                ),
                "maximum_late_dense_native_norm_ratio": (
                    measurements["maximum_late_dense_native_norm_ratio"]
                    <= float(thresholds["maximum_late_dense_native_norm_ratio"])
                ),
                "minimum_late_matrix_dense_task_alignment_retention": (
                    measurements[
                        "minimum_late_matrix_dense_task_alignment_retention"
                    ]
                    >= float(
                        thresholds[
                            "minimum_late_matrix_dense_task_alignment_retention"
                        ]
                    )
                ),
                "maximum_late_matrix_regression_fraction": (
                    measurements["maximum_late_matrix_regression_fraction"]
                    <= float(
                        thresholds["maximum_late_matrix_regression_fraction"]
                    )
                ),
            }
            passed = all(checks.values()) and _all_finite(measurements)
            outcomes[str(steps)] = {
                "measurements": measurements,
                "checks": checks,
                "passed": passed,
            }
            if passed:
                selected = steps
        return {
            "ready": True,
            "passed": selected is not None,
            "selected_ns_steps": selected,
            "selection_rule": "largest candidate depth passing every frozen gate",
            "classification": (
                f"EARLY_STOPPED_MUON_NS{selected}_STABILIZES_PAIR_VQ"
                if selected is not None
                else "EARLY_STOPPED_MUON_POLAR_REGULARIZATION_REJECTED"
            ),
            "candidates": outcomes,
        }

    def _summarize_codec_stability_gate(self) -> dict[str, Any]:
        late = [
            row for row in self.records if row["step"] in self.primary_late_steps
        ]
        if len(late) != len(self.primary_late_steps):
            return {
                "ready": False,
                "passed": False,
                "classification": None,
            }
        thresholds = self.plan["frozen_gate"]
        split_rows = [
            row["codec_neighbor_stability"]["splits"][split]
            for row in late
            for split in ("fit", "heldout")
        ]
        retentions = []
        for row in late:
            fit = row["codec_neighbor_stability"]["splits"]["fit"][
                "ratios"
            ]
            heldout = row["codec_neighbor_stability"]["splits"][
                "heldout"
            ]["ratios"]
            for key in (
                "logit_kl",
                "gradient_error",
                "postpolar_error",
            ):
                left = float(fit[key])
                right = float(heldout[key])
                retentions.append(
                    min(left, right) / max(max(left, right), 1e-30)
                )
        measurements = {
            "maximum_relative_neighbor_energy_mismatch": max(
                float(
                    row["codec_neighbor_stability"]["neighbor_energy"]
                    ["maximum_relative_energy_mismatch"]
                )
                for row in late
            ),
            "minimum_late_actual_to_isotropic_logit_kl_ratio": min(
                float(row["ratios"]["logit_kl"]) for row in split_rows
            ),
            "minimum_late_actual_to_isotropic_gradient_error_ratio": min(
                float(row["ratios"]["gradient_error"])
                for row in split_rows
            ),
            "minimum_late_actual_to_isotropic_postpolar_error_ratio": min(
                float(row["ratios"]["postpolar_error"])
                for row in split_rows
            ),
            "minimum_late_actual_worse_matrix_fraction": min(
                float(row["actual_worse_matrix_fraction"])
                for row in split_rows
            ),
            "minimum_fit_to_heldout_excess_ratio_retention": min(retentions),
            "minimum_virtual_weight_energy_recovery_weighted": min(
                float(
                    row["virtual_weight"][
                        "weighted_virtual_weight_energy_recovery"
                    ]
                )
                for row in late
            ),
            "minimum_virtual_weight_energy_recovery_every_matrix": min(
                float(
                    row["virtual_weight"][
                        "worst_matrix_virtual_weight_energy_recovery"
                    ]
                )
                for row in late
            ),
        }
        checks = {
            key: (
                value
                <= float(thresholds[key])
                if key == "maximum_relative_neighbor_energy_mismatch"
                else value >= float(thresholds[key])
            )
            for key, value in measurements.items()
        }
        passed = all(checks.values()) and _all_finite(late)
        return {
            "ready": True,
            "passed": passed,
            "classification": (
                "CODEC_NEIGHBOR_PATH_INSTABILITY_CONFIRMED"
                if passed
                else "CODEC_NEIGHBOR_STABILITY_NOT_PRIMARY"
            ),
            "measurements": measurements,
            "checks": checks,
        }

    def _summarize_spectral_damping_gate(self) -> dict[str, Any]:
        late = [
            row for row in self.records if row["step"] in self.primary_late_steps
        ]
        if len(late) != len(self.primary_late_steps):
            return {
                "ready": False,
                "passed": False,
                "classification": None,
                "selected_relative_rms_ridge": None,
            }
        thresholds = self.plan["spectral_damping_gate"]
        ridges = tuple(
            float(value)
            for value in self.plan["spectral_damping_frontier"][
                "relative_rms_ridge"
            ]
        )
        outcomes: dict[str, Any] = {}
        selected = None
        for ridge in ridges:
            key = f"{ridge:.8g}"
            rows = [
                record["same_momentum_polar"]["spectral_damping"][key]["all"]
                for record in late
            ]
            measurements = {
                "minimum_late_relative_error_closure_vs_native": min(
                    float(row["relative_error_closure_vs_native"])
                    for row in rows
                ),
                "maximum_late_relative_error_amplification": max(
                    float(row["relative_error_amplification"]) for row in rows
                ),
                "maximum_late_candidate_polar_relative_error": max(
                    float(row["candidate_polar"]["relative_error"])
                    for row in rows
                ),
                "minimum_late_dense_native_cosine": min(
                    float(row["dense_native"]["cosine"]) for row in rows
                ),
                "minimum_late_matrix_dense_native_cosine": min(
                    float(row["minimum_matrix_dense_native_cosine"])
                    for row in rows
                ),
                "minimum_late_dense_native_norm_ratio": min(
                    float(row["dense_native_norm_ratio"]) for row in rows
                ),
                "maximum_late_dense_native_norm_ratio": max(
                    float(row["dense_native_norm_ratio"]) for row in rows
                ),
                "minimum_late_matrix_dense_task_alignment_retention": min(
                    float(row["minimum_matrix_dense_task_alignment_retention"])
                    for row in rows
                ),
                "maximum_late_matrix_regression_fraction": max(
                    float(row["matrix_regression_fraction"]) for row in rows
                ),
                "minimum_virtual_weight_energy_recovery_weighted": min(
                    float(
                        record["virtual_weight"][
                            "weighted_virtual_weight_energy_recovery"
                        ]
                    )
                    for record in late
                ),
                "minimum_virtual_weight_energy_recovery_every_matrix": min(
                    float(
                        record["virtual_weight"][
                            "worst_matrix_virtual_weight_energy_recovery"
                        ]
                    )
                    for record in late
                ),
            }
            checks = {
                key_name: (
                    value <= float(thresholds[key_name])
                    if key_name.startswith("maximum_")
                    else value >= float(thresholds[key_name])
                )
                for key_name, value in measurements.items()
            }
            passed = all(checks.values()) and _all_finite(measurements)
            outcomes[key] = {
                "measurements": measurements,
                "checks": checks,
                "passed": passed,
            }
            if selected is None and passed:
                selected = ridge
        return {
            "ready": True,
            "passed": selected is not None,
            "selected_relative_rms_ridge": selected,
            "selection_rule": "smallest ridge passing every frozen gate",
            "classification": (
                "NORM_PRESERVING_SPECTRAL_DAMPING_CONFIRMED"
                if selected is not None
                else "NORM_PRESERVING_SPECTRAL_DAMPING_REJECTED"
            ),
            "candidates": outcomes,
        }

    def probe(
        self,
        *,
        step: int,
        run_identity_sha256: str,
        fixed_eval_indices_sha256: str,
        terminal: bool,
    ) -> dict[str, Any] | None:
        if int(step) not in self.probe_steps:
            return None
        was_training = self.model.training
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and "cuda" in self.device
            else None
        )
        self.model.eval()
        self._active_probe_step = int(step)
        cache_prepared = False
        try:
            if hasattr(self.model, "prepare_block_fht_cache"):
                self.model.prepare_block_fht_cache(dtype=self.dtype)
                cache_prepared = True
            captured: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
            split_payload: dict[str, Any] = {}
            for split in ("fit", "heldout"):
                variants: dict[str, dict[str, torch.Tensor]] = {}
                losses: dict[str, list[float]] = {}
                capture_variants = ["dense", "native", "center", "plus"]
                if self.codec_stability_enabled:
                    capture_variants.extend(
                        ["isotropic_minus", "isotropic_plus"]
                    )
                for variant in capture_variants:
                    gradients, variant_losses = self._capture_gradients(
                        split=split, variant=variant
                    )
                    variants[variant] = gradients
                    losses[variant] = variant_losses
                variants["antithetic"] = antithetic_average(
                    variants["native"], variants["plus"]
                )
                if self.codec_stability_enabled:
                    variants["isotropic_antithetic"] = antithetic_average(
                        variants["isotropic_minus"],
                        variants["isotropic_plus"],
                    )
                captured[split] = variants
                split_payload[split] = {
                    "losses": losses,
                    "comparisons": {
                        variant: gradient_comparison(
                            variants["dense"], variants[variant]
                        )
                        for variant in ("native", "center", "antithetic")
                    },
                }
            cross_window = {
                variant: gradient_cross_cosine(
                    captured[variant_split][variant],
                    captured["heldout"]["dense"],
                )
                for variant_split, variant in (
                    ("fit", "dense"),
                    ("fit", "native"),
                    ("fit", "center"),
                    ("fit", "antithetic"),
                )
            }
            record = {
                "step": int(step),
                "virtual_weight": self._weight_center_metrics(),
                "splits": split_payload,
                "cross_window": cross_window,
            }
            if self.tangent_atlas_enabled:
                dense_updates = self._same_momentum_dense_updates(
                    captured["heldout"]["dense"]
                )
                record["tangent_atlas"] = self._evaluate_tangent_atlas(
                    dense_updates=dense_updates,
                    dense_gradients=captured["heldout"]["dense"],
                )
            if self.codec_stability_enabled:
                codec_splits = {}
                for split in ("fit", "heldout"):
                    variants = captured[split]
                    actual_gradient = gradient_comparison(
                        variants["center"], variants["antithetic"]
                    )
                    isotropic_gradient = gradient_comparison(
                        variants["center"],
                        variants["isotropic_antithetic"],
                    )
                    actual_polar = self._same_momentum_polar_comparison(
                        variants["center"], variants["antithetic"]
                    )
                    isotropic_polar = self._same_momentum_polar_comparison(
                        variants["center"],
                        variants["isotropic_antithetic"],
                    )
                    logit = self._capture_logit_stability(split=split)
                    codec_splits[split] = {
                        "logit": logit,
                        "gradient": {
                            "actual": actual_gradient,
                            "isotropic": isotropic_gradient,
                        },
                        "same_momentum_polar": {
                            "actual": actual_polar,
                            "isotropic": isotropic_polar,
                        },
                        "ratios": {
                            "logit_kl": logit["actual_to_isotropic_ratio"],
                            "gradient_error": (
                                actual_gradient["aggregate"]["relative_error"]
                                / max(
                                    isotropic_gradient["aggregate"][
                                        "relative_error"
                                    ],
                                    1e-30,
                                )
                            ),
                            "postpolar_error": (
                                actual_polar["all"]["polar"]["relative_error"]
                                / max(
                                    isotropic_polar["all"]["polar"][
                                        "relative_error"
                                    ],
                                    1e-30,
                                )
                            ),
                        },
                        "actual_worse_matrix_fraction": (
                            self._actual_worse_matrix_fraction(
                                actual_gradient, isotropic_gradient
                            )
                        ),
                    }
                record["codec_neighbor_stability"] = {
                    "neighbor_energy": self._neighbor_energy_metrics(),
                    "splits": codec_splits,
                }
            if self.polar_amplification_enabled:
                record["same_momentum_polar"] = (
                    self._same_momentum_polar_comparison(
                        captured["heldout"]["dense"],
                        captured["heldout"]["center"],
                    )
                )
            self.records.append(record)
            gate = (
                self._summarize_tangent_atlas_gate()
                if self.tangent_atlas_enabled
                else self._combined_gate()
            )
            payload = {
                "schema_version": (
                    CODEC_STABILITY_RESULT_SCHEMA
                    if self.codec_stability_enabled
                    else (
                    TANGENT_ATLAS_RESULT_SCHEMA
                    if self.tangent_atlas_enabled
                    else SPECTRAL_DAMPING_RESULT_SCHEMA
                        if self.spectral_damping_enabled
                        else (
                            REGULARIZED_POLAR_RESULT_SCHEMA
                            if self.regularized_polar_enabled
                            else (
                                POLAR_RESULT_SCHEMA
                                if self.polar_amplification_enabled
                                else RESULT_SCHEMA
                            )
                        )
                    )
                ),
                "status": "finished" if terminal else "running",
                "plan": {"path": str(self.plan_path), "sha256": self.plan_sha256},
                "source_config": {
                    "path": str(self.observer.source_config_path),
                    "sha256": self.observer.source_config_sha256,
                },
                "run_identity_sha256": run_identity_sha256,
                "fixed_eval_indices_sha256": fixed_eval_indices_sha256,
                "fit_and_heldout_indices_disjoint": self.indices_disjoint,
                "persistent_pair_vq_training_bytes": self.persistent_training_bytes,
                "records": self.records,
                "gate": gate,
            }
            if not _all_finite(payload):
                raise RuntimeError("functional-oracle payload contains non-finite values")
            atomic_json(self.result_path, payload)
            return {"step": int(step), "gate": gate}
        finally:
            self.model.zero_grad(set_to_none=True)
            if cache_prepared and hasattr(self.model, "flush_block_fht_cache"):
                self.model.flush_block_fht_cache()
            self.model.train(was_training)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
