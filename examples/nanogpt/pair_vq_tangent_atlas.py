"""Zero-update full-rank Givens-atlas ceiling for Pair-VQ MLP panels.

The production candidate keeps Pair-VQ as a coarse value chart and adds one
or two two-sided diagonal/Givens atoms per 768 x 768 panel.  This module is an
oracle only: it fits the atom to the current dense-minus-Pair-VQ residual and
projects an already-computed ambient Muon direction into the fitted tangent.
It never updates the language model.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch


TensorTuple = tuple[torch.Tensor, ...]


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def fixed_matchings(*, width: int, stages: int, seed: int) -> torch.Tensor:
    """Return deterministic disjoint pairings for an arbitrary even width."""
    if width <= 0 or width % 2:
        raise ValueError("Givens width must be positive and even")
    if stages <= 0:
        raise ValueError("Givens stages must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows: list[torch.Tensor] = []
    seen: set[tuple[int, ...]] = set()
    while len(rows) < stages:
        row = torch.randperm(width, generator=generator)
        key = tuple(int(value) for value in row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return torch.stack(rows)


def givens_transform(
    values: torch.Tensor,
    angles: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    """Apply a product of fixed-connectivity learned Givens stages."""
    if angles.ndim != 2:
        raise ValueError("angles must have shape [stages, width/2]")
    stages, pairs = angles.shape
    width = pairs * 2
    if tuple(permutations.shape) != (stages, width):
        raise ValueError("permutation shape disagrees with angles")
    if values.shape[-1] != width:
        raise ValueError("value width disagrees with Givens chart")
    output = values
    leading = [1] * (values.ndim - 1)
    for stage in range(stages):
        permutation = permutations[stage].to(output.device)
        inverse = torch.argsort(permutation)
        paired = output.index_select(-1, permutation).reshape(
            *output.shape[:-1], pairs, 2
        )
        theta = angles[stage].reshape(*leading, pairs)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        left, right = paired[..., 0], paired[..., 1]
        rotated = torch.stack(
            (cosine * left - sine * right, sine * left + cosine * right),
            dim=-1,
        ).reshape_as(output)
        output = rotated.index_select(-1, inverse)
    return output


@dataclass(frozen=True)
class AtlasProtocol:
    width: int
    stages: int
    atoms: int
    fit_steps: int
    fit_learning_rate: float
    fit_weight_decay: float
    fit_gradient_clip: float
    cg_iterations: int
    cg_tolerance: float
    cg_ridge: float
    seed: int

    @property
    def coordinates_per_panel(self) -> int:
        return self.atoms * self.width * (self.stages + 1)


@dataclass
class AtlasState:
    left_angles: torch.Tensor
    diagonal: torch.Tensor
    right_angles: torch.Tensor

    def parameters(self) -> TensorTuple:
        return (self.left_angles, self.diagonal, self.right_angles)

    def detached(self, *, requires_grad: bool = False) -> "AtlasState":
        return AtlasState(
            self.left_angles.detach().requires_grad_(requires_grad),
            self.diagonal.detach().requires_grad_(requires_grad),
            self.right_angles.detach().requires_grad_(requires_grad),
        )


class FullRankTangentPanel:
    """Sum of full-rank two-sided diagonal/Givens atoms."""

    def __init__(self, protocol: AtlasProtocol, *, identity: str, device: str):
        self.protocol = protocol
        self.device = device
        base = _stable_seed(identity, protocol.seed)
        left = []
        right = []
        for atom in range(protocol.atoms):
            left.append(
                fixed_matchings(
                    width=protocol.width,
                    stages=protocol.stages,
                    seed=base + 2 * atom,
                )
            )
            right.append(
                fixed_matchings(
                    width=protocol.width,
                    stages=protocol.stages,
                    seed=base + 2 * atom + 1,
                )
            )
        self.left_permutations = torch.stack(left).to(device)
        self.right_permutations = torch.stack(right).to(device)

    def initial_state(self, target: torch.Tensor) -> AtlasState:
        p = self.protocol
        generator = torch.Generator(device=self.device)
        generator.manual_seed(_stable_seed("initial", p.seed + target.numel()))
        angle_scale = 0.05
        left = (
            torch.randn(
                p.atoms,
                p.stages,
                p.width // 2,
                generator=generator,
                device=self.device,
            )
            * angle_scale
        )
        right = (
            torch.randn(
                p.atoms,
                p.stages,
                p.width // 2,
                generator=generator,
                device=self.device,
            )
            * angle_scale
        )
        diagonal = torch.randn(
            p.atoms,
            p.width,
            generator=generator,
            device=self.device,
        )
        diagonal.mul_(target.float().square().mean().sqrt() / math.sqrt(p.atoms))
        return AtlasState(left, diagonal, right).detached(requires_grad=True)

    def materialize_from(self, *parameters: torch.Tensor) -> torch.Tensor:
        left_angles, diagonal, right_angles = parameters
        width = self.protocol.width
        identity = torch.eye(width, device=self.device, dtype=torch.float32)
        result = torch.zeros_like(identity)
        for atom in range(self.protocol.atoms):
            values = givens_transform(
                identity,
                right_angles[atom],
                self.right_permutations[atom],
            )
            values = values * diagonal[atom]
            values = givens_transform(
                values.transpose(0, 1),
                left_angles[atom],
                self.left_permutations[atom],
            ).transpose(0, 1)
            result = result + values
        return result

    def materialize(self, state: AtlasState) -> torch.Tensor:
        return self.materialize_from(*state.parameters())

    def fit(self, target: torch.Tensor) -> tuple[AtlasState, dict[str, float]]:
        target = target.to(self.device, dtype=torch.float32)
        state = self.initial_state(target)
        parameters = list(state.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.protocol.fit_learning_rate,
            weight_decay=self.protocol.fit_weight_decay,
        )
        denominator = target.square().mean().clamp_min(1e-20)
        losses: list[float] = []
        maximum_gradient = 0.0
        for _ in range(self.protocol.fit_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = self.materialize(state)
            loss = (prediction - target).square().mean() / denominator
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite tangent-atlas fit objective")
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.protocol.fit_gradient_clip
                )
            )
            if not math.isfinite(gradient):
                raise RuntimeError("non-finite tangent-atlas fit gradient")
            maximum_gradient = max(maximum_gradient, gradient)
            optimizer.step()
            losses.append(float(loss.detach()))
        fitted = state.detached(requires_grad=False)
        prediction = self.materialize(fitted).detach()
        return fitted, {
            "initial_normalized_mse": losses[0],
            "final_normalized_mse": losses[-1],
            "minimum_normalized_mse": min(losses),
            "maximum_preclip_gradient_norm": maximum_gradient,
            **direction_metrics(target, prediction),
        }

    def project_tangent(
        self, state: AtlasState, requested: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        requested = requested.to(self.device, dtype=torch.float32)
        active = state.detached(requires_grad=True)
        primals = active.parameters()

        def function(*parameters: torch.Tensor) -> torch.Tensor:
            return self.materialize_from(*parameters)

        _value, vjp = torch.func.vjp(function, *primals)
        right_hand_side = tuple(value.detach() for value in vjp(requested))

        def operator(vector: TensorTuple) -> TensorTuple:
            _base, jvp = torch.func.jvp(function, primals, vector)
            adjoint = vjp(jvp)
            return tuple(
                value.detach() + self.protocol.cg_ridge * coordinate
                for value, coordinate in zip(adjoint, vector, strict=True)
            )

        coordinates, cg = conjugate_gradient(
            operator,
            right_hand_side,
            maximum_iterations=self.protocol.cg_iterations,
            tolerance=self.protocol.cg_tolerance,
        )
        _base, projected = torch.func.jvp(function, primals, coordinates)
        projected = projected.detach()
        return projected, {**cg, **direction_metrics(requested, projected)}


def _dot(left: TensorTuple, right: TensorTuple) -> torch.Tensor:
    return sum(
        (a.double() * b.double()).sum()
        for a, b in zip(left, right, strict=True)
    )


def _add(
    left: TensorTuple, right: TensorTuple, scale: torch.Tensor | float
) -> TensorTuple:
    return tuple(
        a + b * scale for a, b in zip(left, right, strict=True)
    )


def conjugate_gradient(
    operator: Callable[[TensorTuple], TensorTuple],
    right_hand_side: TensorTuple,
    *,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[TensorTuple, dict[str, float]]:
    solution = tuple(torch.zeros_like(value) for value in right_hand_side)
    residual = tuple(value.clone() for value in right_hand_side)
    direction = tuple(value.clone() for value in residual)
    residual_energy = _dot(residual, residual)
    initial_energy = float(residual_energy)
    iterations = 0
    for iteration in range(maximum_iterations):
        product = operator(direction)
        denominator = _dot(direction, product).clamp_min(1e-30)
        alpha = residual_energy / denominator
        solution = _add(solution, direction, alpha)
        next_residual = _add(residual, product, -alpha)
        next_energy = _dot(next_residual, next_residual)
        iterations = iteration + 1
        if float(next_energy.sqrt()) <= tolerance * max(
            math.sqrt(initial_energy), 1e-30
        ):
            residual = next_residual
            residual_energy = next_energy
            break
        beta = next_energy / residual_energy.clamp_min(1e-30)
        direction = _add(next_residual, direction, beta)
        residual = next_residual
        residual_energy = next_energy
    return solution, {
        "cg_iterations": float(iterations),
        "cg_initial_residual_norm": math.sqrt(initial_energy),
        "cg_final_residual_norm": math.sqrt(float(residual_energy)),
    }


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
        "relative_error": math.sqrt(error_energy / max(reference_energy, 1e-30)),
        "cosine": cosine,
        "positive_line_recovery": max(cosine, 0.0) ** 2,
        "candidate_to_reference_energy": candidate_energy
        / max(reference_energy, 1e-30),
    }


def split_panels(matrix: torch.Tensor, width: int) -> list[torch.Tensor]:
    rows, columns = matrix.shape
    if (rows, columns) == (4 * width, width):
        return [matrix[index * width : (index + 1) * width] for index in range(4)]
    if (rows, columns) == (width, 4 * width):
        return [matrix[:, index * width : (index + 1) * width] for index in range(4)]
    raise ValueError(f"unsupported MLP matrix shape {tuple(matrix.shape)}")


def join_panels(panels: Iterable[torch.Tensor], shape: torch.Size) -> torch.Tensor:
    values = list(panels)
    if len(values) != 4:
        raise ValueError("exactly four panels are required")
    if shape[0] == 4 * shape[1]:
        return torch.cat(values, dim=0)
    if shape[1] == 4 * shape[0]:
        return torch.cat(values, dim=1)
    raise ValueError(f"unsupported MLP matrix shape {tuple(shape)}")


def evaluate_matrix(
    *,
    identity: str,
    dense_weight: torch.Tensor,
    compact_weight: torch.Tensor,
    requested_update: torch.Tensor,
    protocol: AtlasProtocol,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    residual_panels = split_panels(dense_weight - compact_weight, protocol.width)
    request_panels = split_panels(requested_update, protocol.width)
    projected: list[torch.Tensor] = []
    fitted_panels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for index, (residual, requested) in enumerate(
        zip(residual_panels, request_panels, strict=True)
    ):
        panel = FullRankTangentPanel(
            protocol,
            identity=f"{identity}:panel{index}",
            device=device,
        )
        state, fit = panel.fit(residual)
        fitted_panels.append(panel.materialize(state).detach().cpu())
        direction, tangent = panel.project_tangent(state, requested)
        projected.append(direction.cpu())
        rows.append({"panel": index, "fit": fit, "tangent": tangent})
        del panel, state, direction
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    candidate = join_panels(projected, requested_update.shape)
    fitted_residual = join_panels(fitted_panels, dense_weight.shape)
    dense_cpu = dense_weight.detach().float().cpu()
    compact_cpu = compact_weight.detach().float().cpu()
    return candidate, {
        "coordinates": 4 * protocol.coordinates_per_panel,
        "ambient": int(requested_update.numel()),
        "compression_vs_dense_values": requested_update.numel()
        / (4 * protocol.coordinates_per_panel),
        "fit": aggregate_panel_metrics(rows, "fit"),
        "value": direction_metrics(
            dense_cpu, compact_cpu + fitted_residual
        ),
        "tangent": aggregate_panel_metrics(rows, "tangent"),
        "panels": rows,
    }


def aggregate_panel_metrics(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float]:
    selected = [row[key] for row in rows]
    reference = sum(float(row["reference_energy"]) for row in selected)
    candidate = sum(float(row["candidate_energy"]) for row in selected)
    error = sum(float(row["error_energy"]) for row in selected)
    inner = sum(
        float(row["cosine"])
        * math.sqrt(
            float(row["reference_energy"]) * float(row["candidate_energy"])
        )
        for row in selected
    )
    cosine = inner / max(math.sqrt(reference * candidate), 1e-30)
    return {
        "reference_energy": reference,
        "candidate_energy": candidate,
        "error_energy": error,
        "relative_error": math.sqrt(error / max(reference, 1e-30)),
        "cosine": cosine,
        "positive_line_recovery": max(cosine, 0.0) ** 2,
        "worst_panel_cosine": min(float(row["cosine"]) for row in selected),
    }
