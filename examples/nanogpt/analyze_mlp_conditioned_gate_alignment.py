"""Screen a token-conditioned MLP output gate before training.

Static bilateral charts have held-out repair capacity, but CE, Muon/pullback
metrics, and residual summary objectives do not identify the useful chart
direction.  This probe adds the smallest coordinate family that exposes
token-conditioned output orientation without a learned basis:

    update' = update + update * (slope * LN(residual) + bias)

``slope`` and ``bias`` are residual-width vectors per layer and initialize to
zero, so the gate is exactly identity.  The diagnostic compares causal CE and
dense-teacher MLP-output MSE gradients in these same gate coordinates on
fixed fit and held-out tokens.  No parameter update is applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    collect_model,
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_chart_gradient_alignment import (
    load_chart_model,
    vector_alignment,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import normalized_fht_last_dim


GATE_GROUPS = ("slope", "bias")


@dataclass(frozen=True)
class GateSplitData:
    name: str
    batches: list[torch.Tensor]
    token_sha256: str
    mlp_input: dict[int, torch.Tensor]
    source_mlp_out: dict[int, torch.Tensor]
    pre_gelu: dict[int, torch.Tensor]
    teacher_mlp_out: dict[int, torch.Tensor]
    cproj_weight: dict[int, torch.Tensor]
    cproj_bias: dict[int, torch.Tensor | None]


class ResidualConditionedOutputGate(torch.nn.Module):
    """Identity-initialized token-conditioned residual-width diagonal."""

    def __init__(self, width: int, scale: float = 1.0) -> None:
        super().__init__()
        self.slope = torch.nn.Parameter(torch.zeros(int(width)))
        self.bias = torch.nn.Parameter(torch.zeros(int(width)))
        self.scale = float(scale)

    def modulation(self, condition: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.addcmul(
            self.bias,
            condition,
            self.slope,
        )

    def forward(
        self, condition: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        if condition.shape != update.shape:
            raise ValueError(
                "gate condition and update must be aligned, got "
                f"{tuple(condition.shape)} and {tuple(update.shape)}"
            )
        return torch.addcmul(
            update,
            update,
            self.modulation(condition),
        )


class FixedBasisBilinearOutputGate(torch.nn.Module):
    """Identity-initialized fixed-basis token-conditioned channel mixer.

    With a fixed orthogonal signed/permuted block-Hadamard basis ``Q``, this
    applies

        update + Q^-1[(Q update) * (slope * Q condition + bias)].

    Unlike the raw diagonal gate, each spectral coordinate can rotate the
    update across residual channels.  The basis is fixed and only the two
    residual-width coordinate vectors are trainable.
    """

    def __init__(
        self,
        width: int,
        scale: float = 1.0,
        *,
        basis_block_size: int = 256,
        seed: int = 271828,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.scale = float(scale)
        self.basis_block_size = int(basis_block_size)
        if (
            self.basis_block_size <= 0
            or self.basis_block_size & (self.basis_block_size - 1)
            or self.width % self.basis_block_size
        ):
            raise ValueError(
                "basis_block_size must be a power of two dividing width"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutation = torch.randperm(
            self.width, generator=generator, device="cpu"
        )
        signs = (
            torch.randint(
                0,
                2,
                (self.width,),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            * 2.0
            - 1.0
        )
        self.register_buffer("permutation", permutation, persistent=True)
        self.register_buffer(
            "inverse_permutation",
            torch.argsort(permutation),
            persistent=True,
        )
        self.register_buffer("signs", signs, persistent=True)
        self.slope = torch.nn.Parameter(torch.zeros(self.width))
        self.bias = torch.nn.Parameter(torch.zeros(self.width))

    def _basis(self, values: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        signs = self.signs.to(device=values.device, dtype=values.dtype)
        if inverse:
            values = values * signs
            grouped = values.reshape(
                *values.shape[:-1],
                self.width // self.basis_block_size,
                self.basis_block_size,
            )
            values = normalized_fht_last_dim(grouped).reshape_as(values)
            return values.index_select(-1, self.inverse_permutation)
        values = values.index_select(-1, self.permutation)
        grouped = values.reshape(
            *values.shape[:-1],
            self.width // self.basis_block_size,
            self.basis_block_size,
        )
        values = normalized_fht_last_dim(grouped).reshape_as(values)
        return values * signs

    def modulation(self, condition: torch.Tensor) -> torch.Tensor:
        spectral_condition = self._basis(condition, inverse=False)
        return self.scale * torch.addcmul(
            self.bias.to(dtype=condition.dtype),
            spectral_condition,
            self.slope.to(dtype=condition.dtype),
        )

    def forward(
        self, condition: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        if condition.shape != update.shape:
            raise ValueError(
                "gate condition and update must be aligned, got "
                f"{tuple(condition.shape)} and {tuple(update.shape)}"
            )
        spectral_update = self._basis(update, inverse=False)
        correction = spectral_update * self.modulation(condition)
        return update + self._basis(correction, inverse=True)


class _FixedBlockHadamardBasis(torch.nn.Module):
    """One fixed signed/permuted normalized block-Hadamard basis."""

    def __init__(
        self,
        width: int,
        *,
        basis_block_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.basis_block_size = int(basis_block_size)
        if (
            self.basis_block_size <= 0
            or self.basis_block_size & (self.basis_block_size - 1)
            or self.width % self.basis_block_size
        ):
            raise ValueError(
                "basis_block_size must be a power of two dividing width"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutation = torch.randperm(
            self.width, generator=generator, device="cpu"
        )
        signs = (
            torch.randint(
                0,
                2,
                (self.width,),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            * 2.0
            - 1.0
        )
        self.register_buffer("permutation", permutation, persistent=True)
        self.register_buffer(
            "inverse_permutation",
            torch.argsort(permutation),
            persistent=True,
        )
        self.register_buffer("signs", signs, persistent=True)

    def transform(
        self,
        values: torch.Tensor,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        signs = self.signs.to(device=values.device, dtype=values.dtype)
        if inverse:
            values = values * signs
            grouped = values.reshape(
                *values.shape[:-1],
                self.width // self.basis_block_size,
                self.basis_block_size,
            )
            values = normalized_fht_last_dim(grouped).reshape_as(values)
            return values.index_select(-1, self.inverse_permutation)
        values = values.index_select(-1, self.permutation)
        grouped = values.reshape(
            *values.shape[:-1],
            self.width // self.basis_block_size,
            self.basis_block_size,
        )
        values = normalized_fht_last_dim(grouped).reshape_as(values)
        return values * signs


class UntiedFixedBasisBilinearOutputGate(torch.nn.Module):
    """Non-symmetric fixed-basis bilinear token-conditioned mixer.

    The condition, update, and output correction use independent fixed
    orthogonal bases:

        update + Q_out^-1[
            (Q_update update) *
            (slope * Q_condition condition + bias)
        ].

    Only ``slope`` and ``bias`` are trainable.  Untying the fixed bases makes
    the bilinear operator non-diagonal in any single basis without adding a
    learned basis or increasing the number of learned coordinates.
    """

    def __init__(
        self,
        width: int,
        scale: float = 1.0,
        *,
        basis_block_size: int = 256,
        condition_seed: int = 271828,
        update_seed: int = 376557,
        output_seed: int = 481286,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.scale = float(scale)
        self.condition_basis = _FixedBlockHadamardBasis(
            self.width,
            basis_block_size=basis_block_size,
            seed=condition_seed,
        )
        self.update_basis = _FixedBlockHadamardBasis(
            self.width,
            basis_block_size=basis_block_size,
            seed=update_seed,
        )
        self.output_basis = _FixedBlockHadamardBasis(
            self.width,
            basis_block_size=basis_block_size,
            seed=output_seed,
        )
        self.slope = torch.nn.Parameter(torch.zeros(self.width))
        self.bias = torch.nn.Parameter(torch.zeros(self.width))

    def modulation(self, condition: torch.Tensor) -> torch.Tensor:
        spectral_condition = self.condition_basis.transform(
            condition,
            inverse=False,
        )
        return self.scale * torch.addcmul(
            self.bias.to(dtype=condition.dtype),
            spectral_condition,
            self.slope.to(dtype=condition.dtype),
        )

    def forward(
        self,
        condition: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if condition.shape != update.shape:
            raise ValueError(
                "gate condition and update must be aligned, got "
                f"{tuple(condition.shape)} and {tuple(update.shape)}"
            )
        spectral_update = self.update_basis.transform(
            update,
            inverse=False,
        )
        correction = spectral_update * self.modulation(condition)
        return update + self.output_basis.transform(
            correction,
            inverse=True,
        )


class PostGeluConditionedBilinearOutputGate(
    UntiedFixedBasisBilinearOutputGate
):
    """Use the current token's post-GELU activation as gate condition.

    A fixed signed four-to-one projection maps the 4x expansion activation
    back to residual width, then per-token RMS normalization removes a
    redundant magnitude degree of freedom.  The parent class applies the
    independently based bilinear output correction.  Only its residual-width
    slope/bias vectors are trainable.
    """

    def __init__(
        self,
        width: int,
        scale: float = 1.0,
        *,
        basis_block_size: int = 256,
        condition_seed: int = 271828,
        update_seed: int = 376557,
        output_seed: int = 481286,
        projection_seed: int = 586015,
        expansion: int = 4,
        rms_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(
            width,
            scale,
            basis_block_size=basis_block_size,
            condition_seed=condition_seed,
            update_seed=update_seed,
            output_seed=output_seed,
        )
        self.expansion = int(expansion)
        self.rms_epsilon = float(rms_epsilon)
        if self.expansion <= 0:
            raise ValueError("expansion must be positive")
        if self.rms_epsilon <= 0.0:
            raise ValueError("rms_epsilon must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(projection_seed))
        projection_signs = (
            torch.randint(
                0,
                2,
                (self.expansion, self.width),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            * 2.0
            - 1.0
        )
        self.register_buffer(
            "projection_signs",
            projection_signs,
            persistent=True,
        )

    def activation_condition(self, activated: torch.Tensor) -> torch.Tensor:
        expected = self.expansion * self.width
        if activated.shape[-1] != expected:
            raise ValueError(
                "post-GELU condition width mismatch: expected "
                f"{expected}, got {activated.shape[-1]}"
            )
        grouped = activated.reshape(
            *activated.shape[:-1],
            self.expansion,
            self.width,
        )
        signs = self.projection_signs.to(
            device=activated.device,
            dtype=activated.dtype,
        )
        condition = (grouped * signs).sum(dim=-2) / math.sqrt(
            self.expansion
        )
        rms = condition.float().square().mean(
            dim=-1, keepdim=True
        ).add(self.rms_epsilon).sqrt()
        return condition / rms.to(dtype=condition.dtype)

    def forward(
        self,
        activated: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            self.activation_condition(activated),
            update,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def gate_key(layer: int, group: str) -> str:
    if group not in GATE_GROUPS:
        raise ValueError(f"invalid gate group {group!r}")
    return f"layer.{int(layer)}.{group}"


def split_gate_key(key: str) -> tuple[int, str]:
    prefix, layer, group = key.split(".", 2)
    if prefix != "layer" or group not in GATE_GROUPS:
        raise ValueError(f"invalid gate-gradient key {key!r}")
    return int(layer), group


def flatten(
    gradients: dict[str, torch.Tensor], selected: list[str]
) -> torch.Tensor:
    return torch.cat(
        [gradients[key].detach().float().reshape(-1).cpu() for key in selected]
    )


def alignment_rows(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    comparison: str,
    split: str,
) -> list[dict[str, object]]:
    if set(left) != set(right):
        raise ValueError("gradient maps do not have identical keys")
    keys = sorted(left)
    layers = sorted({split_gate_key(key)[0] for key in keys})
    groups = sorted({split_gate_key(key)[1] for key in keys})
    rows: list[dict[str, object]] = []

    def append(scope: str, layer: int | None, group: str | None) -> None:
        selected = [
            key
            for key in keys
            if (layer is None or split_gate_key(key)[0] == layer)
            and (group is None or split_gate_key(key)[1] == group)
        ]
        rows.append(
            {
                "comparison": comparison,
                "split": split,
                "scope": scope,
                "layer": "" if layer is None else layer,
                "group": "" if group is None else group,
                **vector_alignment(
                    flatten(left, selected), flatten(right, selected)
                ),
            }
        )

    append("global", None, None)
    for group in groups:
        append("group", None, group)
    for layer in layers:
        append("layer", layer, None)
        for group in groups:
            append("layer_group", layer, group)
    return rows


def load_gate_model(
    checkpoint: Path,
    device: str,
    layers: list[int],
    initial_output_log_gain: float,
    *,
    gate_kind: str = "diagonal",
    basis_block_size: int = 256,
    basis_seed: int = 271828,
) -> tuple[
    torch.nn.Module,
    dict[int, torch.nn.Module],
    list[torch.utils.hooks.RemovableHandle],
]:
    model = load_chart_model(
        checkpoint, device, layers, initial_output_log_gain
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gates: dict[int, torch.nn.Module] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    latest_postgelu: dict[int, torch.Tensor] = {}
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        if gate_kind == "diagonal":
            gate = ResidualConditionedOutputGate(model.config.n_embd)
        elif gate_kind == "fixed_bilinear":
            gate = FixedBasisBilinearOutputGate(
                model.config.n_embd,
                basis_block_size=basis_block_size,
                seed=basis_seed + layer * 64,
            )
        elif gate_kind == "fixed_bilinear_untied":
            gate = UntiedFixedBasisBilinearOutputGate(
                model.config.n_embd,
                basis_block_size=basis_block_size,
                condition_seed=basis_seed + layer * 64,
                update_seed=basis_seed + 104729 + layer * 64,
                output_seed=basis_seed + 2 * 104729 + layer * 64,
            )
        elif gate_kind == "postgelu_bilinear_untied":
            gate = PostGeluConditionedBilinearOutputGate(
                model.config.n_embd,
                basis_block_size=basis_block_size,
                condition_seed=basis_seed + layer * 64,
                update_seed=basis_seed + 104729 + layer * 64,
                output_seed=basis_seed + 2 * 104729 + layer * 64,
                projection_seed=basis_seed + 3 * 104729 + layer * 64,
            )
        else:
            raise ValueError(f"unsupported gate kind: {gate_kind}")
        gate = gate.to(device)
        mlp.add_module("residual_conditioned_output_gate_probe", gate)

        if isinstance(gate, PostGeluConditionedBilinearOutputGate):
            def gelu_hook(
                _module,
                _inputs,
                output,
                *,
                selected_layer: int = layer,
            ) -> None:
                latest_postgelu[selected_layer] = output

            handles.append(mlp.gelu.register_forward_hook(gelu_hook))

        def hook(
            _module,
            inputs,
            output,
            *,
            selected_gate: torch.nn.Module = gate,
            selected_layer: int = layer,
        ):
            if not inputs:
                raise RuntimeError("MLP gate hook received no input")
            if isinstance(
                selected_gate,
                PostGeluConditionedBilinearOutputGate,
            ):
                activated = latest_postgelu.pop(selected_layer, None)
                if activated is None:
                    raise RuntimeError(
                        "post-GELU gate hook did not capture an activation"
                    )
                return selected_gate(activated, output)
            return selected_gate(inputs[0], output)

        handles.append(mlp.register_forward_hook(hook))
        gates[layer] = gate
    return model, gates, handles


def gate_parameters(
    gates: dict[int, torch.nn.Module],
    groups: tuple[str, ...] = GATE_GROUPS,
) -> dict[str, torch.nn.Parameter]:
    if not groups or any(group not in GATE_GROUPS for group in groups):
        raise ValueError(f"invalid gate groups: {groups!r}")
    output: dict[str, torch.nn.Parameter] = {}
    for layer, gate in gates.items():
        for group in groups:
            output[gate_key(layer, group)] = getattr(gate, group)
    return output


def clone_gradients(
    parameters: dict[str, torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, parameter in parameters.items():
        if parameter.grad is None:
            raise RuntimeError(f"gate parameter {key} has no gradient")
        output[key] = parameter.grad.detach().float().cpu().clone()
    return output


def task_ce_gradients(
    model: torch.nn.Module,
    parameters: dict[str, torch.nn.Parameter],
    batches: list[torch.Tensor],
    device: str,
) -> tuple[dict[str, torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    cache_dtype = (
        torch.bfloat16 if device.startswith("cuda") else torch.float32
    )
    model.prepare_block_fht_cache(dtype=cache_dtype)
    losses: list[float] = []
    for tokens in batches:
        inputs = tokens[:, :-1].contiguous().to(device)
        targets = tokens[:, 1:].contiguous().to(device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with context:
            _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss.detach()))
        (loss / len(batches)).backward()
    gradients = clone_gradients(parameters)
    model.flush_block_fht_cache()
    return gradients, sum(losses) / len(losses)


def teacher_mse_gradients(
    model: torch.nn.Module,
    gates: dict[int, torch.nn.Module],
    split: GateSplitData,
    layers: list[int],
    device: str,
    *,
    gate_groups: tuple[str, ...] = GATE_GROUPS,
    use_source_mlp_out: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[int, float]]:
    output: dict[str, torch.Tensor] = {}
    losses: dict[int, float] = {}
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        gate = gates[layer]
        condition = split.mlp_input[layer].to(device)
        target = split.teacher_mlp_out[layer].to(device)
        if use_source_mlp_out:
            prediction = split.source_mlp_out[layer].to(device)
        else:
            pre_gelu = split.pre_gelu[layer].to(device)
            weight = split.cproj_weight[layer].to(device)
            bias = split.cproj_bias[layer]
            bias = bias.to(device) if bias is not None else None
            charted_weight = mlp._materialize_charted_cproj_weight(weight)
            prediction = F.linear(F.gelu(pre_gelu), charted_weight, bias)
        if isinstance(gate, PostGeluConditionedBilinearOutputGate):
            condition = F.gelu(split.pre_gelu[layer].to(device))
        prediction = gate(condition, prediction)
        loss = F.mse_loss(prediction, target)
        parameters = [getattr(gate, group) for group in gate_groups]
        gradients = torch.autograd.grad(loss, parameters)
        losses[layer] = float(loss.detach())
        for group, gradient in zip(gate_groups, gradients, strict=True):
            output[gate_key(layer, group)] = gradient.detach().float().cpu()
    return output, losses


def collect_split(
    *,
    name: str,
    seed: int,
    attention_only: Path,
    plain_cproj: Path,
    data_dir: Path,
    layers: list[int],
    batch_size: int,
    block_size: int,
    batches_count: int,
    sample_cap: int,
    device: str,
) -> GateSplitData:
    batches = fixed_validation_batches(
        data_dir,
        batch_size,
        block_size,
        batches_count,
        seed,
    )
    digest = tensor_sha256(torch.cat(batches))
    teacher, _, _ = collect_model(
        attention_only,
        batches,
        layers,
        sample_cap,
        device,
        collect_pre_gelu=False,
    )
    source, weights, biases = collect_model(
        plain_cproj,
        batches,
        layers,
        sample_cap,
        device,
        collect_pre_gelu=True,
        collect_mlp_input=True,
    )
    return GateSplitData(
        name=name,
        batches=batches,
        token_sha256=digest,
        mlp_input={
            layer: source[(layer, "mlp_input")] for layer in layers
        },
        source_mlp_out={
            layer: source[(layer, "mlp_out")] for layer in layers
        },
        pre_gelu={
            layer: source[(layer, "pre_gelu")] for layer in layers
        },
        teacher_mlp_out={
            layer: teacher[(layer, "mlp_out")] for layer in layers
        },
        cproj_weight=weights,
        cproj_bias=biases,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", required=True, type=Path)
    parser.add_argument("--plain-cproj", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--ce-batches", type=int, default=8)
    parser.add_argument("--sample-cap", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=20260716)
    parser.add_argument("--holdout-sample-seed", type=int, default=20260717)
    parser.add_argument(
        "--initial-output-log-gain", type=float, default=0.125
    )
    parser.add_argument(
        "--gate-kind",
        choices=(
            "diagonal",
            "fixed_bilinear",
            "fixed_bilinear_untied",
            "postgelu_bilinear_untied",
        ),
        default="diagonal",
    )
    parser.add_argument("--basis-block-size", type=int, default=256)
    parser.add_argument("--basis-seed", type=int, default=271828)
    parser.add_argument("--gate-groups", default="slope,bias")
    parser.add_argument(
        "--teacher-source-output",
        action="store_true",
        help=(
            "apply the zero-initialized probe to the checkpoint's captured "
            "MLP output instead of reconstructing an identity chart output"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",") if part]
    gate_groups = tuple(
        part.strip() for part in args.gate_groups.split(",") if part.strip()
    )
    if not layers:
        raise ValueError("at least one layer is required")
    if (
        not gate_groups
        or len(set(gate_groups)) != len(gate_groups)
        or any(group not in GATE_GROUPS for group in gate_groups)
    ):
        raise ValueError(
            f"gate-groups must be unique values from {GATE_GROUPS}"
        )
    if args.sample_cap > args.batch_size * args.block_size * args.batches:
        raise ValueError("sample cap exceeds the available activation rows")
    if args.ce_batches <= 0 or args.ce_batches > args.batches:
        raise ValueError("ce-batches must be in [1, batches]")

    fit = collect_split(
        name="fit",
        seed=args.sample_seed,
        attention_only=args.attention_only,
        plain_cproj=args.plain_cproj,
        data_dir=args.data_dir,
        layers=layers,
        batch_size=args.batch_size,
        block_size=args.block_size,
        batches_count=args.batches,
        sample_cap=args.sample_cap,
        device=args.device,
    )
    holdout = collect_split(
        name="holdout",
        seed=args.holdout_sample_seed,
        attention_only=args.attention_only,
        plain_cproj=args.plain_cproj,
        data_dir=args.data_dir,
        layers=layers,
        batch_size=args.batch_size,
        block_size=args.block_size,
        batches_count=args.batches,
        sample_cap=args.sample_cap,
        device=args.device,
    )

    model, gates, handles = load_gate_model(
        args.plain_cproj,
        args.device,
        layers,
        args.initial_output_log_gain,
        gate_kind=args.gate_kind,
        basis_block_size=args.basis_block_size,
        basis_seed=args.basis_seed,
    )
    parameters = gate_parameters(gates, gate_groups)
    rows: list[dict[str, object]] = []
    ce_by_split: dict[str, dict[str, torch.Tensor]] = {}
    teacher_by_split: dict[str, dict[str, torch.Tensor]] = {}
    diagnostics: dict[str, object] = {}
    try:
        for split in (fit, holdout):
            ce_gradient, ce_loss = task_ce_gradients(
                model,
                parameters,
                split.batches[: args.ce_batches],
                args.device,
            )
            teacher_gradient, teacher_losses = teacher_mse_gradients(
                model,
                gates,
                split,
                layers,
                args.device,
                gate_groups=gate_groups,
                use_source_mlp_out=args.teacher_source_output,
            )
            ce_by_split[split.name] = ce_gradient
            teacher_by_split[split.name] = teacher_gradient
            comparison_rows = alignment_rows(
                ce_gradient,
                teacher_gradient,
                comparison="task_ce_vs_teacher_mse",
                split=split.name,
            )
            rows.extend(comparison_rows)
            diagnostics[split.name] = {
                "task_ce": ce_loss,
                "teacher_mse_by_layer": {
                    str(layer): teacher_losses[layer] for layer in layers
                },
            }
            print(
                f"split={split.name} task_ce={ce_loss:.6f} "
                f"global_cosine={comparison_rows[0]['cosine']:.6f}",
                flush=True,
            )
        rows.extend(
            alignment_rows(
                ce_by_split["fit"],
                ce_by_split["holdout"],
                comparison="task_ce_fit_vs_holdout",
                split="cross_split",
            )
        )
        rows.extend(
            alignment_rows(
                teacher_by_split["fit"],
                teacher_by_split["holdout"],
                comparison="teacher_mse_fit_vs_holdout",
                split="cross_split",
            )
        )
    finally:
        for handle in handles:
            handle.remove()

    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "mlp_conditioned_gate_alignment.csv"
    write_csv(csv_path, rows)
    metadata = {
        "schema_version": "mlp_conditioned_gate_alignment_v1",
        "scientific_scope": (
            "deterministic no-update identity-initialized structural "
            "gradient diagnostic"
        ),
        "attention_only": {
            "path": str(args.attention_only),
            "sha256": sha256(args.attention_only),
        },
        "plain_cproj": {
            "path": str(args.plain_cproj),
            "sha256": sha256(args.plain_cproj),
        },
        "data_dir": str(args.data_dir),
        "layers": layers,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "batches": args.batches,
        "ce_batches": args.ce_batches,
        "sample_cap": args.sample_cap,
        "sample_seed": args.sample_seed,
        "holdout_sample_seed": args.holdout_sample_seed,
        "fit_token_sha256": fit.token_sha256,
        "holdout_token_sha256": holdout.token_sha256,
        "initial_output_log_gain": args.initial_output_log_gain,
        "gate": {
            "kind": args.gate_kind,
            "formula": (
                "update + update * (slope * mlp_input + bias)"
                if args.gate_kind == "diagonal"
                else (
                    "update + Q^-1[(Q update) * "
                    "(slope * Q mlp_input + bias)]"
                    if args.gate_kind == "fixed_bilinear"
                    else (
                        (
                            "update + Q_output^-1[(Q_update update) * "
                            "(slope * Q_condition "
                            "rmsnorm(P_fixed postgelu) + bias)]"
                        )
                        if args.gate_kind
                        == "postgelu_bilinear_untied"
                        else (
                            "update + Q_output^-1[(Q_update update) * "
                            "(slope * Q_condition mlp_input + bias)]"
                        )
                    )
                )
            ),
            "parameters_per_selected_layer": 2 * model.config.n_embd,
            "selected_parameter_groups": list(gate_groups),
            "selected_parameters_per_layer": (
                len(gate_groups) * model.config.n_embd
            ),
            "identity_initialized": True,
            "learned_basis": False,
            "lora_adapter": False,
            "basis_block_size": (
                args.basis_block_size
                if args.gate_kind != "diagonal"
                else None
            ),
            "basis_seed": (
                args.basis_seed
                if args.gate_kind != "diagonal"
                else None
            ),
            "basis_seeds": (
                {
                    "condition": args.basis_seed,
                    "update": args.basis_seed + 104729,
                    "output": args.basis_seed + 2 * 104729,
                }
                if args.gate_kind
                in (
                    "fixed_bilinear_untied",
                    "postgelu_bilinear_untied",
                )
                else None
            ),
            "condition_source": (
                "fixed_signed_four_to_one_rms_normalized_postgelu"
                if args.gate_kind == "postgelu_bilinear_untied"
                else "layer_normalized_residual"
            ),
            "projection_seed": (
                args.basis_seed + 3 * 104729
                if args.gate_kind == "postgelu_bilinear_untied"
                else None
            ),
            "teacher_prediction_source": (
                "captured_checkpoint_mlp_output"
                if args.teacher_source_output
                else "reconstructed_identity_bilateral_chart_output"
            ),
        },
        "diagnostics": diagnostics,
        "global_alignment": [
            row for row in rows if row["scope"] == "global"
        ],
        "source": {
            "path": str(Path(__file__).relative_to(root)),
            "sha256": sha256(Path(__file__)),
            "git_commit": git_head(root),
        },
        "csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
    }
    metadata_path = args.output / "mlp_conditioned_gate_alignment.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "csv_sha256": sha256(csv_path),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
