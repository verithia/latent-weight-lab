#!/usr/bin/env python3
"""Frozen H69 initialization-conditioned paired-neuron manifold audit.

This is a representation audit, not language-model training.  It learns one
shared nonlinear decoder against joint c_fc/c_proj displacement PCs while the
exact step-zero complete neurons act as procedural side information.  No PC,
trajectory vector, W0 tensor, or dense shadow is part of the compact payload.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_synthetic_muon_program_full_audit import (
    load_trajectory_inventory,
)


SCHEMA_VERSION = "nanogpt_mlp_initialization_conditioned_paired_manifold_v1"
PARAMETER_TEMPLATE = "transformer.h.{layer}.mlp.{target}.weight"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def deployment_accounting(
    *, width: int = 768, shared_width: int = 176, latent_width: int = 16,
    layers: int = 12, hidden_width: int = 3072,
) -> dict[str, int | float]:
    shared_maps = 4 * width * shared_width
    injection = shared_width * latent_width
    layer_embeddings = layers * shared_width
    live_latents = layers * latent_width
    total = shared_maps + injection + layer_embeddings + live_latents
    dense = layers * 2 * hidden_width * width
    # P_u/P_v are evaluated once when the deterministic W0 is regenerated.
    # C and Q_u/Q_v are evaluated whenever a live latent is materialized into
    # dense MLP weights.  The ordinary token path remains the same two dense
    # matrix multiplies after materialization.
    static_key_matrix_flops = layers * hidden_width * 4 * width * shared_width
    live_latent_matrix_flops = (
        layers * 2 * shared_width * latent_width
        + layers * hidden_width * 4 * width * shared_width
    )
    dense_mlp_matrix_flops_per_token = layers * 4 * hidden_width * width
    return {
        "shared_map_values": shared_maps,
        "latent_injection_values": injection,
        "layer_embedding_values": layer_embeddings,
        "live_latent_values": live_latents,
        "total_fp16_values": total,
        "total_checkpoint_payload_bytes": 2 * total,
        "dense_replaced_mlp_fp16_values": dense,
        "dense_replaced_mlp_fp16_bytes": 2 * dense,
        "checkpoint_byte_fraction": total / dense,
        "persistent_w0_bytes": 0,
        "persistent_empirical_basis_bytes": 0,
        "static_key_matrix_flops": static_key_matrix_flops,
        "live_latent_refresh_matrix_flops": live_latent_matrix_flops,
        "dense_mlp_matrix_flops_per_token_after_materialization": (
            dense_mlp_matrix_flops_per_token
        ),
    }


def paired_displacement_pcs(
    fc_states: torch.Tensor,
    proj_states: torch.Tensor,
    *,
    components: int,
    device: str,
) -> dict[str, Any]:
    """Return uncentered Wt-W0 PCs in complete-neuron orientation."""
    if fc_states.ndim != 3 or proj_states.ndim != 3:
        raise ValueError("H69 state tensors must be rank three")
    if fc_states.shape[0] != proj_states.shape[0]:
        raise ValueError("H69 role state counts disagree")
    if fc_states.shape[1:] != proj_states.shape[1:][::-1]:
        raise ValueError("H69 c_fc/c_proj shapes are not paired transposes")
    if components <= 0 or components >= fc_states.shape[0]:
        raise ValueError("invalid H69 component count")

    state_count = int(fc_states.shape[0])
    gram = torch.zeros(
        state_count, state_count, device=device, dtype=torch.float64
    )
    role_inputs = {
        "detector": fc_states,
        "write": proj_states.transpose(1, 2),
    }
    for values in role_inputs.values():
        residual = (
            values.to(device=device, dtype=torch.float32)
            - values[0].to(device=device, dtype=torch.float32)
        ).flatten(1)
        gram += (residual @ residual.T).double()
        del residual
    gram = (gram + gram.T) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    vectors = vectors[:, order]
    selected = vectors[:, :components].float()
    scales = eigenvalues[:components].sqrt().float().clamp_min(1e-20)

    parts: dict[str, torch.Tensor] = {}
    for role, values in role_inputs.items():
        residual = (
            values.to(device=device, dtype=torch.float32)
            - values[0].to(device=device, dtype=torch.float32)
        ).flatten(1)
        part = (selected.T @ residual) / scales[:, None]
        shape = values.shape[1:]
        parts[role] = part.reshape(components, *shape).contiguous()
        del residual, part
    joint_norm = (
        parts["detector"].double().square().flatten(1).sum(1)
        + parts["write"].double().square().flatten(1).sum(1)
    ).sqrt().float().clamp_min(1e-20)
    parts = {
        role: value / joint_norm.view(-1, 1, 1)
        for role, value in parts.items()
    }
    total = float(eigenvalues.sum().clamp_min(1e-30))
    top = eigenvalues[:components].float()
    weights = top / top.sum().clamp_min(1e-30)
    return {
        "detector_w0": fc_states[0].float().contiguous(),
        "write_w0": proj_states[0].transpose(0, 1).float().contiguous(),
        "detector_pcs": parts["detector"],
        "write_pcs": parts["write"],
        "eigenvalues": top,
        "weights": weights,
        "retained_energy_fraction": float(top.double().sum() / max(total, 1e-30)),
        "total_displacement_energy": total,
        "state_count": state_count,
        "step_zero_residual_fro": 0.0,
    }


def load_layer_bundles(
    trajectory_dir: Path,
    *,
    layers: list[int],
    components: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parameters = tuple(
        PARAMETER_TEMPLATE.format(layer=layer, target=target)
        for layer in layers
        for target in ("c_fc", "c_proj")
    )
    states, identity = load_trajectory_inventory(trajectory_dir, parameters)
    bundles = []
    manifest: dict[str, Any] = {
        "run_identity_sha256": identity,
        "layers": layers,
        "components": components,
        "state_count": 239,
        "parameters": {},
    }
    for layer in layers:
        fc_name = PARAMETER_TEMPLATE.format(layer=layer, target="c_fc")
        proj_name = PARAMETER_TEMPLATE.format(layer=layer, target="c_proj")
        bundle = paired_displacement_pcs(
            states[fc_name], states[proj_name],
            components=components, device=device,
        )
        bundle["layer"] = layer
        bundles.append(bundle)
        manifest["parameters"][str(layer)] = {
            "c_fc": fc_name,
            "c_proj": proj_name,
            "c_fc_shape": list(states[fc_name].shape[1:]),
            "c_proj_shape": list(states[proj_name].shape[1:]),
            "retained_energy_fraction": bundle["retained_energy_fraction"],
            "eigenvalues": [float(value) for value in bundle["eigenvalues"]],
            "w0_sha256": {
                "detector": tensor_sha256(bundle["detector_w0"].half()),
                "write": tensor_sha256(bundle["write_w0"].half()),
            },
        }
    del states
    return bundles, manifest


class PairedNeuronDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        width: int,
        shared_width: int,
        latent_width: int,
        deployment_layers: int,
        measured_layers: list[int],
        components: int,
        seed: int,
        linear: bool = False,
        unpaired: bool = False,
    ) -> None:
        super().__init__()
        self.width = width
        self.shared_width = shared_width
        self.latent_width = latent_width
        self.measured_layers = list(measured_layers)
        self.layer_to_row = {
            layer: row for row, layer in enumerate(self.measured_layers)
        }
        self.linear = linear
        self.unpaired = unpaired
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def parameter(shape: tuple[int, ...], fan_in: int) -> torch.nn.Parameter:
            value = torch.randn(shape, generator=generator) / math.sqrt(fan_in)
            return torch.nn.Parameter(value)

        self.p_u = parameter((shared_width, width), width)
        self.p_v = parameter((shared_width, width), width)
        self.q_u = parameter((width, shared_width), shared_width)
        self.q_v = parameter((width, shared_width), shared_width)
        self.c = parameter((shared_width, latent_width), latent_width)
        self.layer_embeddings = torch.nn.Parameter(
            torch.zeros(deployment_layers, shared_width)
        )
        self.codes_u = torch.nn.Parameter(
            0.02 * torch.randn(
                len(measured_layers), components, latent_width,
                generator=generator,
            )
        )
        if unpaired:
            self.codes_v = torch.nn.Parameter(
                0.02 * torch.randn(
                    len(measured_layers), components, latent_width,
                    generator=generator,
                )
            )
        else:
            self.register_parameter("codes_v", None)

    def _hidden(
        self,
        base: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        if self.linear:
            return injection[:, None, :].expand(-1, base.shape[0], -1)
        return F.gelu(base[None, :, :] + injection[:, None, :]) - F.gelu(
            base[None, :, :]
        )

    def predict(
        self,
        layer: int,
        detector_w0: torch.Tensor,
        write_w0: torch.Tensor,
        *,
        neuron_indices: torch.Tensor | None = None,
        zero_codes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.layer_to_row[layer]
        if neuron_indices is not None:
            detector_w0 = detector_w0[neuron_indices]
            write_w0 = write_w0[neuron_indices]
        embedding = self.layer_embeddings[layer]
        base_u = F.linear(detector_w0, self.p_u) + embedding
        base_v = F.linear(write_w0, self.p_v) + embedding
        if self.unpaired:
            codes_u = self.codes_u[row]
            assert self.codes_v is not None
            codes_v = self.codes_v[row]
            injection_u = F.linear(codes_u, self.c)
            injection_v = F.linear(codes_v, self.c)
            hidden_u = self._hidden(base_u, injection_u)
            hidden_v = self._hidden(base_v, injection_v)
        else:
            codes = torch.zeros_like(self.codes_u[row]) if zero_codes else self.codes_u[row]
            injection = F.linear(codes, self.c)
            paired_base = base_u + base_v - embedding
            hidden_u = self._hidden(paired_base, injection)
            hidden_v = hidden_u
        return F.linear(hidden_u, self.q_u), F.linear(hidden_v, self.q_v)

    def compact_state(self, *, deployment_layers: int) -> dict[str, torch.Tensor]:
        return {
            "p_u": self.p_u.detach().half().cpu(),
            "p_v": self.p_v.detach().half().cpu(),
            "q_u": self.q_u.detach().half().cpu(),
            "q_v": self.q_v.detach().half().cpu(),
            "c": self.c.detach().half().cpu(),
            "layer_embeddings": self.layer_embeddings.detach().half().cpu(),
            "live_latents": torch.zeros(
                deployment_layers, self.latent_width, dtype=torch.float16
            ),
        }


def procedural_blind_keys(
    bundles: list[dict[str, Any]], *, seed: int, device: str,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for bundle in bundles:
        copied = dict(bundle)
        for role in ("detector", "write"):
            original = bundle[f"{role}_w0"]
            pseudo = torch.randn(original.shape, generator=generator)
            pseudo *= original.float().std().clamp_min(1e-20)
            copied[f"{role}_w0"] = pseudo.to(device)
        result.append(copied)
    return result


def fit_decoder(
    decoder: PairedNeuronDecoder,
    bundles: list[dict[str, Any]],
    *,
    updates: int,
    neuron_batch: int,
    learning_rate: float,
    betas: tuple[float, float],
    weight_decay: float,
    freeze_decoder: bool,
    progress_offset: int = 0,
) -> list[dict[str, float | int]]:
    if freeze_decoder:
        for name, parameter in decoder.named_parameters():
            parameter.requires_grad_(name.startswith("codes_"))
    parameters = [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, betas=betas, weight_decay=weight_decay
    )
    history = []
    report = {0, updates // 5, updates // 2, 4 * updates // 5, updates - 1}
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=bundles[0]["detector_w0"].device)
        for layer_row, bundle in enumerate(bundles):
            neurons = int(bundle["detector_w0"].shape[0])
            start = (step * neuron_batch + 997 * layer_row) % neurons
            indices = (torch.arange(neuron_batch, device=loss.device) + start) % neurons
            pred_u, pred_v = decoder.predict(
                int(bundle["layer"]),
                bundle["detector_w0"], bundle["write_w0"],
                neuron_indices=indices,
            )
            target_u = bundle["detector_pcs"][:, indices]
            target_v = bundle["write_pcs"][:, indices]
            target_energy = (
                target_u.square().flatten(1).sum(1)
                + target_v.square().flatten(1).sum(1)
            ).clamp_min(1e-20)
            relative = (
                (pred_u - target_u).square().flatten(1).sum(1)
                + (pred_v - target_v).square().flatten(1).sum(1)
            ) / target_energy
            loss = loss + (relative * bundle["weights"]).sum() / len(bundles)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        if step in report:
            row = {
                "iteration": step + 1,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(row)
            print(
                f"iteration {progress_offset + step + 1}: "
                + json.dumps(row, sort_keys=True),
                flush=True,
            )
    return history


def _capture(dot: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dot.square() / (
        prediction.clamp_min(1e-30) * target.clamp_min(1e-30)
    )


@torch.no_grad()
def evaluate_decoder(
    decoder: PairedNeuronDecoder,
    bundles: list[dict[str, Any]],
    *,
    chunk: int,
) -> dict[str, Any]:
    rows = []
    for bundle in bundles:
        count = int(bundle["detector_w0"].shape[0])
        components = int(bundle["detector_pcs"].shape[0])
        device = bundle["detector_w0"].device
        joint_dot = torch.zeros(components, device=device, dtype=torch.float64)
        joint_pred = torch.zeros_like(joint_dot)
        joint_target = torch.zeros_like(joint_dot)
        role_stats = {
            role: {
                key: torch.zeros_like(joint_dot)
                for key in ("dot", "prediction", "target")
            }
            for role in ("detector", "write")
        }
        mse = torch.zeros_like(joint_dot)
        for start in range(0, count, chunk):
            indices = torch.arange(start, min(start + chunk, count), device=device)
            pred_u, pred_v = decoder.predict(
                int(bundle["layer"]), bundle["detector_w0"], bundle["write_w0"],
                neuron_indices=indices,
            )
            targets = {
                "detector": bundle["detector_pcs"][:, indices],
                "write": bundle["write_pcs"][:, indices],
            }
            predictions = {"detector": pred_u, "write": pred_v}
            for role in ("detector", "write"):
                target = targets[role].double().flatten(1)
                prediction = predictions[role].double().flatten(1)
                dot = (target * prediction).sum(1)
                pred_energy = prediction.square().sum(1)
                target_energy = target.square().sum(1)
                role_stats[role]["dot"] += dot
                role_stats[role]["prediction"] += pred_energy
                role_stats[role]["target"] += target_energy
                joint_dot += dot
                joint_pred += pred_energy
                joint_target += target_energy
                mse += (target - prediction).square().sum(1)
        captures = _capture(joint_dot, joint_pred, joint_target).clamp(0, 1)
        relative_mse = mse / joint_target.clamp_min(1e-30)
        weights = bundle["weights"].double()
        role_capture = {
            role: [
                float(value) for value in _capture(
                    stats["dot"], stats["prediction"], stats["target"]
                ).clamp(0, 1)
            ]
            for role, stats in role_stats.items()
        }
        rows.append({
            "layer": int(bundle["layer"]),
            "weighted_top16_capture": float((captures * weights).sum()),
            "minimum_pc_capture": float(captures.min()),
            "median_pc_capture": float(captures.median()),
            "component_captures": [float(value) for value in captures],
            "component_relative_mse": [float(value) for value in relative_mse],
            "detector_component_captures": role_capture["detector"],
            "write_component_captures": role_capture["write"],
            "detector_weighted_capture": float(
                (torch.tensor(role_capture["detector"], dtype=torch.float64) * weights.cpu()).sum()
            ),
            "write_weighted_capture": float(
                (torch.tensor(role_capture["write"], dtype=torch.float64) * weights.cpu()).sum()
            ),
            "retained_energy_fraction": float(bundle["retained_energy_fraction"]),
        })
    return {
        "rows": rows,
        "minimum_weighted_capture": min(row["weighted_top16_capture"] for row in rows),
        "minimum_pc_capture": min(row["minimum_pc_capture"] for row in rows),
    }


def compact_payload(
    decoder: PairedNeuronDecoder, accounting: dict[str, Any],
) -> dict[str, Any]:
    tensors = decoder.compact_state(deployment_layers=12)
    payload_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise ValueError(
            f"H69 compact payload is {payload_bytes}, expected "
            f"{accounting['total_checkpoint_payload_bytes']}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "accounting": accounting,
        "accounted_payload_bytes": payload_bytes,
        "tensors": tensors,
    }


def synthetic_self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(69)
    dev = torch.device(device)
    states = 9
    hidden, width = 12, 8
    fc = torch.randn(states, hidden, width)
    proj = torch.randn(states, width, hidden)
    fc[0].zero_()
    proj[0].zero_()
    bundle = paired_displacement_pcs(fc, proj, components=3, device=device)
    joint_norm = (
        bundle["detector_pcs"].square().flatten(1).sum(1)
        + bundle["write_pcs"].square().flatten(1).sum(1)
    )
    torch.testing.assert_close(joint_norm, torch.ones_like(joint_norm), atol=2e-5, rtol=2e-5)
    decoder = PairedNeuronDecoder(
        width=width, shared_width=6, latent_width=3,
        deployment_layers=12, measured_layers=[0], components=3,
        seed=7,
    ).to(dev)
    zeros_u, zeros_v = decoder.predict(
        0, bundle["detector_w0"].to(dev), bundle["write_w0"].to(dev),
        zero_codes=True,
    )
    if int(torch.count_nonzero(zeros_u)) or int(torch.count_nonzero(zeros_v)):
        raise AssertionError("H69 zero-state parent is not bit-exact")
    predicted_u, predicted_v = decoder.predict(
        0, bundle["detector_w0"].to(dev), bundle["write_w0"].to(dev)
    )
    (predicted_u.square().mean() + predicted_v.square().mean()).backward()
    for name in (
        "p_u", "p_v", "q_u", "q_v", "c", "layer_embeddings", "codes_u"
    ):
        parameter = getattr(decoder, name)
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing finite H69 gradient: {name}")
    accounting = deployment_accounting()
    if accounting["total_fp16_values"] != 545_792:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "joint_pc_norms": [float(value) for value in joint_norm],
        "zero_state_nonzero": 0,
        "accounting": accounting,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(args.device), sort_keys=True))
        return
    if args.plan is None or args.trajectory_dir is None or args.output is None:
        parser.error("--plan, --trajectory-dir, and --output are required")

    started = time.time()
    plan = json.loads(args.plan.read_text())
    layers = [int(value) for value in plan["inputs"]["required_layers"]]
    components = int(plan["inputs"]["top_components"])
    fit = plan["fit"]
    transform = plan["transformation"]
    torch.manual_seed(int(fit["seed"]))
    if args.device.startswith("cuda"):
        # CUDA_VISIBLE_DEVICES isolates the admitted card.  Passing an explicit
        # index before lazy CUDA initialization is rejected by this PRO6 torch
        # build, while the current-device form initializes correctly.
        torch.cuda.reset_peak_memory_stats()
    bundles, input_manifest = load_layer_bundles(
        args.trajectory_dir, layers=layers, components=components,
        device=args.device,
    )
    for bundle in bundles:
        for key in (
            "detector_w0", "write_w0", "detector_pcs", "write_pcs",
            "weights", "eigenvalues",
        ):
            bundle[key] = bundle[key].to(args.device)

    arms: dict[str, dict[str, Any]] = {}
    trained_decoders: dict[str, PairedNeuronDecoder] = {}
    arm_specs = {
        "paired_nonlinear": {"linear": False, "blind": False, "freeze": False, "unpaired": False},
        "equal_state_linear": {"linear": True, "blind": False, "freeze": False, "unpaired": False},
        "w0_blind": {"linear": False, "blind": True, "freeze": False, "unpaired": False},
        "fixed_random_decoder": {"linear": False, "blind": False, "freeze": True, "unpaired": False},
        "unpaired_optimistic": {"linear": False, "blind": False, "freeze": False, "unpaired": True},
    }
    for arm_index, (name, spec) in enumerate(arm_specs.items()):
        active_bundles = (
            procedural_blind_keys(
                bundles, seed=int(fit["seed"]) + 10_000_019, device=args.device
            ) if spec["blind"] else bundles
        )
        decoder = PairedNeuronDecoder(
            width=int(bundles[0]["detector_w0"].shape[1]),
            shared_width=int(transform["shared_width"]),
            latent_width=int(transform["latent_width"]),
            deployment_layers=int(transform["deployment_layer_count"]),
            measured_layers=layers,
            components=components,
            seed=int(fit["seed"]) + arm_index,
            linear=bool(spec["linear"]),
            unpaired=bool(spec["unpaired"]),
        ).to(args.device)
        history = fit_decoder(
            decoder, active_bundles,
            updates=int(fit["updates"]),
            neuron_batch=int(fit["neuron_batch_per_layer"]),
            learning_rate=float(fit["learning_rate"]),
            betas=(float(fit["betas"][0]), float(fit["betas"][1])),
            weight_decay=float(fit["weight_decay"]),
            freeze_decoder=bool(spec["freeze"]),
            progress_offset=arm_index * int(fit["updates"]),
        )
        metrics = evaluate_decoder(
            decoder, active_bundles,
            chunk=int(fit["evaluation_neuron_chunk"]),
        )
        arms[name] = {"history": history, "metrics": metrics}
        trained_decoders[name] = decoder
        print(json.dumps({"arm": name, "metrics": metrics}, sort_keys=True), flush=True)

    candidate = arms["paired_nonlinear"]["metrics"]
    linear = arms["equal_state_linear"]["metrics"]
    blind = arms["w0_blind"]["metrics"]
    linear_by_layer = {row["layer"]: row for row in linear["rows"]}
    blind_by_layer = {row["layer"]: row for row in blind["rows"]}
    margins = []
    for row in candidate["rows"]:
        layer = row["layer"]
        margins.append({
            "layer": layer,
            "candidate_minus_equal_state_linear": row["weighted_top16_capture"]
            - linear_by_layer[layer]["weighted_top16_capture"],
            "candidate_minus_w0_blind": row["weighted_top16_capture"]
            - blind_by_layer[layer]["weighted_top16_capture"],
        })
    zero_nonzero = 0
    zero_max_abs = 0.0
    with torch.no_grad():
        decoder = trained_decoders["paired_nonlinear"]
        for bundle in bundles:
            u, v = decoder.predict(
                int(bundle["layer"]), bundle["detector_w0"], bundle["write_w0"],
                neuron_indices=torch.arange(
                    min(64, bundle["detector_w0"].shape[0]), device=args.device
                ),
                zero_codes=True,
            )
            zero_nonzero += int(torch.count_nonzero(u)) + int(torch.count_nonzero(v))
            zero_max_abs = max(
                zero_max_abs,
                float(u.abs().max()),
                float(v.abs().max()),
            )
    gates = plan["gates"]
    gate_outcomes = {
        "weighted_capture_every_layer": candidate["minimum_weighted_capture"]
        >= float(gates["weighted_capture_every_layer_minimum"]),
        "every_pc_capture": candidate["minimum_pc_capture"]
        >= float(gates["every_pc_capture_minimum"]),
        "linear_margin_every_layer": min(
            row["candidate_minus_equal_state_linear"] for row in margins
        ) >= float(gates["candidate_minus_equal_state_linear_every_layer_minimum"]),
        "w0_blind_margin_every_layer": min(
            row["candidate_minus_w0_blind"] for row in margins
        ) >= float(gates["candidate_minus_w0_blind_every_layer_minimum"]),
        "zero_state_exact": (
            zero_nonzero == 0
            and zero_max_abs <= float(gates["zero_state_max_abs"])
        ),
    }
    finite = all(
        math.isfinite(float(row["weighted_top16_capture"]))
        and math.isfinite(float(row["minimum_pc_capture"]))
        for arm in arms.values() for row in arm["metrics"]["rows"]
    )
    gate_outcomes["all_values_finite"] = finite
    passed = all(gate_outcomes.values())
    accounting = deployment_accounting(
        width=int(bundles[0]["detector_w0"].shape[1]),
        shared_width=int(transform["shared_width"]),
        latent_width=int(transform["latent_width"]),
        layers=int(transform["deployment_layer_count"]),
        hidden_width=int(bundles[0]["detector_w0"].shape[0]),
    )
    payload = compact_payload(trained_decoders["paired_nonlinear"], accounting)

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output / "compact_checkpoint.pt"
    torch.save(payload, checkpoint_path)
    per_layer_path = args.output / "per_layer.csv"
    flat_rows = []
    for arm, result in arms.items():
        for row in result["metrics"]["rows"]:
            flat_rows.append({"arm": arm, **row})
    write_csv(per_layer_path, flat_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "classification": "PASSED" if passed else "REJECTED",
        "gate_outcomes": gate_outcomes,
        "margins": margins,
        "zero_state_nonzero_values": zero_nonzero,
        "zero_state_max_abs": zero_max_abs,
        "accounting": accounting,
        "inputs": input_manifest,
        "arms": arms,
        "limitations": [
            "This is an optimistic top-16 image-capacity audit, not CE training.",
            "Offline PC codes are nuisance fit coordinates and are not deployed state.",
            "One optimizer path does not identify the global solution manifold.",
        ],
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    runtime = time.time() - started
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "classification": result["classification"],
        "repository": {
            "git_commit": git_commit(REPO_ROOT),
            "dirty": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--short"], text=True
            ).splitlines(),
        },
        "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "plan": {"path": str(args.plan.resolve()), "sha256": sha256_file(args.plan)},
        "trajectory": {
            "path": str(args.trajectory_dir.resolve()),
            "identity_sha256": input_manifest["run_identity_sha256"],
            "file_count": len(list(args.trajectory_dir.glob("step_*.pt"))),
        },
        "runtime_seconds": runtime,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda") else 0
        ),
        "outputs": {
            "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "per_layer": {"path": str(per_layer_path), "sha256": sha256_file(per_layer_path)},
        },
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "classification": result["classification"],
        "gate_outcomes": gate_outcomes,
        "result_sha256": metadata["outputs"]["result"]["sha256"],
        "runtime_seconds": runtime,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
