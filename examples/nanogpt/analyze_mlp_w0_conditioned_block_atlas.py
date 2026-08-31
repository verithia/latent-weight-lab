#!/usr/bin/env python3
"""Frozen H70 W0-conditioned block-atlas representation audit."""
from __future__ import annotations

import argparse
import json
import math
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

from examples.nanogpt.analyze_mlp_initialization_conditioned_paired_manifold import (
    _capture,
    git_commit,
    load_layer_bundles,
    sha256_file,
    write_csv,
)


SCHEMA_VERSION = "nanogpt_mlp_w0_conditioned_block_atlas_v1"


def deployment_accounting(
    *,
    shared_width: int = 2048,
    block_width: int = 32,
    latent_width: int = 16,
    width: int = 768,
    hidden_width: int = 3072,
    layers: int = 12,
) -> dict[str, int | float]:
    positions = width // block_width
    key = shared_width * (2 * block_width)
    injection = shared_width * latent_width
    bases = 2 * shared_width * block_width
    layer_embeddings = layers * shared_width
    position_embeddings = positions * shared_width
    live_latents = layers * latent_width
    total = (
        key + injection + bases + layer_embeddings
        + position_embeddings + live_latents
    )
    dense_values = layers * 2 * hidden_width * width
    paired_blocks = layers * hidden_width * positions
    static_flops = paired_blocks * 2 * (2 * block_width) * shared_width
    refresh_flops = (
        layers * 2 * shared_width * latent_width
        + paired_blocks * 4 * shared_width * block_width
    )
    return {
        "shared_key_values": key,
        "latent_injection_values": injection,
        "shared_local_basis_values": bases,
        "layer_embedding_values": layer_embeddings,
        "position_embedding_values": position_embeddings,
        "live_latent_values": live_latents,
        "total_fp16_values": total,
        "total_checkpoint_payload_bytes": 2 * total,
        "dense_replaced_mlp_fp16_values": dense_values,
        "dense_replaced_mlp_fp16_bytes": 2 * dense_values,
        "checkpoint_byte_fraction": total / dense_values,
        "persistent_w0_bytes": 0,
        "persistent_empirical_basis_bytes": 0,
        "persistent_row_or_block_code_bytes": 0,
        "static_key_matrix_flops": static_flops,
        "live_latent_refresh_matrix_flops_with_cached_keys": refresh_flops,
        "live_latent_refresh_matrix_flops_if_keys_recomputed": (
            static_flops + refresh_flops
        ),
        "dense_mlp_matrix_flops_per_token_after_materialization": (
            layers * 4 * hidden_width * width
        ),
        "generated_dense_endpoint_bytes": 2 * dense_values,
        "static_key_cache_bytes_used_by_this_audit": 0,
    }


def blockify_bundle(bundle: dict[str, Any], *, block_width: int) -> dict[str, Any]:
    copied = dict(bundle)
    width = int(bundle["detector_w0"].shape[1])
    if width % block_width:
        raise ValueError("H70 block width does not divide the MLP width")
    positions = width // block_width
    copied["detector_w0_blocks"] = bundle["detector_w0"].reshape(-1, block_width)
    copied["write_w0_blocks"] = bundle["write_w0"].reshape(-1, block_width)
    copied["detector_pc_blocks"] = bundle["detector_pcs"].reshape(
        bundle["detector_pcs"].shape[0], -1, block_width
    )
    copied["write_pc_blocks"] = bundle["write_pcs"].reshape(
        bundle["write_pcs"].shape[0], -1, block_width
    )
    copied["block_positions"] = torch.arange(
        copied["detector_w0_blocks"].shape[0]
    ) % positions
    return copied


class BlockAtlasDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        block_width: int,
        shared_width: int,
        latent_width: int,
        deployment_layers: int,
        positions: int,
        measured_layers: list[int],
        components: int,
        seed: int,
        linear: bool = False,
        unpaired: bool = False,
    ) -> None:
        super().__init__()
        self.block_width = block_width
        self.shared_width = shared_width
        self.latent_width = latent_width
        self.measured_layers = list(measured_layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(measured_layers)}
        self.linear = linear
        self.unpaired = unpaired
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def parameter(shape: tuple[int, ...], fan_in: int) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                torch.randn(shape, generator=generator) / math.sqrt(fan_in)
            )

        self.a = parameter((shared_width, 2 * block_width), 2 * block_width)
        self.c = parameter((shared_width, latent_width), latent_width)
        self.d_u = parameter((block_width, shared_width), shared_width)
        self.d_v = parameter((block_width, shared_width), shared_width)
        self.layer_embeddings = torch.nn.Parameter(
            torch.zeros(deployment_layers, shared_width)
        )
        self.position_embeddings = torch.nn.Parameter(
            torch.zeros(positions, shared_width)
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

    def _base(
        self,
        layer: int,
        detector_blocks: torch.Tensor,
        write_blocks: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        paired = torch.cat((detector_blocks, write_blocks), dim=-1)
        return (
            F.linear(paired, self.a)
            + self.layer_embeddings[layer]
            + self.position_embeddings[positions]
        )

    def _hidden(self, base: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        injection = F.linear(codes, self.c)
        if self.linear:
            return injection[:, None, :].expand(-1, base.shape[0], -1)
        return F.gelu(base[None] + injection[:, None]) - F.gelu(base[None])

    def predict(
        self,
        layer: int,
        detector_blocks: torch.Tensor,
        write_blocks: torch.Tensor,
        positions: torch.Tensor,
        *,
        block_indices: torch.Tensor | None = None,
        zero_codes: bool = False,
        one_code: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_indices is not None:
            detector_blocks = detector_blocks[block_indices]
            write_blocks = write_blocks[block_indices]
            positions = positions[block_indices]
        row = self.layer_to_row[layer]
        codes_u = self.codes_u[row]
        if one_code:
            codes_u = codes_u[:1]
        if zero_codes:
            codes_u = torch.zeros_like(codes_u)
        base = self._base(layer, detector_blocks, write_blocks, positions)
        hidden_u = self._hidden(base, codes_u)
        if self.unpaired:
            assert self.codes_v is not None
            codes_v = self.codes_v[row]
            if one_code:
                codes_v = codes_v[:1]
            if zero_codes:
                codes_v = torch.zeros_like(codes_v)
            hidden_v = self._hidden(base, codes_v)
        else:
            hidden_v = hidden_u
        return F.linear(hidden_u, self.d_u), F.linear(hidden_v, self.d_v)

    def compact_state(self, *, deployment_layers: int) -> dict[str, torch.Tensor]:
        return {
            "a": self.a.detach().half().cpu(),
            "c": self.c.detach().half().cpu(),
            "d_u": self.d_u.detach().half().cpu(),
            "d_v": self.d_v.detach().half().cpu(),
            "layer_embeddings": self.layer_embeddings.detach().half().cpu(),
            "position_embeddings": self.position_embeddings.detach().half().cpu(),
            "live_latents": torch.zeros(
                deployment_layers, self.latent_width, dtype=torch.float16
            ),
        }


def procedural_blind_blocks(
    bundles: list[dict[str, Any]], *, seed: int, device: str
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for bundle in bundles:
        copied = dict(bundle)
        for role in ("detector", "write"):
            key = f"{role}_w0_blocks"
            original = bundle[key]
            pseudo = torch.randn(original.shape, generator=generator)
            pseudo *= float(original.float().std().clamp_min(1e-20))
            copied[key] = pseudo.to(device)
        result.append(copied)
    return result


def fit_decoder(
    decoder: BlockAtlasDecoder,
    bundles: list[dict[str, Any]],
    *,
    updates: int,
    block_batch: int,
    learning_rate: float,
    betas: tuple[float, float],
    weight_decay: float,
    freeze_basis: bool,
    progress_offset: int,
) -> list[dict[str, float | int]]:
    if freeze_basis:
        decoder.d_u.requires_grad_(False)
        decoder.d_v.requires_grad_(False)
    parameters = [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, betas=betas, weight_decay=weight_decay
    )
    history = []
    report = {0, updates // 5, updates // 2, 4 * updates // 5, updates - 1}
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=bundles[0]["detector_w0_blocks"].device)
        for row, bundle in enumerate(bundles):
            count = int(bundle["detector_w0_blocks"].shape[0])
            start = (step * block_batch + 8191 * row) % count
            indices = (torch.arange(block_batch, device=loss.device) + start) % count
            pred_u, pred_v = decoder.predict(
                int(bundle["layer"]),
                bundle["detector_w0_blocks"], bundle["write_w0_blocks"],
                bundle["block_positions"], block_indices=indices,
            )
            target_u = bundle["detector_pc_blocks"][:, indices]
            target_v = bundle["write_pc_blocks"][:, indices]
            energy = (
                target_u.square().flatten(1).sum(1)
                + target_v.square().flatten(1).sum(1)
            ).clamp_min(1e-20)
            relative = (
                (pred_u - target_u).square().flatten(1).sum(1)
                + (pred_v - target_v).square().flatten(1).sum(1)
            ) / energy
            loss = loss + relative.mean() / len(bundles)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        if step in report:
            item = {
                "iteration": step + 1,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(item)
            print(
                f"iteration {progress_offset + step + 1}: "
                + json.dumps(item, sort_keys=True),
                flush=True,
            )
    return history


@torch.no_grad()
def evaluate_decoder(
    decoder: BlockAtlasDecoder,
    bundles: list[dict[str, Any]],
    *,
    chunk: int,
) -> dict[str, Any]:
    rows = []
    for bundle in bundles:
        components = int(bundle["detector_pc_blocks"].shape[0])
        count = int(bundle["detector_w0_blocks"].shape[0])
        device = bundle["detector_w0_blocks"].device
        joint = {
            key: torch.zeros(components, device=device, dtype=torch.float64)
            for key in ("dot", "prediction", "target")
        }
        role_stats = {
            role: {
                key: torch.zeros_like(joint["dot"])
                for key in ("dot", "prediction", "target")
            }
            for role in ("detector", "write")
        }
        for start in range(0, count, chunk):
            indices = torch.arange(start, min(start + chunk, count), device=device)
            pred_u, pred_v = decoder.predict(
                int(bundle["layer"]),
                bundle["detector_w0_blocks"], bundle["write_w0_blocks"],
                bundle["block_positions"], block_indices=indices,
            )
            for role, prediction, target in (
                ("detector", pred_u, bundle["detector_pc_blocks"][:, indices]),
                ("write", pred_v, bundle["write_pc_blocks"][:, indices]),
            ):
                prediction = prediction.double().flatten(1)
                target = target.double().flatten(1)
                dot = (prediction * target).sum(1)
                pred_energy = prediction.square().sum(1)
                target_energy = target.square().sum(1)
                role_stats[role]["dot"] += dot
                role_stats[role]["prediction"] += pred_energy
                role_stats[role]["target"] += target_energy
                joint["dot"] += dot
                joint["prediction"] += pred_energy
                joint["target"] += target_energy
        captures = _capture(
            joint["dot"], joint["prediction"], joint["target"]
        ).clamp(0, 1)
        weights = bundle["weights"].double()
        role_captures = {
            role: _capture(
                stats["dot"], stats["prediction"], stats["target"]
            ).clamp(0, 1)
            for role, stats in role_stats.items()
        }
        rows.append({
            "layer": int(bundle["layer"]),
            "weighted_top16_capture": float((captures * weights).sum()),
            "uniform_mean_capture": float(captures.mean()),
            "minimum_pc_capture": float(captures.min()),
            "median_pc_capture": float(captures.median()),
            "component_captures": [float(value) for value in captures],
            "detector_weighted_capture": float(
                (role_captures["detector"] * weights).sum()
            ),
            "write_weighted_capture": float(
                (role_captures["write"] * weights).sum()
            ),
            "retained_energy_fraction": float(bundle["retained_energy_fraction"]),
        })
    return {
        "rows": rows,
        "minimum_weighted_capture": min(x["weighted_top16_capture"] for x in rows),
        "minimum_pc_capture": min(x["minimum_pc_capture"] for x in rows),
    }


@torch.no_grad()
def basis_diagnostics(decoder: BlockAtlasDecoder) -> dict[str, Any]:
    result = {}
    for name in ("d_u", "d_v"):
        singular = torch.linalg.svdvals(getattr(decoder, name).float()).cpu()
        threshold = float(singular.max()) * max(getattr(decoder, name).shape) * 1e-6
        result[name] = {
            "numerical_rank": int((singular > threshold).sum()),
            "largest_singular_value": float(singular.max()),
            "smallest_singular_value": float(singular.min()),
            "condition_number": float(singular.max() / singular.min().clamp_min(1e-30)),
        }
    return result


@torch.no_grad()
def benchmark_one_layer_refresh(
    decoder: BlockAtlasDecoder, bundle: dict[str, Any], *, chunk: int
) -> float:
    device = bundle["detector_w0_blocks"].device
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    checksum = torch.zeros((), device=device)
    count = int(bundle["detector_w0_blocks"].shape[0])
    for start in range(0, count, chunk):
        indices = torch.arange(start, min(start + chunk, count), device=device)
        u, v = decoder.predict(
            int(bundle["layer"]),
            bundle["detector_w0_blocks"], bundle["write_w0_blocks"],
            bundle["block_positions"], block_indices=indices, one_code=True,
        )
        checksum += u.sum() + v.sum()
    torch.cuda.synchronize(device)
    if not torch.isfinite(checksum):
        raise ValueError("nonfinite H70 refresh checksum")
    return time.perf_counter() - started


def compact_payload(
    decoder: BlockAtlasDecoder, accounting: dict[str, Any]
) -> dict[str, Any]:
    tensors = decoder.compact_state(deployment_layers=12)
    payload_bytes = sum(x.numel() * x.element_size() for x in tensors.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise ValueError(f"H70 payload {payload_bytes} != {accounting}")
    return {
        "schema_version": SCHEMA_VERSION,
        "accounting": accounting,
        "accounted_payload_bytes": payload_bytes,
        "tensors": tensors,
    }


def synthetic_self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(70)
    decoder = BlockAtlasDecoder(
        block_width=4, shared_width=12, latent_width=3,
        deployment_layers=12, positions=2, measured_layers=[0],
        components=3, seed=7,
    ).to(device)
    detector = torch.randn(10, 4, device=device)
    write = torch.randn(10, 4, device=device)
    positions = torch.arange(10, device=device) % 2
    zero_u, zero_v = decoder.predict(
        0, detector, write, positions, zero_codes=True
    )
    if int(torch.count_nonzero(zero_u)) or int(torch.count_nonzero(zero_v)):
        raise AssertionError("H70 zero-state identity failed")
    u, v = decoder.predict(0, detector, write, positions)
    (u.square().mean() + v.square().mean()).backward()
    for name in (
        "a", "c", "d_u", "d_v", "layer_embeddings",
        "position_embeddings", "codes_u",
    ):
        parameter = getattr(decoder, name)
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing H70 gradient: {name}")
    accounting = deployment_accounting()
    if accounting["total_fp16_values"] != 368_832:
        raise AssertionError(accounting)
    return {"status": "passed", "accounting": accounting}


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
    fit = plan["fit"]
    transform = plan["transformation"]
    layers = [int(value) for value in plan["inputs"]["required_layers"]]
    components = int(plan["inputs"]["top_components"])
    block_width = int(transform["block_width"])
    shared_width = int(transform["shared_width"])
    latent_width = int(transform["latent_width"])
    torch.manual_seed(int(fit["seed"]))
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    raw_bundles, input_manifest = load_layer_bundles(
        args.trajectory_dir, layers=layers, components=components,
        device=args.device,
    )
    if input_manifest["run_identity_sha256"] != plan["inputs"]["run_identity_sha256"]:
        raise ValueError("H70 run identity mismatch")
    bundles = [blockify_bundle(x, block_width=block_width) for x in raw_bundles]
    for bundle in bundles:
        for key in (
            "detector_w0_blocks", "write_w0_blocks", "detector_pc_blocks",
            "write_pc_blocks", "block_positions", "weights", "eigenvalues",
        ):
            bundle[key] = bundle[key].to(args.device)
    del raw_bundles

    specs = {
        "w0_conditioned_block_atlas": {
            "linear": False, "blind": False, "freeze_basis": False,
            "unpaired": False,
        },
        "equal_state_linear": {
            "linear": True, "blind": False, "freeze_basis": False,
            "unpaired": False,
        },
        "w0_blind": {
            "linear": False, "blind": True, "freeze_basis": False,
            "unpaired": False,
        },
        "fixed_random_local_basis": {
            "linear": False, "blind": False, "freeze_basis": True,
            "unpaired": False,
        },
        "unpaired_optimistic": {
            "linear": False, "blind": False, "freeze_basis": False,
            "unpaired": True,
        },
    }
    arms: dict[str, Any] = {}
    decoders: dict[str, BlockAtlasDecoder] = {}
    updates = int(fit["updates"])
    for arm_index, (name, spec) in enumerate(specs.items()):
        active = procedural_blind_blocks(
            bundles, seed=int(fit["seed"]) + 10_000_019, device=args.device
        ) if spec["blind"] else bundles
        decoder = BlockAtlasDecoder(
            block_width=block_width, shared_width=shared_width,
            latent_width=latent_width,
            deployment_layers=int(transform["deployment_layer_count"]),
            positions=int(transform["block_position_count"]),
            measured_layers=layers, components=components,
            seed=int(fit["seed"]) + arm_index,
            linear=bool(spec["linear"]), unpaired=bool(spec["unpaired"]),
        ).to(args.device)
        history = fit_decoder(
            decoder, active, updates=updates,
            block_batch=int(fit["block_batch_per_layer"]),
            learning_rate=float(fit["learning_rate"]),
            betas=(float(fit["betas"][0]), float(fit["betas"][1])),
            weight_decay=float(fit["weight_decay"]),
            freeze_basis=bool(spec["freeze_basis"]),
            progress_offset=arm_index * updates,
        )
        metrics = evaluate_decoder(
            decoder, active, chunk=int(fit["evaluation_block_chunk"])
        )
        arms[name] = {"history": history, "metrics": metrics}
        decoders[name] = decoder
        print(json.dumps({"arm": name, "metrics": metrics}, sort_keys=True), flush=True)

    candidate = arms["w0_conditioned_block_atlas"]["metrics"]
    linear = {x["layer"]: x for x in arms["equal_state_linear"]["metrics"]["rows"]}
    blind = {x["layer"]: x for x in arms["w0_blind"]["metrics"]["rows"]}
    margins = [{
        "layer": row["layer"],
        "candidate_minus_equal_state_linear": (
            row["weighted_top16_capture"]
            - linear[row["layer"]]["weighted_top16_capture"]
        ),
        "candidate_minus_w0_blind": (
            row["weighted_top16_capture"]
            - blind[row["layer"]]["weighted_top16_capture"]
        ),
    } for row in candidate["rows"]]
    decoder = decoders["w0_conditioned_block_atlas"]
    zero_nonzero = 0
    zero_max_abs = 0.0
    with torch.no_grad():
        for bundle in bundles:
            indices = torch.arange(64, device=args.device)
            u, v = decoder.predict(
                int(bundle["layer"]),
                bundle["detector_w0_blocks"], bundle["write_w0_blocks"],
                bundle["block_positions"], block_indices=indices,
                zero_codes=True,
            )
            zero_nonzero += int(torch.count_nonzero(u)) + int(torch.count_nonzero(v))
            zero_max_abs = max(zero_max_abs, float(u.abs().max()), float(v.abs().max()))
    basis = basis_diagnostics(decoder)
    gates = plan["gates"]
    gate_outcomes = {
        "weighted_capture_every_layer": candidate["minimum_weighted_capture"]
        >= float(gates["weighted_capture_every_layer_minimum"]),
        "every_pc_capture": candidate["minimum_pc_capture"]
        >= float(gates["every_pc_capture_minimum"]),
        "linear_margin_every_layer": min(
            x["candidate_minus_equal_state_linear"] for x in margins
        ) >= float(gates["candidate_minus_equal_state_linear_every_layer_minimum"]),
        "w0_blind_margin_every_layer": min(
            x["candidate_minus_w0_blind"] for x in margins
        ) >= float(gates["candidate_minus_w0_blind_every_layer_minimum"]),
        "local_basis_full_rank": min(
            basis["d_u"]["numerical_rank"], basis["d_v"]["numerical_rank"]
        ) >= int(gates["local_basis_numerical_rank_minimum"]),
        "zero_state_exact": zero_nonzero == 0 and zero_max_abs <= float(
            gates["zero_state_max_abs"]
        ),
    }
    gate_outcomes["all_values_finite"] = all(
        math.isfinite(float(row["weighted_top16_capture"]))
        and math.isfinite(float(row["minimum_pc_capture"]))
        for arm in arms.values() for row in arm["metrics"]["rows"]
    )
    classification = "PASSED" if all(gate_outcomes.values()) else "REJECTED"
    accounting = deployment_accounting(shared_width=shared_width, block_width=block_width)
    if accounting["total_checkpoint_payload_bytes"] != plan["persistent_state"]["total_bytes"]:
        raise ValueError("H70 accounting disagrees with frozen plan")
    refresh_seconds = benchmark_one_layer_refresh(
        decoder, bundles[0], chunk=int(fit["evaluation_block_chunk"])
    )
    payload = compact_payload(decoder, accounting)

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / "compact_checkpoint.pt"
    torch.save(payload, checkpoint)
    per_layer = args.output / "per_layer.csv"
    write_csv(per_layer, [
        {"arm": name, **row}
        for name, arm in arms.items() for row in arm["metrics"]["rows"]
    ])
    result = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "gate_outcomes": gate_outcomes,
        "margins": margins,
        "basis_diagnostics": basis,
        "zero_state_nonzero_values": zero_nonzero,
        "zero_state_max_abs": zero_max_abs,
        "accounting": accounting,
        "training_state": {
            "offline_nuisance_code_fp32_values": len(layers) * components * latent_width,
            "static_key_cache_bytes_used": 0,
            "one_layer_uncached_refresh_seconds": refresh_seconds,
            "twelve_layer_uncached_refresh_seconds_linear_extrapolation": 12 * refresh_seconds,
        },
        "inputs": input_manifest,
        "arms": arms,
        "limitations": [
            "This is an optimistic top-16 image-capacity audit, not CE training.",
            "Offline PC codes are nuisance coordinates and are not deployed state.",
            "One optimizer path does not identify the global solution manifold."
        ],
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
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
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
        ),
        "outputs": {},
    }
    metadata_path = args.output / "metadata.json"
    for name, path in (
        ("result", result_path), ("checkpoint", checkpoint), ("per_layer", per_layer)
    ):
        metadata["outputs"][name] = {"path": str(path), "sha256": sha256_file(path)}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "classification": classification,
        "gate_outcomes": gate_outcomes,
        "result_sha256": metadata["outputs"]["result"]["sha256"],
        "runtime_seconds": metadata["runtime_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
