#!/usr/bin/env python3
"""Frozen H71 whole-neuron-context/local-write representation audit."""
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
    git_commit,
    load_layer_bundles,
    sha256_file,
    write_csv,
)
from examples.nanogpt.analyze_mlp_w0_conditioned_block_atlas import (
    basis_diagnostics,
    benchmark_one_layer_refresh,
    blockify_bundle,
    evaluate_decoder,
    fit_decoder,
    procedural_blind_blocks,
)


SCHEMA_VERSION = "nanogpt_mlp_whole_neuron_context_local_write_v1"


def deployment_accounting(
    *,
    shared_width: int = 1024,
    context_width: int = 128,
    block_width: int = 32,
    latent_width: int = 16,
    width: int = 768,
    hidden_width: int = 3072,
    layers: int = 12,
) -> dict[str, int | float]:
    positions = width // block_width
    paired_width = 2 * width
    paired_block_width = 2 * block_width
    complete_neuron_frame = context_width * paired_width
    local_key = shared_width * paired_block_width
    context_lift = shared_width * context_width
    injection = shared_width * latent_width
    layer_embeddings = layers * shared_width
    position_embeddings = positions * shared_width
    live_latents = layers * latent_width
    total = (
        complete_neuron_frame
        + local_key
        + context_lift
        + injection
        + layer_embeddings
        + position_embeddings
        + live_latents
    )
    dense_values = layers * 2 * hidden_width * width
    paired_neurons = layers * hidden_width
    paired_blocks = paired_neurons * positions
    complete_neuron_flops = paired_neurons * 2 * paired_width * context_width
    local_key_flops = paired_blocks * 2 * paired_block_width * shared_width
    context_lift_flops = paired_neurons * 2 * context_width * shared_width
    static_flops = complete_neuron_flops + local_key_flops + context_lift_flops
    refresh_flops = (
        layers * 2 * shared_width * latent_width
        + paired_blocks * 4 * shared_width * block_width
    )
    return {
        "complete_neuron_frame_values": complete_neuron_frame,
        "local_key_values": local_key,
        "context_lift_values": context_lift,
        "latent_injection_values": injection,
        "layer_embedding_values": layer_embeddings,
        "position_embedding_values": position_embeddings,
        "live_latent_values": live_latents,
        "procedural_local_decoder_values": 0,
        "total_fp16_values": total,
        "total_checkpoint_payload_bytes": 2 * total,
        "dense_replaced_mlp_fp16_values": dense_values,
        "dense_replaced_mlp_fp16_bytes": 2 * dense_values,
        "checkpoint_byte_fraction": total / dense_values,
        "persistent_w0_bytes": 0,
        "persistent_empirical_basis_bytes": 0,
        "persistent_row_or_block_code_bytes": 0,
        "complete_neuron_measurement_matrix_flops": complete_neuron_flops,
        "local_key_matrix_flops": local_key_flops,
        "context_lift_matrix_flops": context_lift_flops,
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


class WholeNeuronContextLocalWriteDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        width: int,
        block_width: int,
        shared_width: int,
        context_width: int,
        latent_width: int,
        deployment_layers: int,
        measured_layers: list[int],
        components: int,
        seed: int,
        context_mode: str = "exact",
        learned_local_decoder: bool = False,
    ) -> None:
        super().__init__()
        if width % block_width:
            raise ValueError("H71 block width must divide the MLP width")
        if context_mode not in {"exact", "none", "shuffled"}:
            raise ValueError(f"invalid H71 context mode: {context_mode}")
        self.width = width
        self.block_width = block_width
        self.positions = width // block_width
        self.shared_width = shared_width
        self.context_width = context_width
        self.latent_width = latent_width
        self.measured_layers = list(measured_layers)
        self.layer_to_row = {layer: row for row, layer in enumerate(measured_layers)}
        self.context_mode = context_mode
        self.learned_local_decoder = learned_local_decoder
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def parameter(shape: tuple[int, ...], fan_in: int) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                torch.randn(shape, generator=generator) / math.sqrt(fan_in)
            )

        self.p = parameter((context_width, 2 * width), 2 * width)
        self.a = parameter((shared_width, 2 * block_width), 2 * block_width)
        self.b = parameter((shared_width, context_width), context_width)
        self.c = parameter((shared_width, latent_width), latent_width)
        d_u = torch.randn(
            block_width, shared_width, generator=generator
        ) / math.sqrt(shared_width)
        d_v = torch.randn(
            block_width, shared_width, generator=generator
        ) / math.sqrt(shared_width)
        if learned_local_decoder:
            self.d_u = torch.nn.Parameter(d_u)
            self.d_v = torch.nn.Parameter(d_v)
        else:
            self.register_buffer("d_u", d_u)
            self.register_buffer("d_v", d_v)
        self.layer_embeddings = torch.nn.Parameter(
            torch.zeros(deployment_layers, shared_width)
        )
        self.position_embeddings = torch.nn.Parameter(
            torch.zeros(self.positions, shared_width)
        )
        self.codes_u = torch.nn.Parameter(
            0.02
            * torch.randn(
                len(measured_layers), components, latent_width,
                generator=generator,
            )
        )

    def _whole_rows(
        self, detector_blocks: torch.Tensor, write_blocks: torch.Tensor
    ) -> torch.Tensor:
        detector = detector_blocks.reshape(-1, self.positions, self.block_width)
        write = write_blocks.reshape(-1, self.positions, self.block_width)
        detector = detector.reshape(-1, self.width)
        write = write.reshape(-1, self.width)
        return torch.cat((detector, write), dim=-1)

    def _base(
        self,
        layer: int,
        detector_blocks: torch.Tensor,
        write_blocks: torch.Tensor,
        positions: torch.Tensor,
        row_indices: torch.Tensor,
        whole_rows: torch.Tensor,
    ) -> torch.Tensor:
        if self.context_mode == "none":
            context = torch.zeros(
                row_indices.shape[0], self.context_width,
                device=detector_blocks.device, dtype=detector_blocks.dtype,
            )
        else:
            context_rows = row_indices
            if self.context_mode == "shuffled":
                context_rows = (context_rows + 193) % whole_rows.shape[0]
            context = F.linear(whole_rows[context_rows], self.p)
        paired_blocks = torch.cat((detector_blocks, write_blocks), dim=-1)
        return (
            F.linear(paired_blocks, self.a)
            + F.linear(context, self.b)
            + self.layer_embeddings[layer]
            + self.position_embeddings[positions]
        )

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
        whole_rows = self._whole_rows(detector_blocks, write_blocks)
        if block_indices is None:
            row_indices = torch.arange(
                detector_blocks.shape[0], device=detector_blocks.device
            ) // self.positions
        else:
            row_indices = block_indices // self.positions
            detector_blocks = detector_blocks[block_indices]
            write_blocks = write_blocks[block_indices]
            positions = positions[block_indices]
        row = self.layer_to_row[layer]
        codes = self.codes_u[row]
        if one_code:
            codes = codes[:1]
        if zero_codes:
            codes = torch.zeros_like(codes)
        base = self._base(
            layer, detector_blocks, write_blocks, positions, row_indices, whole_rows
        )
        injection = F.linear(codes, self.c)
        hidden = F.gelu(base[None] + injection[:, None]) - F.gelu(base[None])
        return F.linear(hidden, self.d_u), F.linear(hidden, self.d_v)

    def compact_state(self, *, deployment_layers: int) -> dict[str, torch.Tensor]:
        return {
            "p": self.p.detach().half().cpu(),
            "a": self.a.detach().half().cpu(),
            "b": self.b.detach().half().cpu(),
            "c": self.c.detach().half().cpu(),
            "layer_embeddings": self.layer_embeddings.detach().half().cpu(),
            "position_embeddings": self.position_embeddings.detach().half().cpu(),
            "live_latents": torch.zeros(
                deployment_layers, self.latent_width, dtype=torch.float16
            ),
        }


def compact_payload(
    decoder: WholeNeuronContextLocalWriteDecoder,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    tensors = decoder.compact_state(deployment_layers=12)
    payload_bytes = sum(x.numel() * x.element_size() for x in tensors.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise ValueError(f"H71 payload {payload_bytes} != {accounting}")
    return {
        "schema_version": SCHEMA_VERSION,
        "accounting": accounting,
        "accounted_payload_bytes": payload_bytes,
        "procedural_local_decoder_seed": 20260902,
        "tensors": tensors,
    }


def synthetic_self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(71)
    decoder = WholeNeuronContextLocalWriteDecoder(
        width=8,
        block_width=4,
        shared_width=12,
        context_width=3,
        latent_width=3,
        deployment_layers=12,
        measured_layers=[0],
        components=3,
        seed=7,
    ).to(device)
    detector = torch.randn(10, 4, device=device)
    write = torch.randn(10, 4, device=device)
    positions = torch.arange(10, device=device) % 2
    zero_u, zero_v = decoder.predict(
        0, detector, write, positions, zero_codes=True
    )
    if int(torch.count_nonzero(zero_u)) or int(torch.count_nonzero(zero_v)):
        raise AssertionError("H71 zero-state identity failed")
    u, v = decoder.predict(0, detector, write, positions)
    (u.square().mean() + v.square().mean()).backward()
    for name in (
        "p", "a", "b", "c", "layer_embeddings",
        "position_embeddings", "codes_u",
    ):
        parameter = getattr(decoder, name)
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing H71 gradient: {name}")
    if decoder.d_u.requires_grad or decoder.d_v.requires_grad:
        raise AssertionError("H71 procedural local decoders are not frozen")
    shuffled = WholeNeuronContextLocalWriteDecoder(
        width=8,
        block_width=4,
        shared_width=12,
        context_width=3,
        latent_width=3,
        deployment_layers=12,
        measured_layers=[0],
        components=3,
        seed=7,
        context_mode="shuffled",
    ).to(device)
    shuffled.load_state_dict(decoder.state_dict())
    shuffled_u, _ = shuffled.predict(0, detector, write, positions)
    if torch.equal(u, shuffled_u):
        raise AssertionError("H71 shuffled context is not causally distinct")
    accounting = deployment_accounting()
    if accounting["total_fp16_values"] != 446_656:
        raise AssertionError(accounting)
    if "d_u" in decoder.compact_state(deployment_layers=12):
        raise AssertionError("H71 procedural decoder leaked into checkpoint")
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
    if plan["inputs"]["host"] != "PRO6":
        raise ValueError("H71 is authorized only on PRO6")
    fit = plan["fit"]
    transform = plan["transformation"]
    layers = [int(value) for value in plan["inputs"]["required_layers"]]
    components = int(plan["inputs"]["top_components"])
    block_width = int(transform["block_width"])
    shared_width = int(transform["shared_width"])
    context_width = int(transform["complete_neuron_context_width"])
    latent_width = int(transform["latent_width"])
    seed = int(fit["seed"])
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    raw_bundles, input_manifest = load_layer_bundles(
        args.trajectory_dir,
        layers=layers,
        components=components,
        device=args.device,
    )
    if input_manifest["run_identity_sha256"] != plan["inputs"]["run_identity_sha256"]:
        raise ValueError("H71 run identity mismatch")
    width = int(raw_bundles[0]["detector_w0"].shape[1])
    bundles = [blockify_bundle(x, block_width=block_width) for x in raw_bundles]
    for bundle in bundles:
        for key in (
            "detector_w0_blocks", "write_w0_blocks", "detector_pc_blocks",
            "write_pc_blocks", "block_positions", "weights", "eigenvalues",
        ):
            bundle[key] = bundle[key].to(args.device)
    del raw_bundles

    specs = {
        "whole_neuron_context_local_write": {
            "context_mode": "exact", "blind": False, "learned_decoder": False,
        },
        "no_whole_neuron_context": {
            "context_mode": "none", "blind": False, "learned_decoder": False,
        },
        "shuffled_whole_neuron_context": {
            "context_mode": "shuffled", "blind": False,
            "learned_decoder": False,
        },
        "w0_blind": {
            "context_mode": "exact", "blind": True, "learned_decoder": False,
        },
        "learned_local_decoder": {
            "context_mode": "exact", "blind": False, "learned_decoder": True,
        },
    }
    arms: dict[str, Any] = {}
    decoders: dict[str, WholeNeuronContextLocalWriteDecoder] = {}
    updates = int(fit["updates"])
    for arm_index, (name, spec) in enumerate(specs.items()):
        active = (
            procedural_blind_blocks(
                bundles, seed=seed + 10_000_019, device=args.device
            )
            if spec["blind"]
            else bundles
        )
        decoder = WholeNeuronContextLocalWriteDecoder(
            width=width,
            block_width=block_width,
            shared_width=shared_width,
            context_width=context_width,
            latent_width=latent_width,
            deployment_layers=int(transform["deployment_layer_count"]),
            measured_layers=layers,
            components=components,
            seed=seed,
            context_mode=str(spec["context_mode"]),
            learned_local_decoder=bool(spec["learned_decoder"]),
        ).to(args.device)
        history = fit_decoder(
            decoder,
            active,
            updates=updates,
            block_batch=int(fit["block_batch_per_layer"]),
            learning_rate=float(fit["learning_rate"]),
            betas=(float(fit["betas"][0]), float(fit["betas"][1])),
            weight_decay=float(fit["weight_decay"]),
            freeze_basis=False,
            progress_offset=arm_index * updates,
        )
        metrics = evaluate_decoder(
            decoder, active, chunk=int(fit["evaluation_block_chunk"])
        )
        arms[name] = {"history": history, "metrics": metrics}
        decoders[name] = decoder
        print(json.dumps({"arm": name, "metrics": metrics}, sort_keys=True), flush=True)

    candidate = arms["whole_neuron_context_local_write"]["metrics"]
    no_context = {
        row["layer"]: row
        for row in arms["no_whole_neuron_context"]["metrics"]["rows"]
    }
    shuffled = {
        row["layer"]: row
        for row in arms["shuffled_whole_neuron_context"]["metrics"]["rows"]
    }
    margins = [
        {
            "layer": row["layer"],
            "candidate_minus_no_context": (
                row["weighted_top16_capture"]
                - no_context[row["layer"]]["weighted_top16_capture"]
            ),
            "candidate_minus_shuffled_context": (
                row["weighted_top16_capture"]
                - shuffled[row["layer"]]["weighted_top16_capture"]
            ),
        }
        for row in candidate["rows"]
    ]
    decoder = decoders["whole_neuron_context_local_write"]
    zero_nonzero = 0
    zero_max_abs = 0.0
    with torch.no_grad():
        for bundle in bundles:
            indices = torch.arange(64, device=args.device)
            u, v = decoder.predict(
                int(bundle["layer"]),
                bundle["detector_w0_blocks"],
                bundle["write_w0_blocks"],
                bundle["block_positions"],
                block_indices=indices,
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
        "no_context_margin_every_layer": min(
            row["candidate_minus_no_context"] for row in margins
        ) >= float(gates["candidate_minus_no_context_every_layer_minimum"]),
        "shuffled_context_margin_every_layer": min(
            row["candidate_minus_shuffled_context"] for row in margins
        ) >= float(gates["candidate_minus_shuffled_context_every_layer_minimum"]),
        "procedural_local_basis_full_rank": min(
            basis["d_u"]["numerical_rank"], basis["d_v"]["numerical_rank"]
        ) >= int(gates["procedural_local_basis_numerical_rank_minimum"]),
        "zero_state_exact": zero_nonzero == 0 and zero_max_abs <= float(
            gates["zero_state_max_abs"]
        ),
    }
    gate_outcomes["all_values_finite"] = all(
        math.isfinite(float(row["weighted_top16_capture"]))
        and math.isfinite(float(row["minimum_pc_capture"]))
        for arm in arms.values()
        for row in arm["metrics"]["rows"]
    )
    classification = "PASSED" if all(gate_outcomes.values()) else "REJECTED"
    accounting = deployment_accounting(
        shared_width=shared_width,
        context_width=context_width,
        block_width=block_width,
        latent_width=latent_width,
        width=width,
    )
    if accounting["total_checkpoint_payload_bytes"] != plan["persistent_state"]["total_bytes"]:
        raise ValueError("H71 accounting disagrees with frozen plan")
    refresh_seconds = benchmark_one_layer_refresh(
        decoder, bundles[0], chunk=int(fit["evaluation_block_chunk"])
    )
    payload = compact_payload(decoder, accounting)

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / "compact_checkpoint.pt"
    torch.save(payload, checkpoint)
    per_layer = args.output / "per_layer.csv"
    write_csv(
        per_layer,
        [
            {"arm": name, **row}
            for name, arm in arms.items()
            for row in arm["metrics"]["rows"]
        ],
    )
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
            "offline_nuisance_code_fp32_values": (
                len(layers) * components * latent_width
            ),
            "static_key_cache_bytes_used": 0,
            "one_layer_uncached_refresh_seconds": refresh_seconds,
            "twelve_layer_uncached_refresh_seconds_linear_extrapolation": (
                12 * refresh_seconds
            ),
        },
        "inputs": input_manifest,
        "arms": arms,
        "limitations": [
            "This is an optimistic top-16 image-capacity audit, not CE training.",
            "Offline PC codes are nuisance coordinates and are not deployed state.",
            "One optimizer path does not identify every possible task manifold.",
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
        ("result", result_path),
        ("checkpoint", checkpoint),
        ("per_layer", per_layer),
    ):
        metadata["outputs"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": classification,
                "gate_outcomes": gate_outcomes,
                "result_sha256": metadata["outputs"]["result"]["sha256"],
                "runtime_seconds": metadata["runtime_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
