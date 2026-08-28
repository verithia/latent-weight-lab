#!/usr/bin/env python3
"""First-PC audit for a soft-token, one-step synthetic Muon program.

The deployable latent is one global soft-token prompt plus per-matrix scalar
amplitudes.  Replaying a task loss at W0 generates MLP gradients, and the
fixed Muon NS5 map turns them into dense updates.  This analyzer projects
saved residual PCs through the exact mixed-Hessian/polar tangent.  It does not
optimize the prompt or run a language-model CE endpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.func import functional_call

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon import zeropower_via_newtonschulz5
from examples.nanogpt.train import TokenBatchSource, make_cpu_generator


TRAJECTORY_SCHEMA = "nanogpt_parameter_trajectory_v1"
PROBE_SCHEMA = "nanogpt_optimizer_probe_v1"
DENSE_MLP_SCALARS = 56_623_104
DEPLOYED_MLP_MATRICES = 24
PARAMETER_NAMES = {
    "c_fc": "transformer.h.6.mlp.c_fc.weight",
    "c_proj": "transformer.h.6.mlp.c_proj.weight",
}
TensorTuple = tuple[torch.Tensor, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def latent_accounting(prompt_length: int, width: int) -> dict[str, int | float]:
    prompt_scalars = prompt_length * width
    total_scalars = prompt_scalars + DEPLOYED_MLP_MATRICES
    return {
        "prompt_length": prompt_length,
        "prompt_width": width,
        "prompt_scalars": prompt_scalars,
        "amplitude_scalars": DEPLOYED_MLP_MATRICES,
        "total_scalars": total_scalars,
        "dense_mlp_denominator_scalars": DENSE_MLP_SCALARS,
        "deployable_scalar_fraction": total_scalars / DENSE_MLP_SCALARS,
        "deployable_checkpoint_fp16_bytes": 2 * total_scalars,
        "fp32_prompt_master_bytes_during_fit": 4 * prompt_scalars,
        "fp32_prompt_gradient_bytes_during_fit": 4 * prompt_scalars,
        "adam_fp32_moment_bytes_during_fit": 8 * prompt_scalars,
    }


def initialization_match(
    reconstructed: torch.Tensor,
    stored: torch.Tensor,
) -> dict[str, bool | float | str]:
    """Compare procedural FP32 W0 to its trajectory storage representation."""
    reconstructed_cpu = reconstructed.detach().cpu()
    stored_cpu = stored.detach().cpu()
    roundtrip = reconstructed_cpu.to(stored_cpu.dtype)
    relative = float(
        (reconstructed_cpu.float() - stored_cpu.float()).norm()
        / reconstructed_cpu.float().norm().clamp_min(1e-30)
    )
    cosine = float(
        (reconstructed_cpu.double() * stored_cpu.double()).sum()
        / (
            reconstructed_cpu.double().norm()
            * stored_cpu.double().norm()
        ).clamp_min(1e-30)
    )
    return {
        "stored_dtype": str(stored_cpu.dtype),
        "fp32_reconstruction_dtype": str(reconstructed_cpu.dtype),
        "storage_roundtrip_bitwise_equal": torch.equal(roundtrip, stored_cpu),
        "relative_storage_error": relative,
        "cosine": cosine,
        "accepted": bool(torch.equal(roundtrip, stored_cpu) and relative <= 5e-4),
    }


def _dot(left: TensorTuple, right: TensorTuple) -> torch.Tensor:
    return sum((a.double() * b.double()).sum() for a, b in zip(left, right, strict=True))


def _add(left: TensorTuple, right: TensorTuple, scale: torch.Tensor | float) -> TensorTuple:
    return tuple(a + b * scale for a, b in zip(left, right, strict=True))


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
        if float(next_energy.sqrt()) <= tolerance * max(math.sqrt(initial_energy), 1e-30):
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


def project_direction(
    function: Callable[..., torch.Tensor],
    primals: TensorTuple,
    target: torch.Tensor,
    *,
    cg_iterations: int,
    cg_tolerance: float,
    relative_ridge: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    value, vjp = torch.func.vjp(function, *primals)
    right_hand_side = tuple(item.detach() for item in vjp(target))
    rhs_energy = float(_dot(right_hand_side, right_hand_side))
    if rhs_energy <= 1e-30:
        return torch.zeros_like(target), {
            "cg_iterations": 0.0,
            "cg_initial_residual_norm": 0.0,
            "cg_final_residual_norm": 0.0,
            "ridge_absolute": 0.0,
        }
    _, rayleigh_image = torch.func.jvp(function, primals, right_hand_side)
    rayleigh = float(rayleigh_image.double().square().sum()) / rhs_energy
    ridge = max(relative_ridge * max(rayleigh, 1e-30), 1e-30)

    def operator(vector: TensorTuple) -> TensorTuple:
        _, image = torch.func.jvp(function, primals, vector)
        adjoint = vjp(image)
        return tuple(
            item.detach() + ridge * coordinate
            for item, coordinate in zip(adjoint, vector, strict=True)
        )

    coordinates, diagnostics = conjugate_gradient(
        operator,
        right_hand_side,
        maximum_iterations=cg_iterations,
        tolerance=cg_tolerance,
    )
    _, projected = torch.func.jvp(function, primals, coordinates)
    diagnostics["ridge_absolute"] = ridge
    diagnostics["base_output_norm"] = float(value.double().norm())
    return projected.detach(), diagnostics


def projection_metrics(reference: torch.Tensor, projected: torch.Tensor) -> dict[str, float]:
    reference64 = reference.double()
    projected64 = projected.double()
    reference_energy = float(reference64.square().sum())
    projected_energy = float(projected64.square().sum())
    cross = float((reference64 * projected64).sum())
    error = float((reference64 - projected64).square().sum())
    capture = (2.0 * cross - projected_energy) / max(reference_energy, 1e-30)
    cosine = cross / max(math.sqrt(reference_energy * projected_energy), 1e-30)
    return {
        "reference_energy": reference_energy,
        "projected_energy": projected_energy,
        "cross": cross,
        "path_energy_capture": max(min(capture, 1.0), -1.0),
        "relative_error": math.sqrt(error / max(reference_energy, 1e-30)),
        "cosine": max(min(cosine, 1.0), -1.0),
    }


def principal_component(
    states: torch.Tensor,
    *,
    anchored: bool,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    residual = states - (states[0] if anchored else states.mean(0))
    flat = residual.flatten(1).float()
    gram = flat @ flat.T
    gram = (gram + gram.T) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    vectors = vectors[:, order]
    pc = (vectors[:, 0] @ flat) / eigenvalues[0].sqrt().clamp_min(1e-20)
    total = float(eigenvalues.sum().clamp_min(1e-30))
    fraction = float(eigenvalues[0]) / total
    return pc.reshape(states.shape[1:]).contiguous(), fraction, {
        "leading_eigenvalue": float(eigenvalues[0]),
        "total_path_energy": total,
    }


def load_trajectory_parameter(path: Path, parameter: str) -> tuple[torch.Tensor, str]:
    files = sorted(path.glob("step_*.pt"))
    if len(files) != 239:
        raise ValueError(f"expected 239 states, found {len(files)}")
    states: list[torch.Tensor] = []
    identity: str | None = None
    previous_step = -1
    for file in files:
        payload = torch.load(file, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != TRAJECTORY_SCHEMA:
            raise ValueError(f"unexpected trajectory schema in {file}")
        step = int(payload["step"])
        if step != previous_step + 1:
            raise ValueError("trajectory steps are not contiguous from zero")
        previous_step = step
        observed = str(payload["run_identity_sha256"])
        identity = observed if identity is None else identity
        if observed != identity:
            raise ValueError("trajectory identity changed")
        states.append(payload["parameters"][parameter].detach().contiguous())
    return torch.stack(states), str(identity)


def load_probe_weights(path: Path, parameter: str) -> tuple[torch.Tensor, list[int], str]:
    files = sorted(path.glob("step_*.pt"))
    if len(files) != 100:
        raise ValueError(f"expected 100 probes, found {len(files)}")
    weights: list[torch.Tensor] = []
    steps: list[int] = []
    identity: str | None = None
    for file in files:
        payload = torch.load(file, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != PROBE_SCHEMA:
            raise ValueError(f"unexpected probe schema in {file}")
        step = int(payload["step"])
        if steps and step <= steps[-1]:
            raise ValueError("probe steps are not strictly increasing")
        observed = str(payload["run_identity_sha256"])
        identity = observed if identity is None else identity
        if observed != identity:
            raise ValueError("probe identity changed")
        weights.append(payload["parameters"][parameter]["weight_before_step"].detach().contiguous())
        steps.append(step)
    return torch.stack(weights), steps, str(identity)


def build_dense_model(config: dict[str, Any], device: str) -> GPT:
    torch.manual_seed(int(config["model_seed"]))
    model_config = GPTConfig(
        block_size=int(config["block_size"]),
        vocab_size=int(config["vocab_size"]),
        n_layer=int(config["n_layer"]),
        n_head=int(config["n_head"]),
        n_embd=int(config["n_embd"]),
        dropout=float(config["dropout"]),
        bias=bool(config["bias"]),
        block_fht=False,
        tie_word_embeddings=bool(config["tie_word_embeddings"]),
    )
    return GPT(model_config).to(device).eval()


def make_prompt(
    model: GPT,
    config: dict[str, Any],
    *,
    prompt_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = make_cpu_generator(int(config["train_data_seed"]))
    assert generator is not None
    source = TokenBatchSource(Path(config["data_dir"]))
    x_cpu, y_cpu = source.get_batch_cpu(
        "train",
        int(config["batch_size"]),
        int(config["block_size"]),
        generator=generator,
    )
    ids = x_cpu[0:1, :prompt_length].to(device)
    targets = y_cpu[0:1, :prompt_length].to(device)
    prompt = model.transformer.wte(ids).detach().float().contiguous()
    return prompt, targets, {
        "input_token_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
        "target_token_sha256": hashlib.sha256(targets.cpu().numpy().tobytes()).hexdigest(),
        "input_minimum": int(ids.min()),
        "input_maximum": int(ids.max()),
        "target_minimum": int(targets.min()),
        "target_maximum": int(targets.max()),
    }


def math_sdpa_context() -> Any:
    if not torch.cuda.is_available():
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(SDPBackend.MATH)


def make_program_function(
    model: GPT,
    *,
    parameter: str,
    targets: torch.Tensor,
    ns_steps: int,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dict[str, Any]]:
    parameters = {name: value.detach() for name, value in model.named_parameters()}
    buffers = {name: value.detach() for name, value in model.named_buffers()}
    if parameter not in parameters:
        raise ValueError(f"missing selected parameter {parameter}")
    selected_weight = parameters[parameter]
    static_parameters = {name: value for name, value in parameters.items() if name != parameter}

    def task_loss(weight: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        call_parameters = dict(static_parameters)
        call_parameters[parameter] = weight
        with math_sdpa_context():
            _, loss = functional_call(
                model,
                (call_parameters, buffers),
                (None, targets),
                {"input_embeddings": prompt},
                tie_weights=True,
                strict=False,
            )
        assert loss is not None
        return loss

    gradient_function = torch.func.grad(task_loss, argnums=0)

    def program(prompt: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        gradient = gradient_function(selected_weight, prompt)
        return amplitude * zeropower_via_newtonschulz5(gradient, steps=ns_steps)

    with torch.no_grad():
        selected_norm = float(selected_weight.double().norm())
    return program, {
        "selected_weight_shape": list(selected_weight.shape),
        "selected_weight_norm": selected_norm,
    }


def self_test(device: str) -> dict[str, float | str]:
    torch.manual_seed(20260902)
    weight = torch.randn(7, 5, device=device)
    prompt = torch.randn(1, 6, 5, device=device)
    target = torch.randn(1, 6, 7, device=device)

    def loss_fn(active_weight: torch.Tensor, active_prompt: torch.Tensor) -> torch.Tensor:
        prediction = active_prompt @ active_weight.T
        return 0.5 * (prediction - target).square().mean()

    gradient_fn = torch.func.grad(loss_fn, argnums=0)

    def function(active_prompt: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        return amplitude * zeropower_via_newtonschulz5(
            gradient_fn(weight, active_prompt), steps=5
        )

    primals = (prompt, torch.ones((), device=device))
    direction = (torch.randn_like(prompt), torch.randn((), device=device))
    _, exact_target = torch.func.jvp(function, primals, direction)
    projected, diagnostics = project_direction(
        function,
        primals,
        exact_target,
        cg_iterations=80,
        cg_tolerance=1e-8,
        relative_ridge=1e-10,
    )
    metrics = projection_metrics(exact_target, projected)
    if metrics["path_energy_capture"] < 0.999:
        raise AssertionError((metrics, diagnostics))
    return {"status": "passed", **metrics, **diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--run-a-probe-dir", type=Path)
    parser.add_argument("--run-b-probe-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-length", type=int, default=737)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--cg-iterations", type=int, default=20)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-ridge", type=float, default=1e-6)
    parser.add_argument("--parameters", default="c_fc,c_proj")
    parser.add_argument("--paths", default="trajectory_centered,common_centered")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    required = (args.config, args.trajectory_dir, args.run_a_probe_dir, args.run_b_probe_dir, args.output)
    if any(value is None for value in required):
        parser.error("config, trajectory, two probe directories, and output are required")
    if args.prompt_length != 737 or args.ns_steps != 5:
        raise ValueError("the frozen H29a oracle requires length 737 and NS5")
    if args.preflight and args.cg_iterations != 1:
        raise ValueError("the exact systems preflight requires one CG iteration")
    if not args.preflight and args.cg_iterations != 20:
        raise ValueError("the binding H29a oracle requires 20 CG iterations")
    selected_suffixes = [item.strip() for item in args.parameters.split(",") if item.strip()]
    paths = [item.strip() for item in args.paths.split(",") if item.strip()]
    if not set(selected_suffixes) <= set(PARAMETER_NAMES):
        raise ValueError("unsupported parameter suffix")
    if not set(paths) <= {"trajectory_centered", "common_centered"}:
        raise ValueError("unsupported path family")
    if args.preflight and (selected_suffixes != ["c_proj"] or paths != ["trajectory_centered"]):
        raise ValueError("the frozen preflight uses c_proj trajectory_centered only")

    accounting = latent_accounting(args.prompt_length, 768)
    if int(accounting["total_scalars"]) != 566_040 or float(accounting["deployable_scalar_fraction"]) > 0.01:
        raise ValueError("H29a state accounting mismatch")
    output = args.output
    assert output is not None
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = json.loads(args.config.read_text())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_dense_model(config, args.device)
    prompt, targets, prompt_manifest = make_prompt(
        model,
        config,
        prompt_length=args.prompt_length,
        device=args.device,
    )
    rows: list[dict[str, Any]] = []
    identities: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for suffix in selected_suffixes:
        parameter = PARAMETER_NAMES[suffix]
        trajectory_cpu, trajectory_identity = load_trajectory_parameter(args.trajectory_dir, parameter)
        run_a_cpu, steps_a, run_a_identity = load_probe_weights(args.run_a_probe_dir, parameter)
        run_b_cpu, steps_b, run_b_identity = load_probe_weights(args.run_b_probe_dir, parameter)
        if steps_a != steps_b:
            raise ValueError("A/B schedules differ")
        if not torch.equal(run_a_cpu[0], run_b_cpu[0]):
            raise ValueError("A/B step-zero gauge mismatch")
        model_weight = dict(model.named_parameters())[parameter].detach().cpu()
        w0_match = initialization_match(model_weight, run_a_cpu[0])
        if not bool(w0_match["accepted"]):
            raise ValueError(f"model/probe W0 mismatch for {parameter}: {w0_match}")
        common_cpu = 0.5 * (run_a_cpu.float() + run_b_cpu.float())
        path_states = {
            "trajectory_centered": trajectory_cpu,
            "common_centered": common_cpu,
        }
        function, function_manifest = make_program_function(
            model,
            parameter=parameter,
            targets=targets,
            ns_steps=args.ns_steps,
        )
        primals = (prompt, torch.ones((), device=args.device))
        for path_name in paths:
            states = path_states[path_name].to(args.device, dtype=torch.float32)
            pc, fraction, pc_manifest = principal_component(states, anchored=False)
            solve_started = time.time()
            projected, diagnostics = project_direction(
                function,
                primals,
                pc,
                cg_iterations=args.cg_iterations,
                cg_tolerance=args.cg_tolerance,
                relative_ridge=args.relative_ridge,
            )
            metrics = projection_metrics(pc, projected)
            best_possible = fraction * metrics["path_energy_capture"] + (1.0 - fraction)
            rows.append(
                {
                    "parameter": parameter,
                    "path": path_name,
                    "leading_pc_energy_fraction": fraction,
                    "best_possible_total_capture": best_possible,
                    "retains_twenty_percent_possibility": best_possible >= 0.20,
                    "solve_seconds": time.time() - solve_started,
                    **pc_manifest,
                    **metrics,
                    **diagnostics,
                }
            )
            del states, pc, projected
            torch.cuda.empty_cache()
        identities[parameter] = {
            "trajectory_run_identity_sha256": trajectory_identity,
            "run_a_identity_sha256": run_a_identity,
            "run_b_identity_sha256": run_b_identity,
            "w0_storage_match": w0_match,
            "function": function_manifest,
        }
        del trajectory_cpu, run_a_cpu, run_b_cpu, common_cpu, function
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    result_path = output / "pc1_projection.csv"
    accounting_path = output / "accounting.json"
    write_csv(result_path, rows)
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    retained = all(bool(row["retains_twenty_percent_possibility"]) for row in rows)
    script = Path(__file__).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "schema_version": "nanogpt_mlp_synthetic_muon_program_pc1_v1",
        "classification": "PREFLIGHT" if args.preflight else ("RETAINED" if retained else "REJECTED"),
        "preflight": args.preflight,
        "retained": retained,
        "accounting": accounting,
        "prompt_manifest": prompt_manifest,
        "model_config": asdict(model.config),
        "identities": identities,
        "rows": rows,
        "execution": {
            "source_commit": commit,
            "entrypoint": str(script),
            "entrypoint_sha256": sha256(script),
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": time.time() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": args.device,
        },
        "outputs": {
            "projection": {"path": str(result_path), "sha256": sha256(result_path)},
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)},
        },
        "limitations": [
            "Per-parameter PC projections are optimistic and do not prove simultaneous joint-prompt fitting.",
            "The test measures the local decoder tangent at the registered real-embedding anchor.",
            "A first-PC pass authorizes remaining-PC and joint-node audits, never CE directly.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), "classification": metadata["classification"], "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
