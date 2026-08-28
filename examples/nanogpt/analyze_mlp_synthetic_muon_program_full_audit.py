#!/usr/bin/env python3
"""Full joint-path audit for the frozen H29 synthetic Muon program.

This analyzer does not optimize a prompt and does not run CE training.  It
projects four preregistered residual-path PCA families through one shared
six-output mixed-Hessian/NS5 tangent and stores summaries only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_synthetic_muon_program import (
    PROBE_SCHEMA,
    TRAJECTORY_SCHEMA,
    build_dense_model,
    initialization_match,
    latent_accounting,
    make_prompt,
    project_direction,
    projection_metrics,
    sha256,
    write_csv,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program_joint import (
    FROZEN_PARAMETERS,
    make_joint_program_function,
)


LAYER6_PARAMETERS = tuple(
    f"transformer.h.6.mlp.{suffix}.weight" for suffix in ("c_fc", "c_proj")
)
LATE_START_STEP = 159


def load_trajectory_inventory(
    path: Path,
    parameters: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], str]:
    files = sorted(path.glob("step_*.pt"))
    if len(files) != 239:
        raise ValueError(f"expected 239 trajectory states, found {len(files)}")
    values: dict[str, list[torch.Tensor]] = {parameter: [] for parameter in parameters}
    identity: str | None = None
    for expected_step, file in enumerate(files):
        payload = torch.load(file, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != TRAJECTORY_SCHEMA:
            raise ValueError(f"unexpected trajectory schema in {file}")
        step = int(payload["step"])
        if step != expected_step:
            raise ValueError(f"trajectory step mismatch: expected {expected_step}, found {step}")
        observed = str(payload["run_identity_sha256"])
        identity = observed if identity is None else identity
        if observed != identity:
            raise ValueError("trajectory identity changed")
        stored = payload["parameters"]
        for parameter in parameters:
            if parameter not in stored:
                raise ValueError(f"trajectory lacks {parameter}")
            values[parameter].append(stored[parameter].detach().contiguous())
    return {parameter: torch.stack(rows) for parameter, rows in values.items()}, str(identity)


def load_probe_inventory(
    path: Path,
    parameters: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], list[int], str]:
    files = sorted(path.glob("step_*.pt"))
    if len(files) != 100:
        raise ValueError(f"expected 100 optimizer probes, found {len(files)}")
    values: dict[str, list[torch.Tensor]] = {parameter: [] for parameter in parameters}
    steps: list[int] = []
    identity: str | None = None
    for file in files:
        payload = torch.load(file, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != PROBE_SCHEMA:
            raise ValueError(f"unexpected probe schema in {file}")
        step = int(payload["step"])
        if steps and step <= steps[-1]:
            raise ValueError("probe steps are not strictly increasing")
        steps.append(step)
        observed = str(payload["run_identity_sha256"])
        identity = observed if identity is None else identity
        if observed != identity:
            raise ValueError("probe identity changed")
        stored = payload["parameters"]
        for parameter in parameters:
            if parameter not in stored or "weight_before_step" not in stored[parameter]:
                raise ValueError(f"probe lacks weight_before_step for {parameter}")
            values[parameter].append(
                stored[parameter]["weight_before_step"].detach().contiguous()
            )
    return {parameter: torch.stack(rows) for parameter, rows in values.items()}, steps, str(identity)


def joint_principal_components(
    states: dict[str, torch.Tensor],
    *,
    parameter_order: tuple[str, ...],
    component_count: int,
    device: str,
) -> dict[str, Any]:
    """Compute normalized joint PCs while streaming one matrix through CUDA."""
    if not states:
        raise ValueError("no active state parts")
    unknown = set(states) - set(parameter_order)
    if unknown:
        raise ValueError(f"unknown state parts: {sorted(unknown)}")
    counts = {int(value.shape[0]) for value in states.values()}
    if len(counts) != 1:
        raise ValueError("state counts differ")
    state_count = counts.pop()
    if component_count <= 0 or component_count >= state_count:
        raise ValueError("invalid component count")

    gram: torch.Tensor | None = None
    shapes: dict[str, list[int]] = {}
    for parameter in parameter_order:
        if parameter not in states:
            continue
        matrix_states = states[parameter].to(device, dtype=torch.float32)
        centered = matrix_states - matrix_states.mean(dim=0, keepdim=True)
        flat = centered.flatten(1)
        contribution = flat @ flat.T
        gram = contribution if gram is None else gram + contribution
        shapes[parameter] = list(states[parameter].shape[1:])
        del matrix_states, centered, flat, contribution
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert gram is not None
    gram = (gram + gram.T) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    vectors = vectors[:, order]
    selected_vectors = vectors[:, :component_count]
    scales = eigenvalues[:component_count].sqrt().clamp_min(1e-20)

    parts: dict[str, torch.Tensor] = {}
    for parameter in parameter_order:
        if parameter not in states:
            continue
        matrix_states = states[parameter].to(device, dtype=torch.float32)
        centered = matrix_states - matrix_states.mean(dim=0, keepdim=True)
        component_parts = selected_vectors.T @ centered.flatten(1)
        component_parts = component_parts / scales[:, None]
        parts[parameter] = component_parts.detach().cpu().contiguous()
        del matrix_states, centered, component_parts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_energy = float(eigenvalues.sum().clamp_min(1e-30))
    return {
        "state_count": state_count,
        "component_count": component_count,
        "eigenvalues": eigenvalues.detach().cpu(),
        "total_energy": total_energy,
        "parts": parts,
        "shapes": shapes,
    }


def assemble_component(
    bundle: dict[str, Any],
    *,
    component: int,
    parameter_order: tuple[str, ...],
    parameter_shapes: dict[str, tuple[int, ...]],
    device: str,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    parts: dict[str, torch.Tensor] = bundle["parts"]
    for parameter in parameter_order:
        count = 1
        for dimension in parameter_shapes[parameter]:
            count *= dimension
        if parameter in parts:
            part = parts[parameter][component]
            if part.numel() != count:
                raise ValueError(f"part shape mismatch for {parameter}")
            pieces.append(part.to(device, dtype=torch.float32))
        else:
            pieces.append(torch.zeros(count, device=device, dtype=torch.float32))
    target = torch.cat(pieces)
    norm = float(target.double().norm())
    if not torch.isfinite(torch.tensor(norm)) or norm <= 1e-12:
        raise ValueError(f"joint PC has invalid norm: {norm}")
    # The temporal Gram eigensolve and ambient reconstruction are FP32.  Weak
    # late PCs can accumulate a few parts per thousand of roundoff in their
    # reconstructed norm, so enforce the mathematical unit-PC definition
    # explicitly.  Projection capture is scale invariant under this step.
    return target / norm


def audit_path(
    *,
    path_name: str,
    states: dict[str, torch.Tensor],
    function: Any,
    primals: tuple[torch.Tensor, ...],
    parameter_shapes: dict[str, tuple[int, ...]],
    component_count: int,
    cg_iterations: int,
    cg_tolerance: float,
    relative_ridge: float,
    device: str,
    rows: list[dict[str, Any]],
    progress_path: Path,
    completed_before: int,
    total_solves: int,
) -> tuple[dict[str, Any], int]:
    build_started = time.time()
    bundle = joint_principal_components(
        states,
        parameter_order=FROZEN_PARAMETERS,
        component_count=component_count,
        device=device,
    )
    build_seconds = time.time() - build_started
    eigenvalues: torch.Tensor = bundle["eigenvalues"]
    total_energy = float(bundle["total_energy"])
    weighted_capture = 0.0
    path_rows: list[dict[str, Any]] = []
    completed = completed_before
    for component in range(component_count):
        target = assemble_component(
            bundle,
            component=component,
            parameter_order=FROZEN_PARAMETERS,
            parameter_shapes=parameter_shapes,
            device=device,
        )
        solve_started = time.time()
        projected, diagnostics = project_direction(
            function,
            primals,
            target,
            cg_iterations=cg_iterations,
            cg_tolerance=cg_tolerance,
            relative_ridge=relative_ridge,
        )
        metrics = projection_metrics(target, projected)
        energy_fraction = float(eigenvalues[component]) / total_energy
        weighted_contribution = energy_fraction * metrics["path_energy_capture"]
        weighted_capture += weighted_contribution
        row = {
            "path": path_name,
            "component": component + 1,
            "eigenvalue": float(eigenvalues[component]),
            "path_energy_fraction": energy_fraction,
            "weighted_capture_contribution": weighted_contribution,
            "solve_seconds": time.time() - solve_started,
            **metrics,
            **diagnostics,
        }
        rows.append(row)
        path_rows.append(row)
        completed += 1
        progress_path.write_text(
            json.dumps(
                {
                    "state": "running",
                    "completed": completed,
                    "total": total_solves,
                    "path": path_name,
                    "component": component + 1,
                    "weighted_capture_so_far": weighted_capture,
                },
                sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps({"event": "progress", **json.loads(progress_path.read_text())}), flush=True)
        del target, projected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "path": path_name,
        "state_count": int(bundle["state_count"]),
        "component_count": component_count,
        "pca_build_seconds": build_seconds,
        "total_path_energy": total_energy,
        "top_component_energy_fraction": float(eigenvalues[0]) / total_energy,
        "top_k_energy_fraction": float(eigenvalues[:component_count].sum()) / total_energy,
        "weighted_observed_capture": weighted_capture,
        "minimum_component_capture": min(row["path_energy_capture"] for row in path_rows),
        "maximum_component_capture": max(row["path_energy_capture"] for row in path_rows),
        "mean_component_capture": sum(row["path_energy_capture"] for row in path_rows)
        / len(path_rows),
    }
    del bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, completed


def self_test(device: str) -> dict[str, Any]:
    torch.manual_seed(20260904)
    parameter_order = ("a", "b", "c")
    states = {
        "a": torch.randn(9, 3, 2),
        "c": torch.randn(9, 2, 2),
    }
    bundle = joint_principal_components(
        states,
        parameter_order=parameter_order,
        component_count=4,
        device=device,
    )
    shapes = {"a": (3, 2), "b": (2, 3), "c": (2, 2)}
    dense = torch.cat(
        [
            (states["a"] - states["a"].mean(0)).flatten(1),
            torch.zeros(9, 6),
            (states["c"] - states["c"].mean(0)).flatten(1),
        ],
        dim=1,
    ).to(device)
    _, singular_values, right = torch.linalg.svd(dense, full_matrices=False)
    comparisons: list[float] = []
    for component in range(4):
        recovered = assemble_component(
            bundle,
            component=component,
            parameter_order=parameter_order,
            parameter_shapes=shapes,
            device=device,
        )
        comparisons.append(abs(float(recovered @ right[component])))
    eigenvalue_error = float(
        (
            bundle["eigenvalues"][:4].to(device)
            - singular_values[:4].square()
        ).abs().max()
    )
    if min(comparisons) < 0.99999 or eigenvalue_error > 1e-4:
        raise AssertionError((comparisons, eigenvalue_error))
    return {
        "status": "passed",
        "minimum_pc_absolute_cosine": min(comparisons),
        "maximum_eigenvalue_error": eigenvalue_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--run-a-probe-dir", type=Path)
    parser.add_argument("--run-b-probe-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-length", type=int, default=737)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--cg-iterations", type=int, default=20)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-ridge", type=float, default=1e-6)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return
    required = (args.config, args.plan, args.trajectory_dir, args.output)
    if any(value is None for value in required):
        parser.error("config, plan, trajectory, and output are required")
    if not args.preflight and any(
        value is None for value in (args.run_a_probe_dir, args.run_b_probe_dir)
    ):
        parser.error("binding audit requires both A/B probe directories")
    if args.prompt_length != 737 or args.ns_steps != 5:
        raise ValueError("the frozen H29d decoder requires length 737 and NS5")
    expected = (2, 4) if args.preflight else (16, 20)
    if (args.components, args.cg_iterations) != expected:
        raise ValueError(f"this H29d mode requires components/CG={expected}")
    accounting = latent_accounting(args.prompt_length, 768)
    if int(accounting["total_scalars"]) != 566_040:
        raise ValueError("H29d state accounting mismatch")

    assert args.output is not None and args.config is not None and args.plan is not None
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "progress.json"
    total_solves = 2 if args.preflight else 64
    progress_path.write_text(
        json.dumps({"state": "starting", "completed": 0, "total": total_solves}, sort_keys=True)
        + "\n"
    )
    started = time.time()
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_dense_model(config, args.device)
    prompt, targets, prompt_manifest = make_prompt(
        model,
        config,
        prompt_length=args.prompt_length,
        device=args.device,
    )
    function, function_manifest = make_joint_program_function(
        model,
        parameters=FROZEN_PARAMETERS,
        targets=targets,
        ns_steps=args.ns_steps,
    )
    parameter_shapes = {
        name: tuple(dict(model.named_parameters())[name].shape) for name in FROZEN_PARAMETERS
    }
    primals = (prompt, torch.ones(len(FROZEN_PARAMETERS), device=args.device))
    torch.cuda.reset_peak_memory_stats()

    trajectory, trajectory_identity = load_trajectory_inventory(
        args.trajectory_dir,
        FROZEN_PARAMETERS,
    )
    w0_references = {name: values[0].clone() for name, values in trajectory.items()}
    w0_matches = {
        name: initialization_match(dict(model.named_parameters())[name], reference)
        for name, reference in w0_references.items()
    }
    if not all(bool(record["accepted"]) for record in w0_matches.values()):
        raise ValueError(f"model/trajectory W0 mismatch: {w0_matches}")

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    completed = 0
    summaries["joint_centered"], completed = audit_path(
        path_name="joint_centered",
        states=trajectory,
        function=function,
        primals=primals,
        parameter_shapes=parameter_shapes,
        component_count=args.components,
        cg_iterations=args.cg_iterations,
        cg_tolerance=args.cg_tolerance,
        relative_ridge=args.relative_ridge,
        device=args.device,
        rows=rows,
        progress_path=progress_path,
        completed_before=completed,
        total_solves=total_solves,
    )

    identities: dict[str, Any] = {"trajectory": trajectory_identity}
    if not args.preflight:
        late = {name: values[LATE_START_STEP:] for name, values in trajectory.items()}
        summaries["joint_late"], completed = audit_path(
            path_name="joint_late",
            states=late,
            function=function,
            primals=primals,
            parameter_shapes=parameter_shapes,
            component_count=args.components,
            cg_iterations=args.cg_iterations,
            cg_tolerance=args.cg_tolerance,
            relative_ridge=args.relative_ridge,
            device=args.device,
            rows=rows,
            progress_path=progress_path,
            completed_before=completed,
            total_solves=total_solves,
        )
        del late, trajectory

        assert args.run_a_probe_dir is not None and args.run_b_probe_dir is not None
        run_a, steps_a, identity_a = load_probe_inventory(
            args.run_a_probe_dir,
            LAYER6_PARAMETERS,
        )
        run_b, steps_b, identity_b = load_probe_inventory(
            args.run_b_probe_dir,
            LAYER6_PARAMETERS,
        )
        if steps_a != steps_b:
            raise ValueError("A/B probe schedules differ")
        for parameter in LAYER6_PARAMETERS:
            if not torch.equal(run_a[parameter][0], run_b[parameter][0]):
                raise ValueError(f"A/B step-zero mismatch for {parameter}")
            if not torch.equal(run_a[parameter][0], w0_references[parameter]):
                raise ValueError(f"trajectory/probe step-zero mismatch for {parameter}")
        identities.update({"run_a": identity_a, "run_b": identity_b, "probe_steps": steps_a})

        common = {
            parameter: 0.5 * (run_a[parameter].float() + run_b[parameter].float())
            for parameter in LAYER6_PARAMETERS
        }
        summaries["layer6_common"], completed = audit_path(
            path_name="layer6_common",
            states=common,
            function=function,
            primals=primals,
            parameter_shapes=parameter_shapes,
            component_count=args.components,
            cg_iterations=args.cg_iterations,
            cg_tolerance=args.cg_tolerance,
            relative_ridge=args.relative_ridge,
            device=args.device,
            rows=rows,
            progress_path=progress_path,
            completed_before=completed,
            total_solves=total_solves,
        )
        del common

        innovation = {
            parameter: 0.5 * (run_b[parameter].float() - run_a[parameter].float())
            for parameter in LAYER6_PARAMETERS
        }
        summaries["layer6_stream_b_innovation"], completed = audit_path(
            path_name="layer6_stream_b_innovation",
            states=innovation,
            function=function,
            primals=primals,
            parameter_shapes=parameter_shapes,
            component_count=args.components,
            cg_iterations=args.cg_iterations,
            cg_tolerance=args.cg_tolerance,
            relative_ridge=args.relative_ridge,
            device=args.device,
            rows=rows,
            progress_path=progress_path,
            completed_before=completed,
            total_solves=total_solves,
        )
        del innovation, run_a, run_b
    else:
        del trajectory

    torch.cuda.synchronize()
    minimum_pc = min(row["path_energy_capture"] for row in rows)
    gates: dict[str, Any] = {
        "minimum_each_pc_capture": minimum_pc,
        "minimum_each_pc_pass": minimum_pc >= 0.10,
    }
    if args.preflight:
        classification = "PREFLIGHT"
    else:
        gates.update(
            {
                "joint_weighted_pass": summaries["joint_centered"]["weighted_observed_capture"]
                >= 0.50,
                "late_weighted_pass": summaries["joint_late"]["weighted_observed_capture"]
                >= 0.20,
                "common_weighted_pass": summaries["layer6_common"]["weighted_observed_capture"]
                >= 0.50,
                "innovation_weighted_pass": summaries[
                    "layer6_stream_b_innovation"
                ]["weighted_observed_capture"]
                >= 0.20,
            }
        )
        full_pass = all(bool(value) for key, value in gates.items() if key.endswith("_pass"))
        joint_capture = summaries["joint_centered"]["weighted_observed_capture"]
        classification = (
            "REPRESENTATION_PASS"
            if full_pass
            else ("RETAINED_HYBRID_COMPONENT" if joint_capture >= 0.20 else "REJECTED_ONE_MACROSTEP")
        )

    accounting_path = output / "accounting.json"
    projection_path = output / "projection.csv"
    summary_path = output / "summary.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
    write_csv(projection_path, rows)
    summary_path.write_text(
        json.dumps({"classification": classification, "gates": gates, "paths": summaries}, indent=2, sort_keys=True)
        + "\n"
    )
    progress_path.write_text(
        json.dumps(
            {"state": "finished", "completed": completed, "total": total_solves, "classification": classification},
            sort_keys=True,
        )
        + "\n"
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_synthetic_muon_program_full_audit_v1",
        "classification": classification,
        "preflight": args.preflight,
        "accounting": accounting,
        "plan": plan,
        "prompt_manifest": prompt_manifest,
        "function_manifest": function_manifest,
        "identities": identities,
        "w0_storage_matches": w0_matches,
        "summaries": summaries,
        "gates": gates,
        "execution": {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
            "entrypoint": str(script),
            "entrypoint_sha256": sha256(script),
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "plan_path": str(args.plan),
            "plan_sha256": sha256(args.plan),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": time.time() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": sha256(accounting_path)},
            "projection": {"path": str(projection_path), "sha256": sha256(projection_path)},
            "progress": {"path": str(progress_path), "sha256": sha256(progress_path)},
            "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        },
        "limitations": [
            "This is a local tangent representation audit, not prompt optimization or CE.",
            "The six measured matrices are a depth-stratified sample of the 24 replaced MLP matrices.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "finished",
                "classification": classification,
                "metadata": str(metadata_path),
                "summaries": summaries,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
