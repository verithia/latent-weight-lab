from __future__ import annotations

"""Create and validate immutable descriptive dense scaling-fit artifacts.

The utility intentionally uses only the Python standard library so it can run
where launch/configuration checks run.  It fits the registered four-rung dense
ladder only; acceptance is a separate, explicit operator action.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = "dense_scaling_fit_v1"
RESULT_RECORD_SCHEMA_VERSION = "dense_terminal_result_v1"
EQUATION = "L(C) = E + A * C^-alpha"
OBJECTIVE = "sum_squared_error_terminal_held_out_nll"
UNITS = "terminal held-out NLL (nats/token)"
COST_UNITS = "materialized active parameters"
REQUIRED_TIERS = ("124m", "350m", "690m", "985m")
REQUIRED_IDENTITY_KEYS = (
    "config_sha256",
    "source_hashes",
    "data_manifest_sha256",
    "fixed_eval_indices_sha256",
    "eval_protocol_id",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be an exact SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be an exact SHA-256 hex string") from exc
    return value


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def validate_terminal_record(record: object) -> dict[str, Any]:
    """Validate a terminal dense record and return its immutable fit input."""
    if not isinstance(record, dict):
        raise ValueError("dense terminal result must be a JSON object")
    if record.get("schema_version") != RESULT_RECORD_SCHEMA_VERSION:
        raise ValueError("dense terminal result has an unsupported schema_version")
    if record.get("acceptance_state") != "ACCEPTED":
        raise ValueError("dense terminal result is not explicitly ACCEPTED")
    if record.get("family") != "dense":
        raise ValueError("dense terminal result family must be dense")
    tier = record.get("model_tier")
    if tier not in REQUIRED_TIERS:
        raise ValueError("dense terminal result has an unregistered model_tier")

    identity = record.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("dense terminal result identity is required")
    for key in REQUIRED_IDENTITY_KEYS:
        if key not in identity:
            raise ValueError(f"dense terminal result identity is missing {key}")
    _require_sha256(identity["config_sha256"], "identity.config_sha256")
    _require_sha256(identity["data_manifest_sha256"], "identity.data_manifest_sha256")
    _require_sha256(identity["fixed_eval_indices_sha256"], "identity.fixed_eval_indices_sha256")
    if not isinstance(identity["source_hashes"], dict) or not identity["source_hashes"]:
        raise ValueError("identity.source_hashes must be a non-empty object")
    for source_name, source_hash in identity["source_hashes"].items():
        if not isinstance(source_name, str):
            raise ValueError("identity.source_hashes keys must be strings")
        _require_sha256(source_hash, f"identity.source_hashes[{source_name!r}]")
    if not isinstance(identity["eval_protocol_id"], str) or not identity["eval_protocol_id"]:
        raise ValueError("identity.eval_protocol_id must be a non-empty string")

    return {
        "model_tier": tier,
        "terminal_held_out_nll": _finite_number(record.get("terminal_held_out_nll"), "terminal_held_out_nll"),
        "cost": _finite_number(record.get("estimated_active_params"), "estimated_active_params", positive=True),
        "cost_units": COST_UNITS,
        "scheduled_tokens": _finite_number(record.get("scheduled_tokens"), "scheduled_tokens", positive=True),
        "scheduled_tpp": _finite_number(record.get("scheduled_tpp"), "scheduled_tpp", positive=True),
        "identity": identity,
    }


def _fit_at_e(points: list[dict[str, Any]], e_value: float) -> dict[str, float] | None:
    xs = [math.log(point["cost"]) for point in points]
    residuals = [point["terminal_held_out_nll"] - e_value for point in points]
    if any(value <= 0.0 for value in residuals):
        return None
    ys = [math.log(value) for value in residuals]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator
    alpha = -slope
    if not math.isfinite(alpha) or alpha <= 0.0:
        return None
    log_a = y_mean - slope * x_mean
    a_value = math.exp(log_a)
    predictions = [e_value + a_value * point["cost"] ** (-alpha) for point in points]
    sse = sum((prediction - point["terminal_held_out_nll"]) ** 2 for prediction, point in zip(predictions, points, strict=True))
    if not math.isfinite(sse):
        return None
    return {"A": a_value, "alpha": alpha, "E": e_value, "sse": sse}


def fit_scaling_law(
    points: list[dict[str, Any]],
    *,
    e_lower: float | None = None,
    e_upper: float | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit ``L(C)=E+A*C^-alpha`` with deterministic bounded scalar search."""
    if len(points) < 3:
        raise ValueError("at least three terminal inputs are required for a scaling fit")
    min_loss = min(point["terminal_held_out_nll"] for point in points)
    spread = max(point["terminal_held_out_nll"] for point in points) - min_loss
    lower = float(e_lower) if e_lower is not None else min_loss - max(10.0, 10.0 * max(spread, 1.0))
    upper = float(e_upper) if e_upper is not None else min_loss - max(1e-12, abs(min_loss) * 1e-12)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper or upper >= min_loss:
        raise ValueError("invalid E bounds; require finite lower < upper < min terminal NLL")

    # A coarse deterministic scan brackets the best feasible region. A golden
    # search then refines it without requiring NumPy/SciPy.
    grid_size = 1024
    best_index = -1
    best_fit: dict[str, float] | None = None
    candidates: list[dict[str, float] | None] = []
    for index in range(grid_size + 1):
        e_value = lower + (upper - lower) * index / grid_size
        fitted = _fit_at_e(points, e_value)
        candidates.append(fitted)
        if fitted is not None and (best_fit is None or fitted["sse"] < best_fit["sse"]):
            best_fit = fitted
            best_index = index
    if best_fit is None:
        raise ValueError("fit constraints admit no positive-A, positive-alpha solution")

    left = lower + (upper - lower) * max(0, best_index - 1) / grid_size
    right = lower + (upper - lower) * min(grid_size, best_index + 1) / grid_size
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    for _ in range(100):
        first = right - (right - left) / phi
        second = left + (right - left) / phi
        first_fit = _fit_at_e(points, first)
        second_fit = _fit_at_e(points, second)
        first_sse = math.inf if first_fit is None else first_fit["sse"]
        second_sse = math.inf if second_fit is None else second_fit["sse"]
        if first_sse <= second_sse:
            right = second
            if first_fit is not None and first_fit["sse"] < best_fit["sse"]:
                best_fit = first_fit
        else:
            left = first
            if second_fit is not None and second_fit["sse"] < best_fit["sse"]:
                best_fit = second_fit

    constraints = {
        "point_count": len(points),
        "cost_field": "estimated_active_params",
        "cost_units": COST_UNITS,
        "A": "> 0",
        "alpha": "> 0",
        "E": "strictly less than every observed terminal held-out NLL",
        "E_search_bounds": [lower, upper],
        "fit_method": "bounded deterministic E scan plus golden-section refinement; log-linear A/alpha solve at each E",
    }
    return best_fit, constraints


def _leave_one_out(points: list[dict[str, Any]], full_fit: dict[str, float], e_bounds: tuple[float, float]) -> list[dict[str, Any]]:
    sensitivity = []
    for point in points:
        retained = [candidate for candidate in points if candidate["model_tier"] != point["model_tier"]]
        fit, _ = fit_scaling_law(retained, e_lower=e_bounds[0], e_upper=e_bounds[1])
        sensitivity.append({
            "omitted_model_tier": point["model_tier"],
            "retained_model_tiers": [candidate["model_tier"] for candidate in retained],
            "coefficients": {key: fit[key] for key in ("A", "alpha", "E")},
            "objective_value": fit["sse"],
            "delta_from_full": {key: fit[key] - full_fit[key] for key in ("A", "alpha", "E", "sse")},
        })
    return sensitivity


def build_artifact(record_paths: list[Path], *, accepted: bool, e_lower: float | None = None, e_upper: float | None = None) -> dict[str, Any]:
    if len(record_paths) != 4:
        raise ValueError("exactly four accepted dense terminal result records are required")
    inputs = []
    for path in record_paths:
        record = json.loads(path.read_text())
        point = validate_terminal_record(record)
        point["record_path"] = str(path.resolve())
        point["record_sha256"] = sha256_file(path)
        inputs.append(point)
    if {point["model_tier"] for point in inputs} != set(REQUIRED_TIERS):
        raise ValueError("the four records must cover exactly 124m, 350m, 690m, and 985m")
    inputs.sort(key=lambda point: REQUIRED_TIERS.index(point["model_tier"]))

    fitted, constraints = fit_scaling_law(inputs, e_lower=e_lower, e_upper=e_upper)
    leave_one_out = _leave_one_out(
        inputs,
        fitted,
        (float(constraints["E_search_bounds"][0]), float(constraints["E_search_bounds"][1])),
    )
    reference = inputs[0]
    derived_dense_factors = []
    for point in inputs:
        fitted_loss = fitted["E"] + fitted["A"] * point["cost"] ** (-fitted["alpha"])
        derived_dense_factors.append({
            "model_tier": point["model_tier"],
            "cost_factor_vs_124m": point["cost"] / reference["cost"],
            "scheduled_token_factor_vs_124m": point["scheduled_tokens"] / reference["scheduled_tokens"],
            "fitted_terminal_held_out_nll": fitted_loss,
            "terminal_nll_residual": point["terminal_held_out_nll"] - fitted_loss,
        })
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "registered_dense_scaling_fit",
        "state": "ACCEPTED" if accepted else "DRAFT_NOT_ACCEPTED",
        "acceptance": {
            "explicit_cli_flag_required": True,
            "created_with_accept_flag": bool(accepted),
        },
        "equation": EQUATION,
        "objective": OBJECTIVE,
        "units": UNITS,
        "coefficients": {key: fitted[key] for key in ("A", "alpha", "E")},
        "objective_value": fitted["sse"],
        "fit_constraints": constraints,
        "terminal_inputs": inputs,
        "derived_dense_factors": derived_dense_factors,
        "sensitivity": {
            "leave_one_out": leave_one_out,
            "warning": "Four matched rungs provide an initial descriptive fit only; do not claim a robust exponent or extrapolation.",
        },
    }


def validate_dense_fit_artifact(artifact: object) -> dict[str, Any]:
    """Validate the strict contract required before a candidate can launch."""
    if not isinstance(artifact, dict):
        raise ValueError("dense-fit artifact must be a JSON object")
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("dense-fit artifact schema/version is incompatible")
    if artifact.get("artifact_kind") != "registered_dense_scaling_fit":
        raise ValueError("dense-fit artifact kind is incompatible")
    if artifact.get("state") != "ACCEPTED":
        raise ValueError("dense-fit artifact is not ACCEPTED")
    if artifact.get("equation") != EQUATION or artifact.get("objective") != OBJECTIVE or artifact.get("units") != UNITS:
        raise ValueError("dense-fit artifact equation/objective/units are incompatible")
    coefficients = artifact.get("coefficients")
    if not isinstance(coefficients, dict):
        raise ValueError("dense-fit artifact is missing coefficients")
    a_value = _finite_number(coefficients.get("A"), "coefficients.A", positive=True)
    alpha = _finite_number(coefficients.get("alpha"), "coefficients.alpha", positive=True)
    e_value = _finite_number(coefficients.get("E"), "coefficients.E")
    terminal_inputs = artifact.get("terminal_inputs")
    if not isinstance(terminal_inputs, list) or len(terminal_inputs) != 4:
        raise ValueError("dense-fit artifact must contain four terminal inputs")
    tiers = set()
    min_loss = math.inf
    for point in terminal_inputs:
        if not isinstance(point, dict):
            raise ValueError("dense-fit terminal input is invalid")
        tier = point.get("model_tier")
        tiers.add(tier)
        _require_sha256(point.get("record_sha256"), "terminal input record_sha256")
        validated = validate_terminal_record({
            "schema_version": RESULT_RECORD_SCHEMA_VERSION,
            "acceptance_state": "ACCEPTED",
            "family": "dense",
            "model_tier": tier,
            "terminal_held_out_nll": point.get("terminal_held_out_nll"),
            "estimated_active_params": point.get("cost"),
            "scheduled_tokens": point.get("scheduled_tokens"),
            "scheduled_tpp": point.get("scheduled_tpp"),
            "identity": point.get("identity"),
        })
        min_loss = min(min_loss, validated["terminal_held_out_nll"])
    if tiers != set(REQUIRED_TIERS):
        raise ValueError("dense-fit terminal inputs do not cover the four registered tiers")
    if e_value >= min_loss:
        raise ValueError("dense-fit artifact E must be below every terminal input")
    constraints = artifact.get("fit_constraints")
    sensitivity = artifact.get("sensitivity")
    if not isinstance(constraints, dict) or constraints.get("point_count") != 4:
        raise ValueError("dense-fit artifact fit constraints are incomplete")
    factors = artifact.get("derived_dense_factors")
    if not isinstance(factors, list) or len(factors) != 4:
        raise ValueError("dense-fit artifact derived dense factors are incomplete")
    if not isinstance(sensitivity, dict) or not isinstance(sensitivity.get("leave_one_out"), list) or len(sensitivity["leave_one_out"]) != 4:
        raise ValueError("dense-fit artifact sensitivity/leave-one-out output is incomplete")
    return {"A": a_value, "alpha": alpha, "E": e_value}


def write_immutable_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Atomically publish once, refusing to replace an existing fit artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable dense-fit artifact: {path}")
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        # link() is atomic and fails if another process created the final name.
        os.link(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="append", required=True, help="accepted dense terminal-result JSON; provide exactly four")
    parser.add_argument("--output", required=True, help="new immutable dense-fit artifact path")
    parser.add_argument("--accept", action="store_true", help="explicitly mark the created fit artifact ACCEPTED")
    parser.add_argument("--e-lower", type=float, default=None)
    parser.add_argument("--e-upper", type=float, default=None)
    args = parser.parse_args()
    artifact = build_artifact(
        [Path(value) for value in args.record],
        accepted=args.accept,
        e_lower=args.e_lower,
        e_upper=args.e_upper,
    )
    validate_dense_fit_artifact(artifact) if args.accept else None
    output = Path(args.output)
    write_immutable_artifact(output, artifact)
    print(json.dumps({
        "output": str(output.resolve()),
        "sha256": sha256_file(output),
        "state": artifact["state"],
        "coefficients": artifact["coefficients"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
