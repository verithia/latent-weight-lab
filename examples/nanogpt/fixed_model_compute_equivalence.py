#!/usr/bin/env python3
"""Estimate phase-local alpha and dense compute equivalence from loss logs.

This implements the project SOP in
``notes/active/fixed-model-compute-equivalence-sop-20260804.md``.  Compute may
be represented by steps when both curves use the same constant tokens/update.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


EVAL_RE = re.compile(
    r"^step\s+(?P<step>\d+):\s+train loss\s+(?P<train>[0-9.eE+-]+),\s+"
    r"val loss\s+(?P<val>[0-9.eE+-]+)"
)


@dataclass(frozen=True)
class EvalPoint:
    compute: float
    validation_ce: float


def parse_eval_log(path: Path) -> list[EvalPoint]:
    """Return the last logged value for each positive evaluation step."""

    by_step: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVAL_RE.match(line.strip())
        if match and int(match.group("step")) > 0:
            by_step[int(match.group("step"))] = float(match.group("val"))
    points = [EvalPoint(float(step), loss) for step, loss in sorted(by_step.items())]
    if len(points) < 2:
        raise ValueError(f"need at least two positive-step evaluations: {path}")
    return points


def _validate(points: Sequence[EvalPoint]) -> None:
    if len(points) < 2:
        raise ValueError("need at least two points")
    if any(point.compute <= 0 or not math.isfinite(point.validation_ce) for point in points):
        raise ValueError("compute must be positive and validation CE finite")
    if any(left.compute >= right.compute for left, right in zip(points, points[1:])):
        raise ValueError("compute coordinates must be strictly increasing")


def local_slope(points: Sequence[EvalPoint], window: int) -> dict[str, float | int]:
    """Fit ``L = intercept - s*ln(C)`` over the terminal ``window`` points."""

    _validate(points)
    if window < 2 or window > len(points):
        raise ValueError("window must contain between two and all points")
    selected = points[-window:]
    xs = [math.log(point.compute) for point in selected]
    ys = [point.validation_ce for point in selected]
    x_bar = statistics.fmean(xs)
    y_bar = statistics.fmean(ys)
    denominator = sum((x - x_bar) ** 2 for x in xs)
    regression_slope = sum(
        (x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)
    ) / denominator
    loss_slope = -regression_slope
    return {
        "window": window,
        "loss_slope_per_log_compute": loss_slope,
        "alpha_eff_total_loss": loss_slope / points[-1].validation_ce,
    }


def invert_monotone_curve(
    points: Sequence[EvalPoint], target_ce: float, *, allow_one_interval_extrapolation: bool = False
) -> tuple[float, str]:
    """Invert a decreasing curve by interpolation in ``(ln C, L)``."""

    _validate(points)
    if any(left.validation_ce < right.validation_ce for left, right in zip(points, points[1:])):
        raise ValueError("validation curve must be monotonically non-increasing")
    for left, right in zip(points, points[1:]):
        if left.validation_ce >= target_ce >= right.validation_ce:
            fraction = (target_ce - left.validation_ce) / (
                right.validation_ce - left.validation_ce
            )
            log_compute = math.log(left.compute) + fraction * (
                math.log(right.compute) - math.log(left.compute)
            )
            return math.exp(log_compute), "interpolation"
    if not allow_one_interval_extrapolation:
        raise ValueError("target CE is outside the sampled dense-loss range")
    pair = points[:2] if target_ce > points[0].validation_ce else points[-2:]
    left, right = pair
    fraction = (target_ce - left.validation_ce) / (
        right.validation_ce - left.validation_ce
    )
    if not -1.0 <= fraction <= 2.0:
        raise ValueError("target exceeds the one-adjacent-interval extrapolation limit")
    log_compute = math.log(left.compute) + fraction * (
        math.log(right.compute) - math.log(left.compute)
    )
    return math.exp(log_compute), "one_interval_extrapolation"


def terminal_dense_penalty(
    dense: Sequence[EvalPoint], candidate: Sequence[EvalPoint]
) -> dict[str, float | str]:
    _validate(candidate)
    terminal = candidate[-1]
    dense_match, method = invert_monotone_curve(
        dense, terminal.validation_ce, allow_one_interval_extrapolation=True
    )
    return {
        "candidate_terminal_compute": terminal.compute,
        "candidate_terminal_validation_ce": terminal.validation_ce,
        "dense_match_compute": dense_match,
        "candidate_over_dense_compute": terminal.compute / dense_match,
        "method": method,
    }


def common_loss_ratios(
    dense: Sequence[EvalPoint], candidate: Sequence[EvalPoint], samples: int = 101
) -> dict[str, float | int]:
    """Return AttnRes-style horizontal ratios over the curves' loss overlap."""

    _validate(dense)
    _validate(candidate)
    high = min(dense[0].validation_ce, candidate[0].validation_ce)
    low = max(dense[-1].validation_ce, candidate[-1].validation_ce)
    if high <= low:
        raise ValueError("curves have no sampled common-loss interval")
    losses = [low + (high - low) * index / (samples - 1) for index in range(samples)]
    ratios = []
    for loss in losses:
        dense_compute, _ = invert_monotone_curve(dense, loss)
        candidate_compute, _ = invert_monotone_curve(candidate, loss)
        ratios.append(candidate_compute / dense_compute)
    return {
        "samples": samples,
        "loss_low": low,
        "loss_high": high,
        "median_candidate_over_dense_compute": statistics.median(ratios),
        "min_candidate_over_dense_compute": min(ratios),
        "max_candidate_over_dense_compute": max(ratios),
    }


def summarize(dense: Sequence[EvalPoint], candidate: Sequence[EvalPoint] | None) -> dict:
    windows = [window for window in (2, 3, 4) if window <= len(dense)]
    payload = {
        "schema_version": "fixed_model_compute_equivalence_v1",
        "dense_terminal_validation_ce": dense[-1].validation_ce,
        "dense_local_fits": [local_slope(dense, window) for window in windows],
    }
    if candidate is not None:
        payload["terminal_dense_penalty"] = terminal_dense_penalty(dense, candidate)
        payload["common_loss_ratio"] = common_loss_ratios(dense, candidate)
    return payload


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = summarize(
        parse_eval_log(args.dense_log),
        parse_eval_log(args.candidate_log) if args.candidate_log else None,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
