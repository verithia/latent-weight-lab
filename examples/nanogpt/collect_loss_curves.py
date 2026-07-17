from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EVAL_RE = re.compile(r"^step\s+(\d+):\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")
ITER_RE = re.compile(r"^iter\s+(\d+):\s+loss\s+([0-9.]+),\s+time\s+([0-9.]+)ms")


def read_latest_log(log_dir: Path, run_name: str) -> Path:
    latest = log_dir / f"{run_name}_latest"
    if latest.exists():
        first = latest.read_text(encoding="utf-8").splitlines()[0]
        path = Path(first)
        if path.exists():
            return path
        local = log_dir / path.name
        if local.exists():
            return local
        return path
    matches = sorted(log_dir.glob(f"{run_name}_*.log"))
    matches = [path for path in matches if not path.name.endswith("_smi.log")]
    if not matches:
        raise FileNotFoundError(f"no log found for {run_name} in {log_dir}")
    return matches[-1]


def parse_log(path: Path, run_name: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    eval_rows: list[dict[str, str]] = []
    iter_rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVAL_RE.match(line)
        if match:
            step, train_loss, val_loss = match.groups()
            eval_rows.append(
                {
                    "run": run_name,
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "log_path": str(path),
                }
            )
            continue
        match = ITER_RE.match(line)
        if match:
            step, loss, ms = match.groups()
            iter_rows.append(
                {
                    "run": run_name,
                    "iter": step,
                    "loss": loss,
                    "time_ms": ms,
                    "log_path": str(path),
                }
            )
    return eval_rows, iter_rows


def load_baseline(path: Path) -> dict[int, tuple[float, float]]:
    baseline: dict[int, tuple[float, float]] = {}
    if not path.exists():
        return baseline
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            baseline[int(row["step"])] = (float(row["train_loss"]), float(row["val_loss"]))
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--iter-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--baseline-eval", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    baseline = load_baseline(Path(args.baseline_eval)) if args.baseline_eval else {}
    all_eval: list[dict[str, str]] = []
    all_iter: list[dict[str, str]] = []
    summary: list[dict[str, str]] = []
    for run_name in args.runs:
        log_path = read_latest_log(log_dir, run_name)
        eval_rows, iter_rows = parse_log(log_path, run_name)
        for row in eval_rows:
            step = int(row["step"])
            if step in baseline:
                base_train, base_val = baseline[step]
                row["vanilla_train_loss"] = f"{base_train:.4f}"
                row["vanilla_val_loss"] = f"{base_val:.4f}"
                row["train_delta_vs_vanilla"] = f"{float(row['train_loss']) - base_train:.4f}"
                row["val_delta_vs_vanilla"] = f"{float(row['val_loss']) - base_val:.4f}"
            else:
                row["vanilla_train_loss"] = ""
                row["vanilla_val_loss"] = ""
                row["train_delta_vs_vanilla"] = ""
                row["val_delta_vs_vanilla"] = ""
        all_eval.extend(eval_rows)
        all_iter.extend(iter_rows)
        if eval_rows:
            final = eval_rows[-1]
            summary.append(
                {
                    "run": run_name,
                    "final_step": final["step"],
                    "final_train_loss": final["train_loss"],
                    "final_val_loss": final["val_loss"],
                    "final_val_delta_vs_vanilla": final.get("val_delta_vs_vanilla", ""),
                    "eval_points": str(len(eval_rows)),
                    "iter_points": str(len(iter_rows)),
                    "log_path": str(log_path),
                }
            )

    Path(args.eval_output).parent.mkdir(parents=True, exist_ok=True)
    eval_fields = [
        "run",
        "step",
        "train_loss",
        "val_loss",
        "vanilla_train_loss",
        "vanilla_val_loss",
        "train_delta_vs_vanilla",
        "val_delta_vs_vanilla",
        "log_path",
    ]
    with Path(args.eval_output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=eval_fields)
        writer.writeheader()
        writer.writerows(all_eval)
    with Path(args.iter_output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "iter", "loss", "time_ms", "log_path"])
        writer.writeheader()
        writer.writerows(all_iter)
    with Path(args.summary_output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "final_step",
                "final_train_loss",
                "final_val_loss",
                "final_val_delta_vs_vanilla",
                "eval_points",
                "iter_points",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
