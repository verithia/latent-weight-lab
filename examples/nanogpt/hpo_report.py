from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def latest_for_run(log_dir: Path, run_name: str) -> tuple[Path, Path, Path] | None:
    latest = log_dir / f"{run_name}_latest"
    if not latest.exists():
        return None
    lines = latest.read_text().splitlines()
    if len(lines) < 3:
        return None
    return Path(lines[0]), Path(lines[1]), Path(lines[2])


def parse_eval(train_log: Path) -> tuple[int | None, float | None, float | None]:
    if not train_log.exists():
        return None, None, None
    text = train_log.read_text(errors="replace")
    matches = re.findall(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)", text)
    if not matches:
        return None, None, None
    step, train, val = matches[-1]
    return int(step), float(train), float(val)


def parse_peak(smi_log: Path) -> int | None:
    if not smi_log.exists():
        return None
    peak = 0
    for line in smi_log.read_text(errors="replace").splitlines():
        if "," not in line:
            continue
        try:
            peak = max(peak, int(line.split(",", 1)[0].strip()))
        except ValueError:
            pass
    return peak


def parse_status(status_path: Path) -> dict[str, str]:
    if not status_path.exists():
        return {}
    out = {}
    for line in status_path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def load_config(config_dir: Path, run_name: str) -> dict:
    matches = sorted(config_dir.glob(f"{run_name}.json"))
    if not matches:
        suffix = run_name.removeprefix("y800_hpo_r2_mlp_100m_")
        matches = sorted(config_dir.glob(f"y800_hpo_r2_mlp_100m_{suffix}.json"))
    if not matches:
        suffix = run_name.removeprefix("y800_hpo_full_r2_100m_")
        matches = sorted(config_dir.glob(f"y800_hpo_full_r2_100m_{suffix}.json"))
    if not matches:
        return {}
    return json.loads(matches[0].read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="/root/userdata/MappingNetworks/logs")
    parser.add_argument("--config-dir", default="/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs")
    parser.add_argument("--output", default="/root/userdata/MappingNetworks/metrics/hpo_r2_mlp_100m_summary.csv")
    parser.add_argument("run_names", nargs="+")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    config_dir = Path(args.config_dir)
    rows = []
    for run_name in args.run_names:
        paths = latest_for_run(log_dir, run_name)
        if paths is None:
            continue
        train_log, smi_log, status_path = paths
        step, train_loss, val_loss = parse_eval(train_log)
        status = parse_status(status_path)
        cfg = load_config(config_dir, run_name)
        rows.append(
            {
                "run": run_name,
                "stage": status.get("stage", ""),
                "gpu": status.get("gpu_id", ""),
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "peak_vram_mib": parse_peak(smi_log),
                "targets": "+".join(cfg.get("block_fht_targets", [])),
                "latent_ratio": cfg.get("block_fht_latent_ratio", ""),
                "lr": cfg.get("learning_rate", ""),
                "min_lr": cfg.get("min_lr", ""),
                "weight_decay": cfg.get("weight_decay", ""),
            }
        )
    rows.sort(key=lambda row: (float("inf") if row["val_loss"] is None else row["val_loss"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "stage", "gpu", "step", "train_loss", "val_loss", "peak_vram_mib", "targets", "latent_ratio", "lr", "min_lr", "weight_decay"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['run']} stage={row['stage']} val={row['val_loss']} train={row['train_loss']} "
            f"targets={row['targets']} latent={row['latent_ratio']} lr={row['lr']} wd={row['weight_decay']} peak={row['peak_vram_mib']}"
        )


if __name__ == "__main__":
    main()
