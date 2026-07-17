from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_KEYS = [
    "ce",
    "ppl",
    "top1__uniform",
    "top3__uniform",
    "top5__uniform",
    "mrr__uniform",
    "rank__uniform",
    "entropy__uniform",
    "wrong_confidence__uniform",
    "top1__inv_frequency",
    "top5__entropy",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    args = parser.parse_args()

    rows = []
    for item in args.inputs:
        path = Path(item)
        data = json.loads(path.read_text())
        metrics = data["metrics"]
        row = {
            "name": path.stem.removeprefix("proxy_"),
            "tokens": data.get("tokens", ""),
            "checkpoint": data.get("checkpoint", ""),
        }
        for key in args.keys:
            row[key] = metrics.get(key, "")
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "tokens", *args.keys, "checkpoint"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    widths = {field: max(len(field), *(len(f"{row.get(field, ''):.4g}") if isinstance(row.get(field), float) else len(str(row.get(field, ""))) for row in rows)) for field in fieldnames[:-1]}
    print(" | ".join(field.ljust(widths[field]) for field in fieldnames[:-1]))
    print(" | ".join("-" * widths[field] for field in fieldnames[:-1]))
    for row in rows:
        cells = []
        for field in fieldnames[:-1]:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            cells.append(str(value).ljust(widths[field]))
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
