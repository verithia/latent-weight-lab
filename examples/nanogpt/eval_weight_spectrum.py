from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from examples.nanogpt.model import GPT, GPTConfig


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() == "all":
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def load_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def spectrum_metrics(weight: torch.Tensor) -> dict[str, float]:
    w = weight.detach().float().cpu()
    singular = torch.linalg.svdvals(w)
    energy = singular.square()
    total = energy.sum()
    if total <= 0:
        return {"soft_rank": 0.0, "hard_rank": 0.0, "stable_rank": 0.0, "top1_energy": 0.0, "top10_energy": 0.0}
    p = energy / total
    soft = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())
    hard = 1.0 / p.square().sum()
    stable = total / singular.max().square().clamp_min(1e-30)
    top1 = p[0]
    top10 = p[: min(10, p.numel())].sum()
    return {
        "soft_rank": float(soft.item()),
        "hard_rank": float(hard.item()),
        "stable_rank": float(stable.item()),
        "top1_energy": float(top1.item()),
        "top10_energy": float(top10.item()),
        "fro_norm": float(torch.linalg.matrix_norm(w).item()),
        "spectral_norm": float(singular.max().item()),
        "mean": float(w.mean().item()),
        "std": float(w.std().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="3,6,9")
    parser.add_argument("--target", choices=["c_fc", "c_proj"], default="c_fc")
    args = parser.parse_args()

    model = load_model(Path(args.checkpoint), args.device)
    layers = parse_layers(args.layers)
    layer_ids = layers if layers is not None else list(range(len(model.transformer.h)))
    rows = []
    for layer in layer_ids:
        module = getattr(model.transformer.h[layer].mlp, args.target)
        metrics = spectrum_metrics(module.weight)
        rows.append({"layer": layer, "target": args.target, **metrics})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_metrics = [key for key in rows[0] if key not in {"layer", "target"}]
    summary = {"target": args.target, "layers": len(rows)}
    for metric in summary_metrics:
        summary[f"{metric}_mean"] = float(np.mean([float(row[metric]) for row in rows]))
    with Path(args.summary).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
