from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from examples.nanogpt.model import GPT, GPTConfig


REGIMES = ("HEAD", "MID", "TAIL")


def token_regimes(data_dir: Path, vocab_size: int) -> np.ndarray:
    train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(train), minlength=vocab_size).astype(np.int64)
    order = np.argsort(-counts)
    total = int(counts.sum())
    cumsum = np.cumsum(counts[order])
    head_cut = total / 3.0
    mid_cut = 2.0 * total / 3.0
    regime = np.full(vocab_size, 2, dtype=np.int64)
    regime[order[cumsum <= mid_cut]] = 1
    regime[order[cumsum <= head_cut]] = 0
    return regime


def get_batch(data_dir: Path, split: str, batch_size: int, block_size: int, device: str) -> torch.Tensor:
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    return x.to(device)


def effective_ranks(values: torch.Tensor) -> tuple[float, float, float, int]:
    if values.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), int(values.shape[0])
    x = values.float()
    x = x - x.mean(dim=0, keepdim=True)
    # Computing singular values of centered samples is cheaper than materializing D x D covariance.
    singular = torch.linalg.svdvals(x)
    eig = singular.square()
    eig_sum = eig.sum()
    if eig_sum <= 0:
        return 0.0, 0.0, 0.0, int(values.shape[0])
    p = eig / eig_sum
    soft = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())
    hard = 1.0 / p.square().sum()
    asym = torch.log(soft) - torch.log(hard)
    return float(soft.item()), float(hard.item()), float(asym.item()), int(values.shape[0])


class SpectralCollector:
    def __init__(self, model: GPT, regimes: np.ndarray, sample_cap: int, layers: list[int] | None) -> None:
        self.model = model
        self.regimes = regimes
        self.sample_cap = int(sample_cap)
        self.layers = set(layers) if layers is not None else None
        self.current_tokens: torch.Tensor | None = None
        self.samples: dict[tuple[int, str, str], list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[tuple[int, str, str], int] = defaultdict(int)
        self.handles = []
        for idx, block in enumerate(model.transformer.h):
            if self.layers is not None and idx not in self.layers:
                continue
            self.handles.append(block.mlp.c_fc.register_forward_hook(self._hook(idx, "pre")))
            self.handles.append(block.mlp.gelu.register_forward_hook(self._hook(idx, "post")))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer: int, point: str):
        def collect(_module, _inputs, output):
            if self.current_tokens is None:
                return
            acts = output.detach().float().reshape(-1, output.shape[-1]).cpu()
            tokens = self.current_tokens.reshape(-1).detach().cpu().numpy()
            token_regimes = self.regimes[tokens]
            for regime_idx, regime_name in enumerate(REGIMES):
                mask = token_regimes == regime_idx
                if not np.any(mask):
                    continue
                key = (layer, point, regime_name)
                remaining = self.sample_cap - self.counts[key]
                if remaining <= 0:
                    continue
                selected_idx = np.flatnonzero(mask)
                if selected_idx.size > remaining:
                    selected_idx = np.random.choice(selected_idx, size=remaining, replace=False)
                selected = acts[torch.from_numpy(selected_idx)]
                self.samples[key].append(selected)
                self.counts[key] += int(selected.shape[0])

        return collect

    def full(self) -> bool:
        if not self.counts:
            return False
        expected_layers = self.layers if self.layers is not None else set(range(len(self.model.transformer.h)))
        for layer in expected_layers:
            for point in ("pre", "post"):
                for regime in REGIMES:
                    if self.counts[(layer, point, regime)] < self.sample_cap:
                        return False
        return True

    def rows(self) -> list[dict[str, object]]:
        rows = []
        for key in sorted(self.samples):
            layer, point, regime = key
            values = torch.cat(self.samples[key], dim=0)[: self.sample_cap]
            soft, hard, asym, count = effective_ranks(values)
            rows.append(
                {
                    "layer": layer,
                    "point": point,
                    "regime": regime,
                    "samples": count,
                    "soft_rank": soft,
                    "hard_rank": hard,
                    "hard_soft_asym": asym,
                }
            )
        return rows


def load_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() == "all":
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--sample-cap", type=int, default=1024)
    parser.add_argument("--layers", default="3,6,9")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, args.device)
    regimes = token_regimes(data_dir, model.config.vocab_size)
    collector = SpectralCollector(model, regimes, args.sample_cap, parse_layers(args.layers))
    try:
        with torch.no_grad():
            for _ in range(args.max_batches):
                x = get_batch(data_dir, args.split, args.batch_size, args.block_size, args.device)
                collector.current_tokens = x
                model(x, None)
                if collector.full():
                    break
    finally:
        collector.close()

    rows = collector.rows()
    fields = ["layer", "point", "regime", "samples", "soft_rank", "hard_rank", "hard_soft_asym"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if args.summary is not None:
        summary_path = Path(args.summary)
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["point"]), str(row["regime"]))].append(row)
        summary = []
        for (point, regime), group in sorted(grouped.items()):
            summary.append(
                {
                    "point": point,
                    "regime": regime,
                    "layers": len(group),
                    "soft_rank_mean": float(np.nanmean([float(row["soft_rank"]) for row in group])),
                    "hard_rank_mean": float(np.nanmean([float(row["hard_rank"]) for row in group])),
                    "hard_soft_asym_mean": float(np.nanmean([float(row["hard_soft_asym"]) for row in group])),
                }
            )
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
