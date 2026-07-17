from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.model import GPT, GPTConfig


CORE_NAMES = [
    "ce",
    "top1",
    "top2",
    "top3",
    "top5",
    "entropy",
    "rank",
    "mrr",
    "margin",
    "wrong_confidence",
]
WEIGHT_NAMES = ["uniform", "prob", "disagreement", "entropy", "inv_entropy", "frequency", "inv_frequency", "gaussian_nll"]


def load_tokens(path: Path, max_tokens: int, offset: int) -> np.ndarray:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    stop = min(len(data), offset + max_tokens)
    if stop - offset < 2:
        raise ValueError("not enough tokens for proxy eval")
    return np.asarray(data[offset:stop], dtype=np.int64)


def load_token_traces(path: Path, max_tokens: int, offset: int) -> list[np.ndarray]:
    traces: list[np.ndarray] = []
    skipped = 0
    used = 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        tokens = obj.get("tokens") or obj.get("token_ids") or obj.get("ids")
        if tokens is None:
            raise ValueError(f"JSONL row missing tokens/token_ids/ids: {line[:120]}")
        arr = np.asarray(tokens, dtype=np.int64)
        if arr.size < 2:
            continue
        if skipped + arr.size <= offset:
            skipped += arr.size
            continue
        if skipped < offset:
            arr = arr[offset - skipped :]
            skipped = offset
        remaining = max_tokens + 1 - used
        if remaining <= 1:
            break
        arr = arr[:remaining]
        if arr.size >= 2:
            traces.append(arr)
            used += arr.size
    if not traces:
        raise ValueError("no usable token traces found")
    return traces


def iter_batches(tokens: np.ndarray, block_size: int, batch_size: int):
    starts = list(range(0, len(tokens) - block_size - 1, block_size))
    for batch_start in range(0, len(starts), batch_size):
        xs = []
        ys = []
        for start in starts[batch_start : batch_start + batch_size]:
            xs.append(torch.from_numpy(tokens[start : start + block_size].copy()).long())
            ys.append(torch.from_numpy(tokens[start + 1 : start + 1 + block_size].copy()).long())
        if xs:
            yield torch.stack(xs), torch.stack(ys)


def iter_trace_batches(traces: list[np.ndarray], block_size: int, batch_size: int):
    xs = []
    ys = []
    for trace in traces:
        for start in range(0, len(trace) - block_size - 1, block_size):
            xs.append(torch.from_numpy(trace[start : start + block_size].copy()).long())
            ys.append(torch.from_numpy(trace[start + 1 : start + 1 + block_size].copy()).long())
            if len(xs) == batch_size:
                yield torch.stack(xs), torch.stack(ys)
                xs = []
                ys = []
    if xs:
        yield torch.stack(xs), torch.stack(ys)


def frequency_table(tokens: np.ndarray, vocab_size: int) -> torch.Tensor:
    counts = np.bincount(tokens.astype(np.int64), minlength=vocab_size).astype(np.float64)
    freq = counts / max(1.0, counts.max())
    return torch.from_numpy(freq.astype(np.float32))


def frequency_table_from_traces(traces: list[np.ndarray], vocab_size: int) -> torch.Tensor:
    if not traces:
        return torch.ones(vocab_size, dtype=torch.float32)
    targets = [trace[1:] for trace in traces if trace.size > 1]
    return frequency_table(np.concatenate(targets), vocab_size)


def add_weighted(totals: dict[str, float], weight_totals: dict[str, float], core: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]) -> None:
    for cname, values in core.items():
        flat_values = values.float().reshape(-1)
        for wname, weight in weights.items():
            flat_weight = weight.float().reshape(-1).clamp_min(0)
            key = f"{cname}__{wname}"
            totals[key] = totals.get(key, 0.0) + float((flat_values * flat_weight).sum().item())
            weight_totals[key] = weight_totals.get(key, 0.0) + float(flat_weight.sum().item())


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    if args.tokens_jsonl:
        traces = load_token_traces(Path(args.tokens_jsonl), args.max_tokens + 1, args.offset)
        data = None
        freq = frequency_table_from_traces(traces, config.vocab_size).to(args.device)
        batch_iter = iter_trace_batches(traces, config.block_size, args.batch_size)
        source = args.tokens_jsonl
    else:
        if not args.data_bin:
            raise ValueError("one of --data-bin or --tokens-jsonl is required")
        data = load_tokens(Path(args.data_bin), args.max_tokens + 1, args.offset)
        freq = frequency_table(data[1:], config.vocab_size).to(args.device)
        batch_iter = iter_batches(data, config.block_size, args.batch_size)
        source = args.data_bin

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    cache_prepared = False
    if config.block_fht and args.cache_block_fht:
        model.prepare_block_fht_cache(dtype=ptdtype)
        cache_prepared = True

    totals: dict[str, float] = {}
    weight_totals: dict[str, float] = {}
    token_count = 0
    loss_sum = 0.0
    try:
        for x, y in batch_iter:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            with ctx:
                logits, _ = model(x)
            logits = logits.float()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            target_logp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            target_prob = target_logp.exp()
            ce = -target_logp
            loss_sum += float(ce.sum().item())
            token_count += int(y.numel())

            entropy = -(probs * log_probs).sum(dim=-1) / math.log(config.vocab_size)
            top_values, top_indices = torch.topk(probs, k=5, dim=-1)
            top_hits = top_indices.eq(y.unsqueeze(-1))
            top1 = top_hits[..., :1].any(dim=-1).float()
            top2 = top_hits[..., :2].any(dim=-1).float()
            top3 = top_hits[..., :3].any(dim=-1).float()
            top5 = top_hits.any(dim=-1).float()
            target_logits = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            rank = logits.gt(target_logits.unsqueeze(-1)).sum(dim=-1).float() + 1.0
            mrr = rank.reciprocal()
            max_prob = top_values[..., 0]
            margin = max_prob - target_prob
            wrong_confidence = max_prob * (rank > 1).float()
            token_freq = freq.gather(0, y.reshape(-1)).reshape_as(y)
            mean_ce = ce.mean()
            std_ce = ce.std().clamp_min(1e-6)
            gaussian_nll = torch.exp(-((ce - mean_ce) ** 2) / (2 * std_ce**2))

            core = {
                "ce": ce,
                "top1": top1,
                "top2": top2,
                "top3": top3,
                "top5": top5,
                "entropy": entropy,
                "rank": rank,
                "mrr": mrr,
                "margin": margin,
                "wrong_confidence": wrong_confidence,
            }
            weights = {
                "uniform": torch.ones_like(ce),
                "prob": target_prob,
                "disagreement": 1.0 - target_prob,
                "entropy": entropy,
                "inv_entropy": 1.0 - entropy,
                "frequency": token_freq,
                "inv_frequency": 1.0 - token_freq,
                "gaussian_nll": gaussian_nll,
            }
            add_weighted(totals, weight_totals, core, weights)
    finally:
        if cache_prepared:
            model.flush_block_fht_cache()

    metrics = {key: totals[key] / max(weight_totals[key], 1e-12) for key in sorted(totals)}
    ce = loss_sum / max(token_count, 1)
    metrics["ce"] = ce
    metrics["ppl"] = math.exp(min(ce, 20.0))
    return {
        "checkpoint": str(args.checkpoint),
        "source": str(source),
        "tokens": token_count,
        "model_config": checkpoint["model_config"],
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-bin", default=None)
    parser.add_argument("--tokens-jsonl", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--cache-block-fht", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tokens": result["tokens"], "ce": result["metrics"]["ce"], "ppl": result["metrics"]["ppl"]}, sort_keys=True))


if __name__ == "__main__":
    main()
