from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


H100_BF16_TFLOPS = 989.0


def estimate_active_params(n_layer: int, n_embd: int, vocab_size: int = 50304) -> int:
    per_block = 12 * n_embd * n_embd + 13 * n_embd
    embeddings = vocab_size * n_embd + 1024 * n_embd
    final_ln = 2 * n_embd
    return int(embeddings + n_layer * per_block + final_ln)


def parse_perf_lines(text: str) -> list[dict[str, float]]:
    rows = []
    for line in text.splitlines():
        if not line.startswith("perf "):
            continue
        row: dict[str, float] = {}
        for key, value in re.findall(r"(\w+)=([0-9.]+)", line):
            row[key] = float(value)
        if row:
            rows.append(row)
    return rows


def parse_iter_ms(text: str) -> list[float]:
    return [float(match.group(1)) for match in re.finditer(r"iter \d+: loss [0-9.]+, time ([0-9.]+)ms", text)]


def summarize_log(path: Path, active_params: int, tokens_per_iter: int) -> dict:
    text = path.read_text(errors="replace")
    perf = parse_perf_lines(text)
    if perf:
        steady = perf[-min(5, len(perf)):]
        iter_ms = sum(row["iter_ms"] for row in steady) / len(steady)
        tokens_s = sum(row["tokens_per_s"] for row in steady) / len(steady)
        peak_mib = max(row.get("peak_mib", 0.0) for row in perf)
    else:
        iters = parse_iter_ms(text)
        steady_iters = iters[-min(5, len(iters)):]
        iter_ms = sum(steady_iters) / len(steady_iters) if steady_iters else 0.0
        tokens_s = tokens_per_iter / (iter_ms / 1000.0) if iter_ms > 0 else 0.0
        peak_mib = 0.0
    train_tflops = (6.0 * active_params * tokens_s) / 1e12
    mfu = train_tflops / H100_BF16_TFLOPS if H100_BF16_TFLOPS else 0.0
    return {
        "log": str(path),
        "active_params": active_params,
        "tokens_per_iter": tokens_per_iter,
        "iter_ms": iter_ms,
        "tokens_per_s": tokens_s,
        "estimated_train_tflops": train_tflops,
        "h100_bf16_peak_tflops": H100_BF16_TFLOPS,
        "estimated_mfu": mfu,
        "peak_mib": peak_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Training logs containing perf or iter timing lines")
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--tokens-per-iter", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    active = estimate_active_params(args.n_layer, args.n_embd, args.vocab_size)
    summaries = [summarize_log(Path(path), active, args.tokens_per_iter) for path in args.logs]
    payload = {"summaries": summaries}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
