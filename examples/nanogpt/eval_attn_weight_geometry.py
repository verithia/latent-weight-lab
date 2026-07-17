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
    total = energy.sum().clamp_min(1e-30)
    p = energy / total
    entropy = -(p * torch.log(p.clamp_min(1e-30))).sum()
    return {
        "soft_rank": float(torch.exp(entropy).item()),
        "hard_rank": float((1.0 / p.square().sum().clamp_min(1e-30)).item()),
        "stable_rank": float((total / singular.max().square().clamp_min(1e-30)).item()),
        "top1_energy": float(p[0].item()),
        "top10_energy": float(p[: min(10, p.numel())].sum().item()),
        "fro_norm": float(torch.linalg.matrix_norm(w).item()),
        "spectral_norm": float(singular.max().item()),
        "mean": float(w.mean().item()),
        "std": float(w.std().item()),
    }


def norm_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    v = values.detach().float().cpu()
    mean = v.mean().item()
    std = v.std(unbiased=False).item()
    return {
        f"{prefix}_mean": float(mean),
        f"{prefix}_std": float(std),
        f"{prefix}_cv": float(std / max(abs(mean), 1e-30)),
        f"{prefix}_min": float(v.min().item()),
        f"{prefix}_max": float(v.max().item()),
    }


def get_attn_weights(model: GPT, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = model.transformer.h[layer].attn
    if getattr(attn, "split_c_attn", False):
        q = attn.c_attn_q.weight
        k = attn.c_attn_k.weight
        v = attn.c_attn_v.weight
    elif getattr(attn, "qk_pair_c_attn", False):
        qk = attn.c_attn_qk.weight
        q, k = qk.split(attn.n_embd, dim=0)
        v = attn.c_attn_v.weight
    elif getattr(attn, "k_headwise_c_attn", False):
        q = attn.c_attn_q.weight
        k = torch.cat([head.weight for head in attn.c_attn_k_headwise.heads], dim=0)
        v = attn.c_attn_v.weight
    elif getattr(attn, "qk_headwise_c_attn", False):
        qk_heads = [head.weight for head in attn.c_attn_qk_headwise.heads]
        q = torch.cat([head.split(attn.n_embd // attn.n_head, dim=0)[0] for head in qk_heads], dim=0)
        k = torch.cat([head.split(attn.n_embd // attn.n_head, dim=0)[1] for head in qk_heads], dim=0)
        v = attn.c_attn_v.weight
    elif getattr(attn, "qk_tied_c_attn", False) or getattr(attn, "qk_tied_sign_c_attn", False):
        q = attn.c_attn_qk_tied.weight
        if getattr(attn, "qk_tied_sign", None) is None:
            k = q
        else:
            k = q * attn.qk_tied_sign.to(device=q.device, dtype=q.dtype).view(-1, 1)
        v = attn.c_attn_v.weight
    elif getattr(attn, "qk_tied_headwise_c_attn", False) or getattr(attn, "qk_tied_sign_headwise_c_attn", False):
        q = torch.cat([head.weight for head in attn.c_attn_qk_tied_headwise.heads], dim=0)
        if getattr(attn, "qk_tied_sign", None) is None:
            k = q
        else:
            k = q * attn.qk_tied_sign.to(device=q.device, dtype=q.dtype).view(-1, 1)
        v = attn.c_attn_v.weight
    elif (
        getattr(attn, "qk_mix25_headwise_c_attn", False)
        or getattr(attn, "qk_mix50_headwise_c_attn", False)
        or getattr(attn, "qk_mix75_headwise_c_attn", False)
    ):
        qk_heads = [head.weight for head in attn.c_attn_qk_mix_headwise.heads]
        q_raw = torch.cat([head.split(attn.n_embd // attn.n_head, dim=0)[0] for head in qk_heads], dim=0)
        k_raw = torch.cat([head.split(attn.n_embd // attn.n_head, dim=0)[1] for head in qk_heads], dim=0)
        alpha = float(attn.qk_mix_alpha)
        scale = 1.0 / float((1.0 + alpha * alpha) ** 0.5)
        q = (q_raw + alpha * k_raw) * scale
        k = (k_raw + alpha * q_raw) * scale
        v = attn.c_attn_v.weight
    elif getattr(attn, "qk_sameseed_c_attn", False):
        q = attn.c_attn_q_sameseed.weight
        k = attn.c_attn_k_sameseed.weight
        v = attn.c_attn_v.weight
    elif getattr(attn, "qk_sameseed_headwise_c_attn", False):
        q = torch.cat([head.weight for head in attn.c_attn_q_sameseed_headwise.heads], dim=0)
        k = torch.cat([head.weight for head in attn.c_attn_k_sameseed_headwise.heads], dim=0)
        v = attn.c_attn_v.weight
    else:
        qkv = attn.c_attn.weight
        q, k, v = qkv.split(attn.n_embd, dim=0)
    return q.detach(), k.detach(), v.detach(), attn.c_proj.weight.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    model = load_model(Path(args.checkpoint), args.device)
    layer_ids = parse_layers(args.layers)
    if layer_ids is None:
        layer_ids = list(range(len(model.transformer.h)))
    n_head = model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head
    d_in = model.config.n_embd

    rows: list[dict[str, float | int | str]] = []
    for layer in layer_ids:
        q, k, v, proj = get_attn_weights(model, layer)
        for name, weight in [("q", q), ("k", k), ("v", v), ("c_proj", proj)]:
            row: dict[str, float | int | str] = {"label": args.label, "layer": layer, "kind": "matrix", "target": name}
            row.update(spectrum_metrics(weight))
            row.update(norm_stats(weight.norm(dim=1), "row_norm"))
            rows.append(row)

        q_heads = q.reshape(n_head, head_dim, d_in)
        k_heads = k.reshape(n_head, head_dim, d_in)
        v_heads = v.reshape(n_head, head_dim, d_in)
        for head in range(n_head):
            qh = q_heads[head].float().cpu()
            kh = k_heads[head].float().cpu()
            vh = v_heads[head].float().cpu()
            qk = qh @ kh.T
            q_norm = torch.linalg.matrix_norm(qh)
            k_norm = torch.linalg.matrix_norm(kh)
            v_norm = torch.linalg.matrix_norm(vh)
            denom = (q_norm * k_norm).clamp_min(1e-30)
            row = {
                "label": args.label,
                "layer": layer,
                "kind": "head",
                "target": "qk",
                "head": head,
                "q_fro": float(q_norm.item()),
                "k_fro": float(k_norm.item()),
                "v_fro": float(v_norm.item()),
                "qk_fro": float(torch.linalg.matrix_norm(qk).item()),
                "qk_spectral": float(torch.linalg.svdvals(qk).max().item()),
                "qk_score_var_iso": float((qk.square().sum() / d_in).item()),
                "qk_cos_flat": float((qh * kh).sum().div(denom).item()),
                "q_over_k_fro": float((q_norm / k_norm.clamp_min(1e-30)).item()),
                "v_over_k_fro": float((v_norm / k_norm.clamp_min(1e-30)).item()),
            }
            row.update(norm_stats(qh.norm(dim=1), "q_row_norm"))
            row.update(norm_stats(kh.norm(dim=1), "k_row_norm"))
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    numeric_keys = sorted(
        key
        for key in fieldnames
        if key not in {"label", "kind", "target"} and all(isinstance(row.get(key, 0.0), (int, float)) for row in rows)
    )
    groups = sorted({(row["kind"], row["target"]) for row in rows})
    for kind, target in groups:
        group_rows = [row for row in rows if row["kind"] == kind and row["target"] == target]
        out: dict[str, float | str | int] = {"label": args.label, "kind": kind, "target": target, "rows": len(group_rows)}
        for key in numeric_keys:
            vals = [float(row[key]) for row in group_rows if key in row and row[key] != ""]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_std"] = float(np.std(vals))
        summary_rows.append(out)

    summary = Path(args.summary)
    summary_fieldnames = sorted({key for row in summary_rows for key in row})
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"rows": len(rows), "output": str(output), "summary": str(summary)}, indent=2))


if __name__ == "__main__":
    main()
