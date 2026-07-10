from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.model import GPT, GPTConfig, freeze_non_block_fht
from latent_weight_lab import BlockFHTLinear


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def get_batch(data_dir: Path, split: str, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if "cuda" in device:
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, data_dir: Path, args, ctx, cache_model: GPT | None = None, cache_dtype: torch.dtype | None = None) -> dict[str, float]:
    model.eval()
    out = {}
    if cache_model is not None:
        cache_model.prepare_block_fht_cache(dtype=cache_dtype)
    try:
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for idx in range(args.eval_iters):
                x, y = get_batch(data_dir, split, args.batch_size, args.block_size, args.device)
                with ctx:
                    _, loss = model(x, y)
                losses[idx] = loss.item()
            out[split] = float(losses.mean())
    finally:
        if cache_model is not None:
            cache_model.flush_block_fht_cache()
    model.train()
    return out


def cosine_lr(iter_num: int, args) -> float:
    if iter_num < args.warmup_iters:
        return args.learning_rate * (iter_num + 1) / (args.warmup_iters + 1)
    if iter_num > args.lr_decay_iters:
        return args.min_lr
    ratio = (iter_num - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def block_fht_latents(model: GPT) -> list[torch.Tensor]:
    return [module.generator.latent for module in model.modules() if isinstance(module, BlockFHTLinear)]


def latent_rms_hinge_loss(model: GPT, target: float) -> torch.Tensor:
    losses = []
    for latent in block_fht_latents(model):
        rms = latent.float().square().mean().sqrt()
        losses.append(torch.relu(rms - float(target)).square())
    if not losses:
        return next(model.parameters()).new_zeros(())
    return torch.stack(losses).mean()


def perturb_block_fht_latents(model: GPT, sigma: float) -> list[torch.Tensor]:
    noises = []
    with torch.no_grad():
        for latent in block_fht_latents(model):
            noise = torch.randn_like(latent) * float(sigma)
            latent.add_(noise)
            noises.append(noise)
    return noises


def restore_block_fht_latents(model: GPT, noises: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for latent, noise in zip(block_fht_latents(model), noises, strict=True):
            latent.sub_(noise)


def logits_kl_stability_loss(logits: torch.Tensor, perturbed_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = float(temperature)
    reference = F.softmax(logits.detach().float() / temp, dim=-1)
    perturbed_log_probs = F.log_softmax(perturbed_logits.float() / temp, dim=-1)
    return F.kl_div(perturbed_log_probs, reference, reduction="batchmean") * (temp * temp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", required=False)
    parser.add_argument("--out-dir", required=False)
    parser.add_argument("--init-from", choices=["scratch", "resume"], default="scratch")
    parser.add_argument("--method", choices=["baseline", "block_fht"], default="baseline")
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--block-fht-latent-ratio", type=float, default=0.01)
    parser.add_argument("--block-fht-latent-ratios", type=json.loads, default=None)
    parser.add_argument("--block-fht-layers", type=int, default=2)
    parser.add_argument("--block-fht-targets", nargs="+", default=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"])
    parser.add_argument("--block-fht-latent-init-std", type=float, default=0.02)
    parser.add_argument("--block-fht-modulation-alpha", type=float, default=0.0)
    parser.add_argument("--block-fht-modulation-centered", action="store_true")
    parser.add_argument("--block-fht-match-gpt-init", action="store_true")
    parser.add_argument("--block-fht-weight-scale", type=float, default=None)
    parser.add_argument("--block-fht-residual-base-scale", type=float, default=0.0)
    parser.add_argument("--block-fht-output-gain-targets", nargs="+", default=[])
    parser.add_argument("--block-fht-input-gain-targets", nargs="+", default=[])
    parser.add_argument("--block-fht-ffn-pregelu-gain", action="store_true")
    parser.add_argument("--block-fht-ffn-pregelu-bias", action="store_true")
    parser.add_argument("--block-fht-ffn-pregelu-bias-init", type=float, default=0.0)
    parser.add_argument("--block-fht-ffn-lowrank-rank", type=int, default=0)
    parser.add_argument("--block-fht-ffn-lowrank-scale", type=float, default=1.0)
    parser.add_argument("--block-fht-ffn-lowrank-init-std", type=float, default=0.02)
    parser.add_argument("--block-fht-cproj-lowrank-rank", type=int, default=0)
    parser.add_argument("--block-fht-cproj-lowrank-scale", type=float, default=1.0)
    parser.add_argument("--block-fht-cproj-lowrank-init-std", type=float, default=0.02)
    parser.add_argument("--block-fht-cproj-lowrank-mode", choices=["dense", "block_fht"], default="dense")
    parser.add_argument("--block-fht-cproj-lowrank-latent-ratio", type=float, default=None)
    parser.add_argument("--block-fht-cproj-lowrank-b-zero-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-fht-cproj-lowrank-bias", action="store_true")
    parser.add_argument("--block-fht-cproj-tied-cfc-skip", action="store_true")
    parser.add_argument("--block-fht-cproj-tied-cfc-scale-init", type=float, default=0.0)
    parser.add_argument("--block-fht-cproj-tied-cfc-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-fht-cproj-quarter-diag", action="store_true")
    parser.add_argument("--block-fht-cproj-quarter-diag-scale-init", type=float, default=0.0)
    parser.add_argument("--block-fht-cproj-quarter-diag-init-std", type=float, default=0.02)
    parser.add_argument("--block-fht-cproj-spectral-resid-rank", type=int, default=0)
    parser.add_argument("--block-fht-cproj-spectral-resid-scale-init", type=float, default=0.0)
    parser.add_argument("--block-fht-cproj-spectral-resid-seed", type=int, default=0)
    parser.add_argument("--block-fht-ffn-postgelu-std-target", type=float, default=0.0)
    parser.add_argument("--block-fht-ffn-postgelu-std-lambda", type=float, default=0.0)
    parser.add_argument("--block-fht-seed", type=int, default=1000)
    parser.add_argument("--block-fht-cache-weights", action="store_true")
    parser.add_argument("--freeze-non-block-fht", action="store_true")
    parser.add_argument("--train-embeddings-when-frozen", action="store_true")
    parser.add_argument("--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-fht-latent-grad-normalize", action="store_true")
    parser.add_argument("--block-fht-latent-grad-target-rms", type=float, default=0.01)
    parser.add_argument("--mapping-stability-lambda", type=float, default=0.0)
    parser.add_argument("--mapping-stability-sigma", type=float, default=1e-3)
    parser.add_argument("--mapping-stability-temperature", type=float, default=1.0)
    parser.add_argument("--mapping-norm-lambda", type=float, default=0.0)
    parser.add_argument("--mapping-norm-target-rms", type=float, default=0.03)
    parser.add_argument("--perf-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--perf-warmup-iters", type=int, default=5)
    parser.add_argument("--perf-log-interval", type=int, default=10)
    namespace = parser.parse_args()
    config = load_config(namespace.config)
    for key, value in config.items():
        setattr(namespace, key.replace("-", "_"), value)
    if namespace.data_dir is None or namespace.out_dir is None:
        raise ValueError("--data-dir and --out-dir are required, either as args or config keys")
    if namespace.perf_log_interval <= 0:
        raise ValueError("--perf-log-interval must be > 0")
    if namespace.perf_warmup_iters < 0:
        raise ValueError("--perf-warmup-iters must be >= 0")
    return namespace


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    gpt_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        block_fht=args.method == "block_fht",
        block_fht_targets=tuple(args.block_fht_targets),
        block_fht_latent_ratio=args.block_fht_latent_ratio,
        block_fht_latent_ratios=args.block_fht_latent_ratios,
        block_fht_layers=args.block_fht_layers,
        block_fht_seed=args.block_fht_seed,
        block_fht_latent_init_std=args.block_fht_latent_init_std,
        block_fht_modulation_alpha=args.block_fht_modulation_alpha,
        block_fht_modulation_centered=args.block_fht_modulation_centered,
        block_fht_match_gpt_init=args.block_fht_match_gpt_init,
        block_fht_weight_scale=args.block_fht_weight_scale,
        block_fht_residual_base_scale=args.block_fht_residual_base_scale,
        block_fht_output_gain_targets=tuple(args.block_fht_output_gain_targets),
        block_fht_input_gain_targets=tuple(args.block_fht_input_gain_targets),
        block_fht_ffn_pregelu_gain=args.block_fht_ffn_pregelu_gain,
        block_fht_ffn_pregelu_bias=args.block_fht_ffn_pregelu_bias,
        block_fht_ffn_pregelu_bias_init=args.block_fht_ffn_pregelu_bias_init,
        block_fht_ffn_lowrank_rank=args.block_fht_ffn_lowrank_rank,
        block_fht_ffn_lowrank_scale=args.block_fht_ffn_lowrank_scale,
        block_fht_ffn_lowrank_init_std=args.block_fht_ffn_lowrank_init_std,
        block_fht_cproj_lowrank_rank=args.block_fht_cproj_lowrank_rank,
        block_fht_cproj_lowrank_scale=args.block_fht_cproj_lowrank_scale,
        block_fht_cproj_lowrank_init_std=args.block_fht_cproj_lowrank_init_std,
        block_fht_cproj_lowrank_mode=args.block_fht_cproj_lowrank_mode,
        block_fht_cproj_lowrank_latent_ratio=args.block_fht_cproj_lowrank_latent_ratio,
        block_fht_cproj_lowrank_b_zero_init=args.block_fht_cproj_lowrank_b_zero_init,
        block_fht_cproj_lowrank_bias=args.block_fht_cproj_lowrank_bias,
        block_fht_cproj_tied_cfc_skip=args.block_fht_cproj_tied_cfc_skip,
        block_fht_cproj_tied_cfc_scale_init=args.block_fht_cproj_tied_cfc_scale_init,
        block_fht_cproj_tied_cfc_vector=args.block_fht_cproj_tied_cfc_vector,
        block_fht_cproj_quarter_diag=args.block_fht_cproj_quarter_diag,
        block_fht_cproj_quarter_diag_scale_init=args.block_fht_cproj_quarter_diag_scale_init,
        block_fht_cproj_quarter_diag_init_std=args.block_fht_cproj_quarter_diag_init_std,
        block_fht_cproj_spectral_resid_rank=args.block_fht_cproj_spectral_resid_rank,
        block_fht_cproj_spectral_resid_scale_init=args.block_fht_cproj_spectral_resid_scale_init,
        block_fht_cproj_spectral_resid_seed=args.block_fht_cproj_spectral_resid_seed,
        block_fht_ffn_postgelu_std_target=args.block_fht_ffn_postgelu_std_target,
        tie_word_embeddings=args.tie_word_embeddings,
    )

    iter_num = 0
    best_val_loss = 1e9
    if args.init_from == "resume":
        checkpoint = torch.load(out_dir / "ckpt.pt", map_location=args.device)
        model = GPT(GPTConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model"])
        iter_num = int(checkpoint["iter_num"])
        best_val_loss = float(checkpoint["best_val_loss"])
    else:
        model = GPT(gpt_config)
    model.to(args.device)
    if args.method == "block_fht" and args.freeze_non_block_fht:
        freeze_non_block_fht(model, train_embeddings=args.train_embeddings_when_frozen)
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        device_type,
        optimizer=args.optimizer,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
    )
    if args.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    use_weight_cache = (
        args.method == "block_fht"
        and args.block_fht_cache_weights
        and float(args.mapping_stability_lambda) == 0.0
    )
    if args.method == "block_fht" and args.block_fht_cache_weights and not use_weight_cache:
        print("block_fht: disabled weight cache because latent stability perturbations require live weights")
    tokens_per_iter = args.batch_size * args.block_size * args.gradient_accumulation_steps
    print(f"tokens per iteration: {tokens_per_iter:,}")
    print(f"model_config: {asdict(gpt_config)}")
    total_params = sum(param.numel() for param in raw_model.parameters())
    trainable_params = sum(param.numel() for param in raw_model.parameters() if param.requires_grad)
    print(f"parameters: total={total_params:,} trainable={trainable_params:,}")
    if args.method == "block_fht":
        stats = raw_model.block_fht_stats()
        print(
            "block_fht: "
            f"modules={stats['modules']} generated={stats['generated']:,} latent={stats['latent']:,}"
        )

    x, y = get_batch(data_dir, "train", args.batch_size, args.block_size, args.device)
    t0 = time.perf_counter()
    perf_peak_reset = False

    def perf_sync() -> None:
        if args.perf_profile and device_type == "cuda":
            torch.cuda.synchronize()

    def perf_now() -> float:
        perf_sync()
        return time.perf_counter()

    while True:
        eval_ms = 0.0
        lr = cosine_lr(iter_num, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if iter_num % args.eval_interval == 0:
            cache_model = raw_model if use_weight_cache else None
            eval_start = perf_now() if args.perf_profile else 0.0
            losses = estimate_loss(model, data_dir, args, ctx, cache_model=cache_model, cache_dtype=ptdtype)
            if args.perf_profile:
                eval_ms = (perf_now() - eval_start) * 1000.0
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
            if args.save_checkpoint:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(gpt_config),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, out_dir / "ckpt.pt")
                if args.checkpoint_history:
                    torch.save(checkpoint, out_dir / f"ckpt_iter{iter_num:06d}.pt")
        if iter_num >= args.max_iters:
            break

        perf_active = bool(args.perf_profile and iter_num >= args.perf_warmup_iters)
        if perf_active and device_type == "cuda" and not perf_peak_reset:
            torch.cuda.reset_peak_memory_stats()
            perf_peak_reset = True
        iter_start = perf_now() if args.perf_profile else 0.0
        prepare_cache_ms = 0.0
        forward_backward_ms = 0.0
        flush_cache_ms = 0.0
        grad_postprocess_ms = 0.0
        optimizer_ms = 0.0
        data_ms = 0.0

        ce_accum = 0.0
        stability_accum = 0.0
        norm_accum = 0.0
        postgelu_accum = 0.0
        if use_weight_cache:
            section_start = perf_now() if args.perf_profile else 0.0
            raw_model.prepare_block_fht_cache(dtype=ptdtype)
            if args.perf_profile:
                prepare_cache_ms += (perf_now() - section_start) * 1000.0
        for _ in range(args.gradient_accumulation_steps):
            section_start = perf_now() if args.perf_profile else 0.0
            with ctx:
                logits, loss = model(x, y)
                if float(args.mapping_norm_lambda) != 0.0:
                    norm_loss = latent_rms_hinge_loss(raw_model, args.mapping_norm_target_rms)
                    loss = loss + float(args.mapping_norm_lambda) * norm_loss
                    norm_accum += float(norm_loss.detach().item())
                if float(args.block_fht_ffn_postgelu_std_lambda) != 0.0:
                    postgelu_loss = raw_model.postgelu_spread_loss()
                    loss = loss + float(args.block_fht_ffn_postgelu_std_lambda) * postgelu_loss
                    postgelu_accum += float(postgelu_loss.detach().item())
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if float(args.mapping_stability_lambda) != 0.0:
                noises = perturb_block_fht_latents(raw_model, args.mapping_stability_sigma)
                try:
                    with ctx:
                        perturbed_logits, _ = model(x, None)
                        stability_loss = logits_kl_stability_loss(
                            logits,
                            perturbed_logits,
                            args.mapping_stability_temperature,
                        )
                        scaled_stability = (
                            float(args.mapping_stability_lambda)
                            * stability_loss
                            / args.gradient_accumulation_steps
                        )
                    scaler.scale(scaled_stability).backward()
                    stability_accum += float(stability_loss.detach().item())
                finally:
                    restore_block_fht_latents(raw_model, noises)
            ce_accum += float(loss.detach().item()) * args.gradient_accumulation_steps
            if args.perf_profile:
                forward_backward_ms += (perf_now() - section_start) * 1000.0
                section_start = perf_now()
            x, y = get_batch(data_dir, "train", args.batch_size, args.block_size, args.device)
            if args.perf_profile:
                data_ms += (perf_now() - section_start) * 1000.0
        if use_weight_cache:
            section_start = perf_now() if args.perf_profile else 0.0
            raw_model.flush_block_fht_cache()
            if args.perf_profile:
                flush_cache_ms += (perf_now() - section_start) * 1000.0
        section_start = perf_now() if args.perf_profile else 0.0
        if args.grad_clip != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if args.block_fht_latent_grad_normalize:
            for latent in block_fht_latents(raw_model):
                if latent.grad is None:
                    continue
                grad_rms = latent.grad.float().square().mean().sqrt().clamp_min(1e-12)
                latent.grad.mul_(float(args.block_fht_latent_grad_target_rms) / grad_rms)
        if args.perf_profile:
            grad_postprocess_ms += (perf_now() - section_start) * 1000.0
            section_start = perf_now()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if args.perf_profile:
            optimizer_ms += (perf_now() - section_start) * 1000.0
        if perf_active and (iter_num % args.perf_log_interval == 0 or eval_ms > 0.0):
            iter_ms = (perf_now() - iter_start) * 1000.0
            tokens_per_second = tokens_per_iter / max(iter_ms / 1000.0, 1e-12)
            other_ms = iter_ms - prepare_cache_ms - forward_backward_ms - flush_cache_ms - grad_postprocess_ms - optimizer_ms - data_ms
            peak_mib = 0.0
            if device_type == "cuda":
                peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                "perf "
                f"iter={iter_num} "
                f"tokens_per_s={tokens_per_second:.2f} "
                f"iter_ms={iter_ms:.2f} "
                f"prepare_ms={prepare_cache_ms:.2f} "
                f"fwbw_ms={forward_backward_ms:.2f} "
                f"train_compute_ms={forward_backward_ms:.2f} "
                f"flush_ms={flush_cache_ms:.2f} "
                f"grad_ms={grad_postprocess_ms:.2f} "
                f"opt_ms={optimizer_ms:.2f} "
                f"data_ms={data_ms:.2f} "
                f"other_ms={other_ms:.2f} "
                f"eval_ms={eval_ms:.2f} "
                f"peak_mib={peak_mib:.2f}"
            )
        t1 = time.perf_counter()
        if iter_num % args.log_interval == 0:
            msg = f"iter {iter_num}: loss {ce_accum / args.gradient_accumulation_steps:.4f}, time {(t1 - t0) * 1000:.2f}ms"
            if float(args.mapping_stability_lambda) != 0.0:
                msg += f", stability {stability_accum / args.gradient_accumulation_steps:.6f}"
            if float(args.mapping_norm_lambda) != 0.0:
                msg += f", norm {norm_accum / args.gradient_accumulation_steps:.6f}"
            if float(args.block_fht_ffn_postgelu_std_lambda) != 0.0:
                msg += f", postgelu {postgelu_accum / args.gradient_accumulation_steps:.6f}"
            print(msg)
        t0 = t1
        iter_num += 1


if __name__ == "__main__":
    main()
