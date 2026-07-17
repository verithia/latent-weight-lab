from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from examples.nanogpt.muon import Muon
from latent_weight_lab import BlockFHTLinear, flush_block_fht_weight_cache, prepare_block_fht_weight_cache


class MultiOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        for optimizer, state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    block_fht: bool = False
    block_fht_targets: tuple[str, ...] = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    block_fht_latent_ratio: float = 0.01
    block_fht_latent_ratios: dict[str, float] | None = None
    block_fht_layers: int = 2
    block_fht_seed: int = 1000
    block_fht_latent_init_std: float = 0.02
    block_fht_modulation_alpha: float = 0.0
    block_fht_modulation_centered: bool = False
    block_fht_match_gpt_init: bool = False
    block_fht_weight_scale: float | None = None
    block_fht_residual_base_scale: float = 0.0
    block_fht_output_gain_targets: tuple[str, ...] = ()
    block_fht_input_gain_targets: tuple[str, ...] = ()
    block_fht_ffn_pregelu_gain: bool = False
    block_fht_ffn_pregelu_bias: bool = False
    block_fht_ffn_pregelu_bias_init: float = 0.0
    block_fht_ffn_lowrank_rank: int = 0
    block_fht_ffn_lowrank_scale: float = 1.0
    block_fht_ffn_lowrank_init_std: float = 0.02
    block_fht_ffn_spectral_rank: int = 0
    block_fht_ffn_spectral_out_groups: int = 1
    block_fht_ffn_spectral_in_groups: int = 1
    block_fht_cproj_lowrank_rank: int = 0
    block_fht_cproj_lowrank_scale: float = 1.0
    block_fht_cproj_lowrank_init_std: float = 0.02
    block_fht_cproj_lowrank_mode: str = "dense"
    block_fht_cproj_lowrank_latent_ratio: float | None = None
    block_fht_cproj_lowrank_b_zero_init: bool = True
    block_fht_cproj_lowrank_bias: bool = False
    block_fht_cproj_tied_cfc_skip: bool = False
    block_fht_cproj_tied_cfc_scale_init: float = 0.0
    block_fht_cproj_tied_cfc_vector: bool = True
    block_fht_cproj_quarter_diag: bool = False
    block_fht_cproj_quarter_diag_scale_init: float = 0.0
    block_fht_cproj_quarter_diag_init_std: float = 0.02
    block_fht_cproj_spectral_resid_rank: int = 0
    block_fht_cproj_spectral_resid_scale_init: float = 0.0
    block_fht_cproj_spectral_resid_seed: int = 0
    block_fht_ffn_postgelu_std_target: float = 0.0
    tie_word_embeddings: bool = True


QKV_SPLIT_TARGETS = ("attn.c_attn.q", "attn.c_attn.k", "attn.c_attn.v")
MLP_C_FC_GROUP_TARGETS = {
    "mlp.c_fc_group12": 12,
    "mlp.c_fc_group16": 16,
    "mlp.c_fc_group24": 24,
}
MLP_C_PROJ_GROUP_TARGETS = {
    "mlp.c_proj_group12": 12,
}
MLP_C_PROJ_OUT_GROUP_TARGETS = {
    "mlp.c_proj_outgroup12": 12,
    "mlp.c_proj_outgroup16": 16,
}
MLP_C_PROJ_OUT_MIX_TARGET = "mlp.c_proj_outmix"
MLP_C_PROJ_OUT_GROUP_MIX_TARGETS = {
    "mlp.c_proj_outgroup12_mix": 12,
}
MLP_C_PROJ_IN_GROUP_MIX_TARGETS = {
    "mlp.c_proj_group12_inmix": 12,
}


def is_residual_projection_target(target_name: str) -> bool:
    return target_name.endswith("c_proj") or target_name.startswith("mlp.c_proj_")


class HeadwiseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        head_dim: int,
        n_head: int,
        bias: bool,
        config: GPTConfig,
        target_name: str,
        seed_offset: int,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                make_linear(
                    in_features,
                    head_dim,
                    bias,
                    config,
                    target_name,
                    seed_offset + head_idx,
                )
                for head_idx in range(n_head)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cached_weights = [getattr(head, "_cached_weight", None) for head in self.heads]
        if cached_weights and all(weight is not None for weight in cached_weights):
            biases = [getattr(head, "bias", None) for head in self.heads]
            if all(bias is None for bias in biases):
                combined_bias = None
            elif all(bias is not None for bias in biases):
                combined_bias = torch.cat(biases, dim=0)
            else:
                combined_bias = None
                return torch.cat([head(x) for head in self.heads], dim=-1)
            combined_weight = torch.cat(cached_weights, dim=0)
            return F.linear(x, combined_weight, combined_bias)
        return torch.cat([head(x) for head in self.heads], dim=-1)


def next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def normalized_fht_last_dim(values: torch.Tensor) -> torch.Tensor:
    size = values.shape[-1]
    if size & (size - 1):
        raise ValueError("FHT size must be a power of two")
    out = values
    step = 1
    while step < size:
        shape = out.shape
        out = out.reshape(*shape[:-1], -1, 2, step)
        first = out[..., 0, :].clone()
        second = out[..., 1, :].clone()
        out[..., 0, :] = first + second
        out[..., 1, :] = first - second
        out = out.reshape(shape)
        step *= 2
    return out / math.sqrt(size)


class FixedFHTMix(nn.Module):
    def __init__(self, features: int, seed: int) -> None:
        super().__init__()
        self.features = int(features)
        self.padded = next_power_of_two(self.features)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        signs = torch.randint(0, 2, (self.padded,), generator=generator, dtype=torch.float32) * 2.0 - 1.0
        self.register_buffer("signs", signs, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.features:
            raise ValueError(f"expected last dim {self.features}, got {x.shape[-1]}")
        if self.padded != self.features:
            x = F.pad(x, (0, self.padded - self.features))
        x = x * self.signs.to(device=x.device, dtype=x.dtype)
        x = normalized_fht_last_dim(x)
        x = x * self.signs.to(device=x.device, dtype=x.dtype)
        return x[..., : self.features]


class FixedFHTOutputMixLinear(nn.Module):
    def __init__(self, linear: nn.Module, out_features: int, seed: int) -> None:
        super().__init__()
        self.linear = linear
        self.mix = FixedFHTMix(out_features, seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.linear(x))


class FixedFHTInputMixLinear(nn.Module):
    def __init__(self, linear: nn.Module, in_features: int, seed: int) -> None:
        super().__init__()
        self.mix = FixedFHTMix(in_features, seed)
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.mix(x))


class GroupedInputLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int,
        bias: bool,
        config: GPTConfig,
        target_name: str,
        seed_offset: int,
    ) -> None:
        super().__init__()
        if in_features % groups != 0:
            raise ValueError(f"in_features={in_features} is not divisible by groups={groups}")
        group_features = in_features // groups
        self.group_features = group_features
        self.groups = groups
        self.heads = nn.ModuleList(
            [
                make_linear(
                    group_features,
                    out_features,
                    bias and group_idx == 0,
                    config,
                    target_name,
                    seed_offset + group_idx,
                )
                for group_idx in range(groups)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = x.split(self.group_features, dim=-1)
        out = self.heads[0](pieces[0])
        for piece, head in zip(pieces[1:], self.heads[1:], strict=True):
            out = out + head(piece)
        return out


def make_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    config: GPTConfig,
    target_name: str,
    seed_offset: int,
) -> nn.Module:
    if config.block_fht and target_name in config.block_fht_targets:
        latent_ratio = float(config.block_fht_latent_ratio)
        if config.block_fht_latent_ratios is not None and target_name in config.block_fht_latent_ratios:
            latent_ratio = float(config.block_fht_latent_ratios[target_name])
        target_std = 0.02
        if is_residual_projection_target(target_name):
            target_std = 0.02 / math.sqrt(2 * config.n_layer)
        if config.block_fht_weight_scale is not None:
            weight_scale = float(config.block_fht_weight_scale)
        elif config.block_fht_match_gpt_init:
            weight_scale = target_std / float(config.block_fht_latent_init_std)
        else:
            weight_scale = 1.0
        return BlockFHTLinear(
            in_features,
            out_features,
            bias=bias,
            latent_ratio=latent_ratio,
            layers=config.block_fht_layers,
            seed=config.block_fht_seed + seed_offset,
            latent_init_std=config.block_fht_latent_init_std,
            weight_scale=weight_scale,
            modulation_alpha=config.block_fht_modulation_alpha,
            modulation_centered=config.block_fht_modulation_centered,
            residual_base_scale=config.block_fht_residual_base_scale,
            residual_base_std=target_std,
            output_gain=target_name in config.block_fht_output_gain_targets,
            input_gain=target_name in config.block_fht_input_gain_targets,
            spectral_rank=config.block_fht_ffn_spectral_rank if target_name == "mlp.c_fc" else 0,
            spectral_out_groups=config.block_fht_ffn_spectral_out_groups if target_name == "mlp.c_fc" else 1,
            spectral_in_groups=config.block_fht_ffn_spectral_in_groups if target_name == "mlp.c_fc" else 1,
        )
    return nn.Linear(in_features, out_features, bias=bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.qk_pair_c_attn = "attn.c_attn.qk" in config.block_fht_targets
        self.k_headwise_c_attn = "attn.c_attn.k_headwise" in config.block_fht_targets
        self.qk_headwise_c_attn = "attn.c_attn.qk_headwise" in config.block_fht_targets
        self.qk_tied_c_attn = "attn.c_attn.qk_tied" in config.block_fht_targets
        self.qk_tied_sign_c_attn = "attn.c_attn.qk_tied_sign" in config.block_fht_targets
        self.qk_tied_headwise_c_attn = "attn.c_attn.qk_tied_headwise" in config.block_fht_targets
        self.qk_tied_sign_headwise_c_attn = "attn.c_attn.qk_tied_sign_headwise" in config.block_fht_targets
        self.qk_mix25_headwise_c_attn = "attn.c_attn.qk_mix25_headwise" in config.block_fht_targets
        self.qk_mix50_headwise_c_attn = "attn.c_attn.qk_mix50_headwise" in config.block_fht_targets
        self.qk_mix75_headwise_c_attn = "attn.c_attn.qk_mix75_headwise" in config.block_fht_targets
        self.qk_sameseed_c_attn = "attn.c_attn.qk_sameseed" in config.block_fht_targets
        self.qk_sameseed_headwise_c_attn = "attn.c_attn.qk_sameseed_headwise" in config.block_fht_targets
        split_target_present = any(target in config.block_fht_targets for target in QKV_SPLIT_TARGETS)
        self.split_c_attn = split_target_present and not (
            self.qk_pair_c_attn
            or self.k_headwise_c_attn
            or self.qk_headwise_c_attn
            or self.qk_tied_c_attn
            or self.qk_tied_sign_c_attn
            or self.qk_tied_headwise_c_attn
            or self.qk_tied_sign_headwise_c_attn
            or self.qk_mix25_headwise_c_attn
            or self.qk_mix50_headwise_c_attn
            or self.qk_mix75_headwise_c_attn
            or self.qk_sameseed_c_attn
            or self.qk_sameseed_headwise_c_attn
        )
        structured = sum(
            [
                self.split_c_attn,
                self.qk_pair_c_attn,
                self.k_headwise_c_attn,
                self.qk_headwise_c_attn,
                self.qk_tied_c_attn,
                self.qk_tied_sign_c_attn,
                self.qk_tied_headwise_c_attn,
                self.qk_tied_sign_headwise_c_attn,
                self.qk_mix25_headwise_c_attn,
                self.qk_mix50_headwise_c_attn,
                self.qk_mix75_headwise_c_attn,
                self.qk_sameseed_c_attn,
                self.qk_sameseed_headwise_c_attn,
            ]
        )
        if structured > 1 or (structured and "attn.c_attn" in config.block_fht_targets):
            raise ValueError("Use exactly one attn.c_attn transform family per run")
        if structured and "attn.c_attn.q" in config.block_fht_targets and not self.split_c_attn:
            raise ValueError("attn.c_attn.q can only be used with split-QKV")
        if structured and "attn.c_attn.k" in config.block_fht_targets and not self.split_c_attn:
            raise ValueError("attn.c_attn.k can only be used with split-QKV")
        if self.split_c_attn:
            if "attn.c_attn" in config.block_fht_targets:
                raise ValueError("Use either monolithic attn.c_attn or split attn.c_attn.{q,k,v}, not both")
            self.c_attn = None
            self.c_attn_q = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.q", layer_id * 8)
            self.c_attn_k = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.k", layer_id * 8 + 1)
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            self.c_attn_qk = None
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_mix_alpha = 0.0
            self.qk_tied_sign = None
        elif self.qk_pair_c_attn:
            self.c_attn = None
            self.c_attn_q = None
            self.c_attn_k = None
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            if "attn.c_attn.v" not in config.block_fht_targets:
                self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.c_attn_qk = make_linear(config.n_embd, 2 * config.n_embd, config.bias, config, "attn.c_attn.qk", layer_id * 8)
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_mix_alpha = 0.0
            self.qk_tied_sign = None
        elif self.k_headwise_c_attn:
            head_dim = config.n_embd // config.n_head
            self.c_attn = None
            self.c_attn_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.c_attn_k = None
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            if "attn.c_attn.v" not in config.block_fht_targets:
                self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.c_attn_qk = None
            self.c_attn_k_headwise = HeadwiseLinear(config.n_embd, head_dim, config.n_head, config.bias, config, "attn.c_attn.k_headwise", layer_id * 32)
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_mix_alpha = 0.0
            self.qk_tied_sign = None
        elif self.qk_headwise_c_attn:
            head_dim = config.n_embd // config.n_head
            self.c_attn = None
            self.c_attn_q = None
            self.c_attn_k = None
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            if "attn.c_attn.v" not in config.block_fht_targets:
                self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.c_attn_qk = None
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = HeadwiseLinear(config.n_embd, 2 * head_dim, config.n_head, config.bias, config, "attn.c_attn.qk_headwise", layer_id * 32)
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_mix_alpha = 0.0
            self.qk_tied_sign = None
        elif self.qk_tied_c_attn or self.qk_tied_sign_c_attn:
            self.c_attn = None
            self.c_attn_q = None
            self.c_attn_k = None
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            if "attn.c_attn.v" not in config.block_fht_targets:
                self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            target = "attn.c_attn.qk_tied_sign" if self.qk_tied_sign_c_attn else "attn.c_attn.qk_tied"
            self.c_attn_qk = None
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = make_linear(config.n_embd, config.n_embd, config.bias, config, target, layer_id * 8)
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_mix_alpha = 0.0
            if self.qk_tied_sign_c_attn:
                gen = torch.Generator()
                gen.manual_seed(config.block_fht_seed + layer_id * 8191 + 17)
                sign = torch.randint(0, 2, (config.n_embd,), generator=gen, dtype=torch.float32).mul_(2).sub_(1)
                self.register_buffer("qk_tied_sign", sign, persistent=True)
            else:
                self.qk_tied_sign = None
        elif self.qk_tied_headwise_c_attn or self.qk_tied_sign_headwise_c_attn:
            head_dim = config.n_embd // config.n_head
            self.c_attn = None
            self.c_attn_q = None
            self.c_attn_k = None
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            if "attn.c_attn.v" not in config.block_fht_targets:
                self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            target = "attn.c_attn.qk_tied_sign_headwise" if self.qk_tied_sign_headwise_c_attn else "attn.c_attn.qk_tied_headwise"
            self.c_attn_qk = None
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = HeadwiseLinear(config.n_embd, head_dim, config.n_head, config.bias, config, target, layer_id * 32)
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_mix_alpha = 0.0
            if self.qk_tied_sign_headwise_c_attn:
                gen = torch.Generator()
                gen.manual_seed(config.block_fht_seed + layer_id * 8191 + 29)
                sign = torch.randint(0, 2, (config.n_embd,), generator=gen, dtype=torch.float32).mul_(2).sub_(1)
                self.register_buffer("qk_tied_sign", sign, persistent=True)
            else:
                self.qk_tied_sign = None
        elif self.qk_mix25_headwise_c_attn or self.qk_mix50_headwise_c_attn or self.qk_mix75_headwise_c_attn:
            head_dim = config.n_embd // config.n_head
            self.c_attn = None
            self.c_attn_q = None
            self.c_attn_k = None
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            if "attn.c_attn.v" not in config.block_fht_targets:
                self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            if self.qk_mix25_headwise_c_attn:
                target = "attn.c_attn.qk_mix25_headwise"
                self.qk_mix_alpha = 0.25
            elif self.qk_mix50_headwise_c_attn:
                target = "attn.c_attn.qk_mix50_headwise"
                self.qk_mix_alpha = 0.50
            else:
                target = "attn.c_attn.qk_mix75_headwise"
                self.qk_mix_alpha = 0.75
            self.c_attn_qk = None
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = HeadwiseLinear(config.n_embd, 2 * head_dim, config.n_head, config.bias, config, target, layer_id * 32)
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_tied_sign = None
        elif self.qk_sameseed_c_attn:
            self.c_attn = None
            self.c_attn_q = None
            self.c_attn_k = None
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            if "attn.c_attn.v" not in config.block_fht_targets:
                self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.c_attn_qk = None
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.qk_sameseed", layer_id * 8)
            self.c_attn_k_sameseed = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.qk_sameseed", layer_id * 8)
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_mix_alpha = 0.0
            self.qk_tied_sign = None
        elif self.qk_sameseed_headwise_c_attn:
            head_dim = config.n_embd // config.n_head
            self.c_attn = None
            self.c_attn_q = None
            self.c_attn_k = None
            self.c_attn_v = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.v", layer_id * 8 + 2)
            if "attn.c_attn.v" not in config.block_fht_targets:
                self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.c_attn_qk = None
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = HeadwiseLinear(config.n_embd, head_dim, config.n_head, config.bias, config, "attn.c_attn.qk_sameseed_headwise", layer_id * 32)
            self.c_attn_k_sameseed_headwise = HeadwiseLinear(config.n_embd, head_dim, config.n_head, config.bias, config, "attn.c_attn.qk_sameseed_headwise", layer_id * 32)
            self.qk_mix_alpha = 0.0
            self.qk_tied_sign = None
        else:
            self.c_attn = make_linear(config.n_embd, 3 * config.n_embd, config.bias, config, "attn.c_attn", layer_id * 4)
            self.c_attn_q = None
            self.c_attn_k = None
            self.c_attn_v = None
            self.c_attn_qk = None
            self.c_attn_k_headwise = None
            self.c_attn_qk_headwise = None
            self.c_attn_qk_tied = None
            self.c_attn_qk_tied_headwise = None
            self.c_attn_qk_mix_headwise = None
            self.c_attn_q_sameseed = None
            self.c_attn_k_sameseed = None
            self.c_attn_q_sameseed_headwise = None
            self.c_attn_k_sameseed_headwise = None
            self.qk_mix_alpha = 0.0
            self.qk_tied_sign = None
        self.c_proj = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_proj", layer_id * 4 + 1)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.size()
        if self.split_c_attn:
            assert self.c_attn_q is not None and self.c_attn_k is not None and self.c_attn_v is not None
            q = self.c_attn_q(x)
            k = self.c_attn_k(x)
            v = self.c_attn_v(x)
        elif self.qk_pair_c_attn:
            assert self.c_attn_qk is not None and self.c_attn_v is not None
            q, k = self.c_attn_qk(x).split(self.n_embd, dim=2)
            v = self.c_attn_v(x)
        elif self.k_headwise_c_attn:
            assert self.c_attn_q is not None and self.c_attn_k_headwise is not None and self.c_attn_v is not None
            q = self.c_attn_q(x)
            k = self.c_attn_k_headwise(x)
            v = self.c_attn_v(x)
        elif self.qk_headwise_c_attn:
            assert self.c_attn_qk_headwise is not None and self.c_attn_v is not None
            q, k = self.c_attn_qk_headwise(x).split(self.n_embd, dim=2)
            v = self.c_attn_v(x)
        elif self.qk_tied_c_attn or self.qk_tied_sign_c_attn:
            assert self.c_attn_qk_tied is not None and self.c_attn_v is not None
            q = self.c_attn_qk_tied(x)
            if self.qk_tied_sign is None:
                k = q
            else:
                k = q * self.qk_tied_sign.to(device=q.device, dtype=q.dtype)
            v = self.c_attn_v(x)
        elif self.qk_tied_headwise_c_attn or self.qk_tied_sign_headwise_c_attn:
            assert self.c_attn_qk_tied_headwise is not None and self.c_attn_v is not None
            q = self.c_attn_qk_tied_headwise(x)
            if self.qk_tied_sign is None:
                k = q
            else:
                k = q * self.qk_tied_sign.to(device=q.device, dtype=q.dtype)
            v = self.c_attn_v(x)
        elif self.qk_mix25_headwise_c_attn or self.qk_mix50_headwise_c_attn or self.qk_mix75_headwise_c_attn:
            assert self.c_attn_qk_mix_headwise is not None and self.c_attn_v is not None
            q_raw, k_raw = self.c_attn_qk_mix_headwise(x).split(self.n_embd, dim=2)
            alpha = float(self.qk_mix_alpha)
            scale = 1.0 / math.sqrt(1.0 + alpha * alpha)
            q = (q_raw + alpha * k_raw) * scale
            k = (k_raw + alpha * q_raw) * scale
            v = self.c_attn_v(x)
        elif self.qk_sameseed_c_attn:
            assert self.c_attn_q_sameseed is not None and self.c_attn_k_sameseed is not None and self.c_attn_v is not None
            q = self.c_attn_q_sameseed(x)
            k = self.c_attn_k_sameseed(x)
            v = self.c_attn_v(x)
        elif self.qk_sameseed_headwise_c_attn:
            assert self.c_attn_q_sameseed_headwise is not None and self.c_attn_k_sameseed_headwise is not None and self.c_attn_v is not None
            q = self.c_attn_q_sameseed_headwise(x)
            k = self.c_attn_k_sameseed_headwise(x)
            v = self.c_attn_v(x)
        else:
            assert self.c_attn is not None
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        grouped_targets = [target for target in MLP_C_FC_GROUP_TARGETS if target in config.block_fht_targets]
        if len(grouped_targets) > 1:
            raise ValueError("Use exactly one grouped mlp.c_fc target per run")
        if grouped_targets and "mlp.c_fc" in config.block_fht_targets:
            raise ValueError("Use either plain mlp.c_fc or grouped mlp.c_fc, not both")
        if grouped_targets:
            target = grouped_targets[0]
            groups = MLP_C_FC_GROUP_TARGETS[target]
            out_features = 4 * config.n_embd
            if out_features % groups != 0:
                raise ValueError(f"mlp.c_fc output size {out_features} is not divisible by groups={groups}")
            self.c_fc = HeadwiseLinear(
                config.n_embd,
                out_features // groups,
                groups,
                config.bias,
                config,
                target,
                layer_id * 64 + 2,
            )
        else:
            self.c_fc = make_linear(config.n_embd, 4 * config.n_embd, config.bias, config, "mlp.c_fc", layer_id * 4 + 2)
        self.gelu = nn.GELU()
        grouped_proj_targets = [target for target in MLP_C_PROJ_GROUP_TARGETS if target in config.block_fht_targets]
        out_grouped_proj_targets = [target for target in MLP_C_PROJ_OUT_GROUP_TARGETS if target in config.block_fht_targets]
        out_mix_proj = MLP_C_PROJ_OUT_MIX_TARGET in config.block_fht_targets
        out_grouped_mix_proj_targets = [target for target in MLP_C_PROJ_OUT_GROUP_MIX_TARGETS if target in config.block_fht_targets]
        in_grouped_mix_proj_targets = [target for target in MLP_C_PROJ_IN_GROUP_MIX_TARGETS if target in config.block_fht_targets]
        structured_proj_count = (
            len(grouped_proj_targets)
            + len(out_grouped_proj_targets)
            + int(out_mix_proj)
            + len(out_grouped_mix_proj_targets)
            + len(in_grouped_mix_proj_targets)
        )
        if structured_proj_count > 1:
            raise ValueError("Use exactly one grouped mlp.c_proj target per run")
        if structured_proj_count and "mlp.c_proj" in config.block_fht_targets:
            raise ValueError("Use either plain mlp.c_proj or grouped mlp.c_proj, not both")
        if grouped_proj_targets:
            target = grouped_proj_targets[0]
            self.c_proj = GroupedInputLinear(
                4 * config.n_embd,
                config.n_embd,
                MLP_C_PROJ_GROUP_TARGETS[target],
                config.bias,
                config,
                target,
                layer_id * 64 + 33,
            )
        elif out_grouped_proj_targets:
            target = out_grouped_proj_targets[0]
            groups = MLP_C_PROJ_OUT_GROUP_TARGETS[target]
            if config.n_embd % groups != 0:
                raise ValueError(f"mlp.c_proj output size {config.n_embd} is not divisible by groups={groups}")
            self.c_proj = HeadwiseLinear(
                4 * config.n_embd,
                config.n_embd // groups,
                groups,
                config.bias,
                config,
                target,
                layer_id * 64 + 49,
            )
        elif out_mix_proj:
            base = make_linear(4 * config.n_embd, config.n_embd, config.bias, config, MLP_C_PROJ_OUT_MIX_TARGET, layer_id * 64 + 65)
            self.c_proj = FixedFHTOutputMixLinear(base, config.n_embd, config.block_fht_seed + layer_id * 64 + 66)
        elif out_grouped_mix_proj_targets:
            target = out_grouped_mix_proj_targets[0]
            groups = MLP_C_PROJ_OUT_GROUP_MIX_TARGETS[target]
            if config.n_embd % groups != 0:
                raise ValueError(f"mlp.c_proj output size {config.n_embd} is not divisible by groups={groups}")
            base = HeadwiseLinear(
                4 * config.n_embd,
                config.n_embd // groups,
                groups,
                config.bias,
                config,
                target,
                layer_id * 64 + 81,
            )
            self.c_proj = FixedFHTOutputMixLinear(base, config.n_embd, config.block_fht_seed + layer_id * 64 + 82)
        elif in_grouped_mix_proj_targets:
            target = in_grouped_mix_proj_targets[0]
            base = GroupedInputLinear(
                4 * config.n_embd,
                config.n_embd,
                MLP_C_PROJ_IN_GROUP_MIX_TARGETS[target],
                config.bias,
                config,
                target,
                layer_id * 64 + 97,
            )
            self.c_proj = FixedFHTInputMixLinear(base, 4 * config.n_embd, config.block_fht_seed + layer_id * 64 + 98)
        else:
            self.c_proj = make_linear(4 * config.n_embd, config.n_embd, config.bias, config, "mlp.c_proj", layer_id * 4 + 3)
        self.dropout = nn.Dropout(config.dropout)
        self.pregelu_gain = nn.Parameter(torch.ones(4 * config.n_embd)) if config.block_fht_ffn_pregelu_gain else None
        if config.block_fht_ffn_pregelu_bias:
            self.pregelu_bias = nn.Parameter(torch.full((4 * config.n_embd,), float(config.block_fht_ffn_pregelu_bias_init)))
        else:
            self.pregelu_bias = None
        rank = int(config.block_fht_ffn_lowrank_rank)
        if rank > 0:
            self.lowrank_left = nn.Parameter(torch.empty(config.n_embd, rank))
            self.lowrank_right = nn.Parameter(torch.empty(rank, 4 * config.n_embd))
            nn.init.normal_(self.lowrank_left, mean=0.0, std=float(config.block_fht_ffn_lowrank_init_std))
            nn.init.zeros_(self.lowrank_right)
        else:
            self.lowrank_left = None
            self.lowrank_right = None
        self.lowrank_scale = float(config.block_fht_ffn_lowrank_scale)
        cproj_rank = int(config.block_fht_cproj_lowrank_rank)
        if cproj_rank > 0:
            mode = config.block_fht_cproj_lowrank_mode
            if mode == "dense":
                self.cproj_lowrank_left = nn.Parameter(torch.empty(4 * config.n_embd, cproj_rank))
                self.cproj_lowrank_right = nn.Parameter(torch.empty(cproj_rank, config.n_embd))
                nn.init.normal_(self.cproj_lowrank_left, mean=0.0, std=float(config.block_fht_cproj_lowrank_init_std))
                nn.init.zeros_(self.cproj_lowrank_right)
            elif mode == "block_fht":
                latent_ratio = (
                    float(config.block_fht_cproj_lowrank_latent_ratio)
                    if config.block_fht_cproj_lowrank_latent_ratio is not None
                    else float(config.block_fht_latent_ratio)
                )
                self.cproj_lowrank_left = BlockFHTLinear(
                    4 * config.n_embd,
                    cproj_rank,
                    bias=bool(config.block_fht_cproj_lowrank_bias),
                    latent_ratio=latent_ratio,
                    layers=config.block_fht_layers,
                    seed=config.block_fht_seed + layer_id * 64 + 113,
                    latent_init_std=float(config.block_fht_cproj_lowrank_init_std),
                    weight_scale=1.0,
                )
                right_init_std = 0.0 if config.block_fht_cproj_lowrank_b_zero_init else float(config.block_fht_cproj_lowrank_init_std)
                self.cproj_lowrank_right = BlockFHTLinear(
                    cproj_rank,
                    config.n_embd,
                    bias=False,
                    latent_ratio=latent_ratio,
                    layers=config.block_fht_layers,
                    seed=config.block_fht_seed + layer_id * 64 + 114,
                    latent_init_std=right_init_std,
                    weight_scale=1.0,
                )
            else:
                raise ValueError(f"unknown block_fht_cproj_lowrank_mode={mode!r}")
        else:
            self.cproj_lowrank_left = None
            self.cproj_lowrank_right = None
        self.cproj_lowrank_scale = float(config.block_fht_cproj_lowrank_scale)
        if config.block_fht_cproj_tied_cfc_skip:
            if not hasattr(self.c_fc, "weight"):
                raise ValueError("block_fht_cproj_tied_cfc_skip requires mlp.c_fc with a weight attribute")
            scale_shape = (config.n_embd,) if config.block_fht_cproj_tied_cfc_vector else ()
            self.cproj_tied_cfc_scale = nn.Parameter(torch.full(scale_shape, float(config.block_fht_cproj_tied_cfc_scale_init)))
        else:
            self.cproj_tied_cfc_scale = None
        if config.block_fht_cproj_quarter_diag:
            self.cproj_quarter_diag_weight = nn.Parameter(torch.empty(4, config.n_embd))
            nn.init.normal_(self.cproj_quarter_diag_weight, mean=0.0, std=float(config.block_fht_cproj_quarter_diag_init_std))
            self.cproj_quarter_diag_scale = nn.Parameter(torch.tensor(float(config.block_fht_cproj_quarter_diag_scale_init)))
        else:
            self.cproj_quarter_diag_weight = None
            self.cproj_quarter_diag_scale = None
        spectral_rank = int(config.block_fht_cproj_spectral_resid_rank)
        if spectral_rank > 0:
            if spectral_rank > config.n_embd or spectral_rank > 4 * config.n_embd:
                raise ValueError("block_fht_cproj_spectral_resid_rank must be <= n_embd and <= 4*n_embd")
            self.cproj_spectral_resid_diag = nn.Parameter(torch.zeros(spectral_rank))
            self.cproj_spectral_resid_scale = nn.Parameter(torch.tensor(float(config.block_fht_cproj_spectral_resid_scale_init)))
            spectral_seed = int(config.block_fht_cproj_spectral_resid_seed)
            self.cproj_spectral_resid_in_mix = FixedFHTMix(4 * config.n_embd, spectral_seed + layer_id * 64 + 129)
            self.cproj_spectral_resid_out_mix = FixedFHTMix(config.n_embd, spectral_seed + layer_id * 64 + 130)
        else:
            self.cproj_spectral_resid_diag = None
            self.cproj_spectral_resid_scale = None
            self.cproj_spectral_resid_in_mix = None
            self.cproj_spectral_resid_out_mix = None
        self.postgelu_std_target = float(config.block_fht_ffn_postgelu_std_target)
        self.last_postgelu: torch.Tensor | None = None

    def _fused_cached_cproj_lowrank(self, activated: torch.Tensor) -> torch.Tensor | None:
        if self.cproj_lowrank_left is None or self.cproj_lowrank_right is None:
            return None
        if not isinstance(self.cproj_lowrank_left, nn.Module) or not isinstance(self.cproj_lowrank_right, nn.Module):
            return None
        c_proj_weight = getattr(self.c_proj, "_cached_weight", None)
        left_weight = getattr(self.cproj_lowrank_left, "_cached_weight", None)
        right_weight = getattr(self.cproj_lowrank_right, "_cached_weight", None)
        if c_proj_weight is None or left_weight is None or right_weight is None:
            return None
        if getattr(self.cproj_lowrank_left, "bias", None) is not None or getattr(self.cproj_lowrank_right, "bias", None) is not None:
            return None
        c_proj_bias = getattr(self.c_proj, "bias", None)
        combined_weight = c_proj_weight + self.cproj_lowrank_scale * (right_weight @ left_weight)
        return F.linear(activated, combined_weight, c_proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.c_fc(x)
        if self.lowrank_left is not None and self.lowrank_right is not None:
            residual = x.matmul(self.lowrank_left.to(dtype=x.dtype)).matmul(self.lowrank_right.to(dtype=x.dtype))
            hidden = hidden + self.lowrank_scale * residual
        if self.pregelu_gain is not None:
            hidden = hidden * self.pregelu_gain.to(dtype=hidden.dtype)
        if self.pregelu_bias is not None:
            hidden = hidden + self.pregelu_bias.to(dtype=hidden.dtype)
        activated = self.gelu(hidden)
        if self.training and self.postgelu_std_target > 0.0:
            self.last_postgelu = activated
        else:
            self.last_postgelu = None
        out = self._fused_cached_cproj_lowrank(activated)
        fused_cproj_lowrank = out is not None
        if not fused_cproj_lowrank:
            out = self.c_proj(activated)
        if not fused_cproj_lowrank and self.cproj_lowrank_left is not None and self.cproj_lowrank_right is not None:
            if isinstance(self.cproj_lowrank_left, nn.Parameter) and isinstance(self.cproj_lowrank_right, nn.Parameter):
                delta = activated.matmul(self.cproj_lowrank_left.to(dtype=activated.dtype)).matmul(self.cproj_lowrank_right.to(dtype=activated.dtype))
            else:
                delta = self.cproj_lowrank_right(self.cproj_lowrank_left(activated))
            out = out + self.cproj_lowrank_scale * delta
        if self.cproj_tied_cfc_scale is not None:
            tied = F.linear(activated, self.c_fc.weight.t().to(dtype=activated.dtype))
            out = out + self.cproj_tied_cfc_scale.to(dtype=out.dtype) * tied
        if self.cproj_quarter_diag_weight is not None and self.cproj_quarter_diag_scale is not None:
            chunks = activated.view(*activated.shape[:-1], 4, -1)
            quarter = (chunks * self.cproj_quarter_diag_weight.to(dtype=activated.dtype)).sum(dim=-2)
            out = out + self.cproj_quarter_diag_scale.to(dtype=out.dtype) * quarter
        if (
            self.cproj_spectral_resid_diag is not None
            and self.cproj_spectral_resid_scale is not None
            and self.cproj_spectral_resid_in_mix is not None
            and self.cproj_spectral_resid_out_mix is not None
        ):
            rank = self.cproj_spectral_resid_diag.shape[0]
            mixed_in = self.cproj_spectral_resid_in_mix(activated)[..., :rank]
            spectral = mixed_in * self.cproj_spectral_resid_diag.to(dtype=activated.dtype)
            if rank < out.shape[-1]:
                spectral = F.pad(spectral, (0, out.shape[-1] - rank))
            spectral = self.cproj_spectral_resid_out_mix(spectral)
            out = out + self.cproj_spectral_resid_scale.to(dtype=out.dtype) * spectral
        return self.dropout(out)

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        if self.last_postgelu is None or self.postgelu_std_target <= 0.0:
            return None
        values = self.last_postgelu.float().reshape(-1, self.last_postgelu.shape[-1])
        std = values.std(dim=0, unbiased=False)
        target = std.new_tensor(self.postgelu_std_target)
        return torch.relu(target - std).square().mean()


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config, layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        bsz, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def postgelu_spread_loss(self) -> torch.Tensor:
        losses = []
        for block in self.transformer.h:
            loss = block.mlp.postgelu_spread_loss()
            if loss is not None:
                losses.append(loss)
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        optimizer: str = "adamw",
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_adamw_lr_scale: float = 1.0,
    ):
        params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        decay = [param for _, param in params.items() if param.dim() >= 2]
        nodecay = [param for _, param in params.items() if param.dim() < 2]
        if optimizer == "muon":
            matrix = [
                param
                for name, param in params.items()
                if param.dim() >= 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name
            ]
            other = [
                param
                for name, param in params.items()
                if param.dim() < 2 or "wte" in name or "wpe" in name or "lm_head" in name
            ]
            optimizers = []
            if matrix:
                optimizers.append(
                    Muon(
                        matrix,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=muon_ns_steps,
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = 1.0
            if other:
                fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
                use_fused = fused_available and device_type == "cuda"
                extra_args = {"fused": True} if use_fused else {}
                adamw_lr = learning_rate * float(muon_adamw_lr_scale)
                optimizers.append(torch.optim.AdamW([{"params": other, "weight_decay": 0.0, "lr_scale": float(muon_adamw_lr_scale)}], lr=adamw_lr, betas=betas, **extra_args))
            else:
                adamw_lr = learning_rate * float(muon_adamw_lr_scale)
            print(
                f"optimizer=muon matrix_tensors={len(matrix)} adamw_other_tensors={len(other)} "
                f"momentum={muon_momentum} ns_steps={muon_ns_steps} "
                f"adamw_lr_scale={float(muon_adamw_lr_scale)} adamw_lr={adamw_lr}"
            )
            return MultiOptimizer(optimizers)
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        print(f"optimizer=adamw tensors: decay={len(decay)} nodecay={len(nodecay)} fused={use_fused}")
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra_args)


    def block_fht_stats(self) -> dict[str, int]:
        generated = 0
        latent = 0
        modules = 0
        for module in self.modules():
            if isinstance(module, BlockFHTLinear):
                modules += 1
                generated += module.in_features * module.out_features
                latent += module.generator.latent.numel()
        return {"modules": modules, "generated": generated, "latent": latent}

    def prepare_block_fht_cache(self, dtype: torch.dtype | None = None) -> None:
        prepare_block_fht_weight_cache(self, dtype=dtype)

    def flush_block_fht_cache(self) -> None:
        flush_block_fht_weight_cache(self)


def freeze_non_block_fht(model: nn.Module, train_embeddings: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, BlockFHTLinear):
            module.generator.latent.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
            if module.output_gain is not None:
                module.output_gain.requires_grad_(True)
            if module.input_gain is not None:
                module.input_gain.requires_grad_(True)
            if module.spectral_core is not None:
                module.spectral_core.requires_grad_(True)
                module.spectral_log_out_gain.requires_grad_(True)
                module.spectral_log_in_gain.requires_grad_(True)
    if train_embeddings and isinstance(model, GPT):
        model.transformer.wte.weight.requires_grad_(True)
        model.transformer.wpe.weight.requires_grad_(True)
        for param in model.transformer.ln_f.parameters():
            param.requires_grad_(True)
