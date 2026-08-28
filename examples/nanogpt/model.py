from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from examples.nanogpt.muon import GroupedMuon, Muon
from examples.nanogpt.muon_matched_givens import (
    MuonMatchedGivens,
    MuonMatchedGivensLinear,
)
from latent_weight_lab import (
    BlockFHTLinear,
    ProductFHTLinear,
    fixed_basis_transform,
    postgelu_multihead_mix,
    flush_block_fht_weight_cache,
    prepare_block_fht_weight_cache,
    restore_block_fht_weight_cache,
    suspend_block_fht_weight_cache,
)


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

    def consume_muon_matched_givens_diagnostics(
        self,
    ) -> list[dict[str, object]]:
        diagnostics: list[dict[str, object]] = []
        for optimizer in self.optimizers:
            consume = getattr(optimizer, "consume_diagnostics", None)
            if consume is not None:
                diagnostics.extend(consume())
        return diagnostics


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
    block_fht_muon_latent_targets: tuple[str, ...] = ()
    block_fht_muon_latent_rows: int = 32
    block_fht_layers: int = 2
    block_fht_seed: int = 1000
    block_fht_global_output: bool = False
    block_fht_latent_init_std: float = 0.02
    block_fht_modulation_alpha: float = 0.0
    block_fht_modulation_centered: bool = False
    block_fht_quadratic_targets: tuple[str, ...] = ()
    block_fht_quadratic_scale: float = 0.0
    block_fht_quadratic_seed_offset: int = 104729
    block_fht_match_gpt_init: bool = False
    block_fht_weight_scale: float | None = None
    block_fht_residual_base_scale: float = 0.0
    block_fht_affine_delta_targets: tuple[str, ...] = ()
    block_fht_affine_delta_scale: float = 1.0
    block_fht_output_gain_targets: tuple[str, ...] = ()
    block_fht_input_gain_targets: tuple[str, ...] = ()
    block_fht_attn_cayley_targets: tuple[str, ...] = ()
    block_fht_attn_cayley_output_targets: tuple[str, ...] = ()
    block_fht_attn_cayley_bilateral_targets: tuple[str, ...] = ()
    block_fht_attn_cayley_rank: int = 0
    block_fht_attn_cayley_ranks: dict[str, int] | None = None
    block_fht_attn_cayley_scale: float = 1.0
    block_fht_attn_cayley_seed: int = 618033
    block_fht_ffn_pregelu_gain: bool = False
    block_fht_ffn_pregelu_bias: bool = False
    block_fht_ffn_pregelu_bias_init: float = 0.0
    block_fht_ffn_lowrank_rank: int = 0
    block_fht_ffn_lowrank_scale: float = 1.0
    block_fht_ffn_lowrank_init_std: float = 0.02
    compact_mlp_gradient_seeded_rank: int = 0
    compact_mlp_gradient_seeded_scale: float = 1.0
    compact_native_mlp: str = ""
    compact_native_mlp_branches: int = 4
    compact_native_mlp_paths: int = 3
    compact_native_mlp_seed: int = 20260827
    compact_native_mlp_group_size: int = 4
    compact_native_mlp_core_width: int = 736
    compact_native_mlp_factor_rank: int = 20
    compact_native_mlp_shared_width: int = 236
    compact_native_mlp_private_width: int = 10
    compact_native_mlp_reflectors_per_side: int = 4
    compact_native_mlp_householder_epsilon: float = 1e-6
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
    block_fht_cproj_spectral_resid_muon_matrix: bool = False
    block_fht_cproj_spectral_resid_full_core: bool = False
    block_fht_cproj_product_fht_factors: int = 0
    block_fht_cproj_product_fht_diagonal_scale: float = 1.0
    block_fht_cproj_product_fht_weight_space_muon: bool = True
    block_fht_cproj_product_fht_natural_gradient: bool = True
    block_fht_cproj_product_fht_pullback_normalize: bool = False
    block_fht_cproj_product_fht_pullback_max_coordinate_update: float = 0.02
    block_fht_cproj_product_fht_pullback_refresh_interval: int = 1
    block_fht_cproj_product_fht_pullback_probe: bool = False
    block_fht_cproj_product_fht_muon_momentum: float = 0.95
    block_fht_cproj_product_fht_muon_ns_steps: int = 5
    block_fht_mlp_cproj_muon_matched_givens: bool = False
    block_fht_mlp_cproj_muon_matched_givens_stages: int = 32
    block_fht_mlp_cproj_muon_matched_givens_residual_stages: int = 0
    block_fht_mlp_cproj_muon_matched_givens_neighbors: int = 64
    block_fht_mlp_cproj_muon_matched_givens_refresh_interval: int = 60
    block_fht_mlp_cproj_muon_matched_givens_fast_fresh: bool = False
    block_fht_mlp_cproj_muon_matched_givens_seed: int = 161803
    block_fht_ffn_postgelu_std_target: float = 0.0
    block_fht_mlp_shared_hidden_gain: bool = False
    block_fht_mlp_shared_hidden_gain_scale: float = 1.0
    block_fht_mlp_activation_chart: bool = False
    block_fht_mlp_activation_chart_channel_scale: float = 1.0
    block_fht_mlp_activation_chart_common_scale: float = 1.0
    block_fht_mlp_activation_chart_gauge_scale: float = 1.0
    block_fht_mlp_pregelu_block_rotation_stages: int = 0
    block_fht_mlp_pregelu_block_rotation_size: int = 32
    block_fht_mlp_pregelu_block_rotation_basis_size: int = 256
    block_fht_mlp_pregelu_block_rotation_coordinate_scale: float = 1.0
    block_fht_mlp_pregelu_block_rotation_seed: int = 161803
    block_fht_mlp_pregelu_cache_retain_graph: bool = False
    block_fht_mlp_hidden_block_rotation_stages: int = 0
    block_fht_mlp_hidden_block_rotation_size: int = 32
    block_fht_mlp_hidden_block_rotation_basis_size: int = 256
    block_fht_mlp_hidden_block_rotation_coordinate_scale: float = 1.0
    block_fht_mlp_hidden_block_rotation_seed: int = 314159
    block_fht_mlp_hidden_gain: bool = False
    block_fht_mlp_hidden_gain_scale: float = 1.0
    block_fht_mlp_hidden_log_gain_init: float = 0.0
    block_fht_mlp_output_rotation_stages: int = 0
    block_fht_mlp_output_rotation_seed: int = 271828
    block_fht_mlp_output_block_rotation_stages: int = 0
    block_fht_mlp_output_block_rotation_size: int = 32
    block_fht_mlp_output_block_rotation_basis_size: int = 256
    block_fht_mlp_output_block_rotation_coordinate_scale: float = 1.0
    block_fht_mlp_residual_output_gain: bool = False
    block_fht_mlp_residual_output_gain_scale: float = 1.0
    block_fht_mlp_residual_output_log_gain_init: float = 0.0
    block_fht_mlp_residual_conditioned_output_gate: bool = False
    block_fht_mlp_residual_conditioned_output_gate_scale: float = 1.0
    block_fht_mlp_residual_conditioned_output_gate_layers: tuple[int, ...] = ()
    block_fht_mlp_residual_conditioned_output_gate_bias: bool = True
    block_fht_mlp_residual_conditioned_output_gate_fixed_basis: bool = False
    block_fht_mlp_residual_conditioned_output_gate_untied_bases: bool = False
    block_fht_mlp_residual_conditioned_output_gate_basis_block_size: int = 256
    block_fht_mlp_residual_conditioned_output_gate_basis_seed: int = 271828
    block_fht_mlp_residual_conditioned_output_gate_update_basis_seed: int = 376557
    block_fht_mlp_residual_conditioned_output_gate_output_basis_seed: int = 481286
    block_fht_mlp_conditioned_output_gate_source: str = "residual"
    block_fht_mlp_conditioned_output_gate_projection_seed: int = 586015
    block_fht_mlp_conditioned_output_gate_rms_epsilon: float = 1e-6
    block_fht_mlp_postgelu_hidden_self_gate: bool = False
    block_fht_mlp_postgelu_hidden_self_gate_scale: float = 1.0
    block_fht_mlp_postgelu_hidden_self_gate_layers: tuple[int, ...] = ()
    block_fht_mlp_postgelu_hidden_self_gate_heads: int = 1
    block_fht_mlp_postgelu_hidden_self_gate_head_seed_stride: int = 1000003
    block_fht_mlp_postgelu_hidden_self_gate_basis_block_size: int = 256
    block_fht_mlp_postgelu_hidden_self_gate_condition_basis_seed: int = 271828
    block_fht_mlp_postgelu_hidden_self_gate_update_basis_seed: int = 376557
    block_fht_mlp_postgelu_hidden_self_gate_output_basis_seed: int = 481286
    block_fht_mlp_postgelu_hidden_self_gate_rms_epsilon: float = 1e-6
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
        # Pin generator-driven initialization to CPU even when a caller uses
        # ``with torch.device("cuda")`` for fast checkpoint construction.
        # PyTorch otherwise inherits the ambient CUDA device for randint and
        # rejects the explicitly CPU generator.
        signs = (
            torch.randint(
                0,
                2,
                (self.padded,),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            * 2.0
            - 1.0
        )
        self.register_buffer("signs", signs, persistent=True)

    def basis_columns(self, rank: int) -> torch.Tensor:
        """Materialize exact leading columns of the signed normalized FHT.

        A low-rank linear correction should not execute the Python-stage FHT
        over every token.  These deterministic columns let the same transform
        run as two accelerator GEMMs around a learned diagonal spectrum.
        """
        rank = int(rank)
        if rank <= 0 or rank > self.features:
            raise ValueError(f"rank must be in [1, {self.features}]")
        rows = torch.arange(
            self.features,
            dtype=torch.int64,
            device=self.signs.device,
        ).view(-1, 1)
        cols = torch.arange(
            rank,
            dtype=torch.int64,
            device=self.signs.device,
        ).view(1, -1)
        bits = rows.bitwise_and(cols)
        parity = torch.zeros_like(bits)
        while bits.any():
            parity = parity.bitwise_xor(bits.bitwise_and(1))
            bits = bits.bitwise_right_shift(1)
        hadamard = (1.0 - 2.0 * parity.float()) / math.sqrt(self.padded)
        return (
            self.signs[: self.features].view(-1, 1)
            * hadamard
            * self.signs[:rank].view(1, -1)
        )

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


class LearnedGivensOutputMix(nn.Module):
    """Identity-initialized compact orthogonal channel mixing."""

    def __init__(self, features: int, stages: int, seed: int) -> None:
        super().__init__()
        self.features = int(features)
        self.stages = int(stages)
        if self.features <= 0 or self.features % 2:
            raise ValueError("LearnedGivensOutputMix requires positive even features")
        if self.stages <= 0:
            raise ValueError("LearnedGivensOutputMix stages must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutations = torch.stack(
            [torch.randperm(self.features, generator=generator) for _ in range(self.stages)]
        )
        self.register_buffer("permutations", permutations, persistent=True)
        self.register_buffer(
            "inverse_permutations",
            torch.argsort(permutations, dim=1),
            persistent=True,
        )
        # Keep this one-dimensional so a Muon recipe assigns the angle
        # coordinate to its AdamW fallback, not to matrix orthogonalization.
        self.angles = nn.Parameter(torch.zeros(self.stages * (self.features // 2)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.features:
            raise ValueError(
                f"expected last dimension {self.features}, got {values.shape[-1]}"
            )
        result = values
        angles = self.angles.view(self.stages, self.features // 2)
        for stage in range(self.stages):
            permuted = result.index_select(-1, self.permutations[stage])
            pairs = permuted.reshape(*permuted.shape[:-1], self.features // 2, 2)
            angle = angles[stage].to(device=values.device, dtype=values.dtype)
            cosine = angle.cos()
            sine = angle.sin()
            first = cosine * pairs[..., 0] - sine * pairs[..., 1]
            second = sine * pairs[..., 0] + cosine * pairs[..., 1]
            rotated = torch.stack((first, second), dim=-1).reshape_as(permuted)
            result = rotated.index_select(-1, self.inverse_permutations[stage])
        return result


class LearnedLowRankCayleyMix(nn.Module):
    """Identity-initialized low-rank orthogonal channel chart.

    The skew generator is ``K = scale * (U V^T - V U^T)`` and the applied
    row-vector operator is the exact Cayley transform
    ``R = (I-K)^-1 (I+K)``.  ``U`` starts at zero, so the initial function is
    exactly unchanged; ``V`` starts as a seeded unit frame, which gives
    ``U`` a nonzero first-order task gradient.  Only two thin channel factors
    are learned—there is no learned dense basis or additive weight residual.

    Applying the transform uses a ``2r x 2r`` Woodbury solve and two thin
    matrix products, avoiding a materialized ``features x features`` rotation.
    """

    def __init__(
        self,
        features: int,
        rank: int,
        seed: int,
        coordinate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.features = int(features)
        self.rank = int(rank)
        self.coordinate_scale = float(coordinate_scale)
        if self.features <= 0:
            raise ValueError("features must be positive")
        if self.rank <= 0 or self.rank > self.features:
            raise ValueError("rank must be in [1, features]")
        if (
            not math.isfinite(self.coordinate_scale)
            or self.coordinate_scale <= 0.0
        ):
            raise ValueError("coordinate_scale must be positive and finite")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        right = torch.randn(
            self.features,
            self.rank,
            generator=generator,
            dtype=torch.float32,
        )
        right = F.normalize(right, dim=0)
        # Keep chart coordinates one-dimensional so the Muon recipe assigns
        # them to its AdamW fallback instead of orthogonalizing thin factors.
        self.left = nn.Parameter(torch.zeros(self.features * self.rank))
        self.right = nn.Parameter(right.reshape(-1))

        identity = torch.eye(self.rank, dtype=torch.float32)
        zero = torch.zeros_like(identity)
        symplectic = torch.cat(
            (
                torch.cat((zero, identity), dim=1),
                torch.cat((-identity, zero), dim=1),
            ),
            dim=0,
        )
        self.register_buffer("symplectic", symplectic, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.features:
            raise ValueError(
                f"expected last dimension {self.features}, "
                f"got {values.shape[-1]}"
            )
        solve_dtype = (
            torch.float32
            if values.dtype in (torch.float16, torch.bfloat16)
            else values.dtype
        )
        left = (
            self.coordinate_scale
            * self.left.to(device=values.device, dtype=solve_dtype)
        ).view(self.features, self.rank)
        right = F.normalize(
            self.right.to(device=values.device, dtype=solve_dtype).view(
                self.features, self.rank
            ),
            dim=0,
        )
        factors = torch.cat((left, right), dim=1)
        symplectic = self.symplectic.to(
            device=values.device, dtype=solve_dtype
        )
        gram = factors.transpose(0, 1) @ factors
        identity = torch.eye(
            2 * self.rank, device=values.device, dtype=solve_dtype
        )
        middle = torch.linalg.solve(
            identity - symplectic @ gram,
            symplectic,
        )

        factors_compute = factors.to(dtype=values.dtype)
        middle_compute = middle.to(dtype=values.dtype)
        projected = values @ factors_compute
        correction = (
            (projected @ middle_compute)
            @ factors_compute.transpose(0, 1)
        )
        return values + 2.0 * correction


class LearnedFHTBlockOrthogonalOutputMix(nn.Module):
    """Compact global output chart with fixed FHT bases and learned rotations.

    Each identity-initialized stage conjugates independent small Cayley
    rotations by a fixed signed, permuted block-FHT basis.  The basis has no
    learned parameters; only the upper-triangular coordinates of the
    block-skew generators are optimized.
    """

    def __init__(
        self,
        features: int,
        stages: int,
        rotation_block_size: int,
        basis_block_size: int,
        seed: int,
        coordinate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.features = int(features)
        self.stages = int(stages)
        self.rotation_block_size = int(rotation_block_size)
        self.basis_block_size = int(basis_block_size)
        self.coordinate_scale = float(coordinate_scale)
        if self.features <= 0:
            raise ValueError("features must be positive")
        if self.stages <= 0:
            raise ValueError("stages must be positive")
        if (
            not math.isfinite(self.coordinate_scale)
            or self.coordinate_scale <= 0.0
        ):
            raise ValueError("coordinate_scale must be positive and finite")
        if (
            self.rotation_block_size <= 1
            or self.features % self.rotation_block_size
        ):
            raise ValueError(
                "rotation_block_size must be > 1 and divide features"
            )
        if (
            self.basis_block_size <= 0
            or self.basis_block_size & (self.basis_block_size - 1)
            or self.features % self.basis_block_size
        ):
            raise ValueError(
                "basis_block_size must be a power of two dividing features"
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutations = torch.stack(
            [
                torch.randperm(
                    self.features, generator=generator, device="cpu"
                )
                for _ in range(self.stages)
            ]
        )
        signs = (
            torch.randint(
                0,
                2,
                (self.stages, self.features),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            * 2.0
            - 1.0
        )
        upper_rows, upper_columns = torch.triu_indices(
            self.rotation_block_size,
            self.rotation_block_size,
            offset=1,
            device="cpu",
        )
        self.register_buffer("permutations", permutations, persistent=True)
        self.register_buffer(
            "inverse_permutations",
            torch.argsort(permutations, dim=1),
            persistent=True,
        )
        self.register_buffer("signs", signs, persistent=True)
        self.register_buffer("upper_rows", upper_rows, persistent=False)
        self.register_buffer("upper_columns", upper_columns, persistent=False)
        self.rotation_blocks = self.features // self.rotation_block_size
        self.coordinates_per_block = (
            self.rotation_block_size * (self.rotation_block_size - 1) // 2
        )
        # Keep the parameter one-dimensional: these are chart coordinates,
        # not dense matrices for Muon's matrix update.
        self.coordinates = nn.Parameter(
            torch.zeros(
                self.stages
                * self.rotation_blocks
                * self.coordinates_per_block
            )
        )

    def _basis(
        self, values: torch.Tensor, stage: int, inverse: bool
    ) -> torch.Tensor:
        signs = self.signs[stage].to(
            device=values.device, dtype=values.dtype
        )
        if inverse:
            values = values * signs
            grouped = values.reshape(
                *values.shape[:-1],
                self.features // self.basis_block_size,
                self.basis_block_size,
            )
            values = normalized_fht_last_dim(grouped).reshape_as(values)
            return values.index_select(
                -1, self.inverse_permutations[stage]
            )
        values = values.index_select(-1, self.permutations[stage])
        grouped = values.reshape(
            *values.shape[:-1],
            self.features // self.basis_block_size,
            self.basis_block_size,
        )
        values = normalized_fht_last_dim(grouped).reshape_as(values)
        return values * signs

    def _rotations(
        self, stage: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        solve_dtype = (
            torch.float32
            if dtype in (torch.float16, torch.bfloat16)
            else dtype
        )
        coordinates = self.coordinates.view(
            self.stages,
            self.rotation_blocks,
            self.coordinates_per_block,
        )[stage].to(device=device, dtype=solve_dtype)
        coordinates = self.coordinate_scale * coordinates
        skew = coordinates.new_zeros(
            self.rotation_blocks,
            self.rotation_block_size,
            self.rotation_block_size,
        )
        rows = self.upper_rows.to(device=device)
        columns = self.upper_columns.to(device=device)
        skew[:, rows, columns] = coordinates
        skew[:, columns, rows] = -coordinates
        identity = torch.eye(
            self.rotation_block_size, device=device, dtype=solve_dtype
        ).expand(self.rotation_blocks, -1, -1)
        # Cayley(A) = (I-A)^-1(I+A) is exactly orthogonal for skew A and
        # equals identity at the zero-coordinate initialization.
        return torch.linalg.solve(
            identity - skew, identity + skew
        ).to(dtype=dtype)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.features:
            raise ValueError(
                f"expected last dimension {self.features}, "
                f"got {values.shape[-1]}"
            )
        result = values
        for stage in range(self.stages):
            result = self._basis(result, stage, inverse=False)
            blocks = result.reshape(
                *result.shape[:-1],
                self.rotation_blocks,
                self.rotation_block_size,
            )
            rotations = self._rotations(
                stage, device=result.device, dtype=result.dtype
            )
            blocks = torch.einsum("...gi,gij->...gj", blocks, rotations)
            result = self._basis(
                blocks.reshape_as(result), stage, inverse=True
            )
        return result

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the exact inverse row-vector operator without materializing it."""
        if values.shape[-1] != self.features:
            raise ValueError(
                f"expected last dimension {self.features}, "
                f"got {values.shape[-1]}"
            )
        result = values
        for stage in reversed(range(self.stages)):
            result = self._basis(result, stage, inverse=False)
            blocks = result.reshape(
                *result.shape[:-1],
                self.rotation_blocks,
                self.rotation_block_size,
            )
            rotations = self._rotations(
                stage, device=result.device, dtype=result.dtype
            )
            blocks = torch.einsum(
                "...gi,gij->...gj",
                blocks,
                rotations.transpose(-1, -2),
            )
            result = self._basis(
                blocks.reshape_as(result), stage, inverse=True
            )
        return result

    def matrix(self, reference: torch.Tensor) -> torch.Tensor:
        """Materialize the row-vector operator for efficient weight folding."""
        identity = torch.eye(
            self.features,
            device=reference.device,
            dtype=reference.dtype,
        )
        return self(identity)


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


class GradientSeededLowRankLinear(nn.Linear):
    """A frozen full-rank seed plus a gradient-bootstrapped rank-r delta.

    The dense seed is a non-persistent buffer initialized by GPT's normal
    initialization pass. Checkpoints therefore retain only the learned
    low-rank factors. A one-batch bootstrap sets the right factor to the top
    right singular vectors of the seed weight gradient while the left factor
    remains zero, preserving the exact seeded function.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        rank: int,
        scale: float,
        target_name: str,
    ) -> None:
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError(
                f"gradient-seeded rank must be in [1, {min(in_features, out_features)}]"
            )
        super().__init__(in_features, out_features, bias=bias)
        seed_weight = self.weight.detach()
        del self._parameters["weight"]
        self.register_buffer("weight", seed_weight, persistent=False)
        self.gradient_seeded_left = nn.Parameter(
            torch.zeros(out_features, rank)
        )
        self.gradient_seeded_right = nn.Parameter(
            torch.zeros(in_features, rank)
        )
        self.rank = int(rank)
        self.scale = float(scale)
        self.target_name = str(target_name)
        self._bootstrap_enabled = False
        self._bootstrap_complete = False

    def enable_gradient_bootstrap(self) -> None:
        if self._bootstrap_complete:
            raise RuntimeError("gradient bootstrap is already complete")
        self.weight.requires_grad_(True)
        self._bootstrap_enabled = True

    @torch.no_grad()
    def finish_gradient_bootstrap(self) -> dict[str, float | int | str]:
        if not self._bootstrap_enabled or self.weight.grad is None:
            raise RuntimeError("gradient bootstrap weight gradient is missing")
        gradient = self.weight.grad.float()
        _, singular_values, vh = torch.linalg.svd(
            gradient, full_matrices=False
        )
        right = vh[: self.rank].T
        self.gradient_seeded_left.zero_()
        self.gradient_seeded_right.copy_(
            right.to(dtype=self.gradient_seeded_right.dtype)
        )
        captured = singular_values[: self.rank].double().square().sum()
        total = singular_values.double().square().sum().clamp_min(1e-30)
        self.weight.grad = None
        self.weight.requires_grad_(False)
        self._bootstrap_enabled = False
        self._bootstrap_complete = True
        return {
            "target": self.target_name,
            "rank": self.rank,
            "gradient_energy_capture": float(captured / total),
            "fixed_seed_scalars": self.in_features * self.out_features,
            "trainable_factor_scalars": self.rank
            * (self.in_features + self.out_features),
        }

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base = F.linear(input, self.weight, self.bias)
        hidden = input.matmul(
            self.gradient_seeded_right.to(dtype=input.dtype)
        )
        delta = hidden.matmul(
            self.gradient_seeded_left.T.to(dtype=input.dtype)
        )
        return base + self.scale * delta


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
        latent_dim = max(1, round(in_features * out_features * latent_ratio))
        latent_shape = None
        if target_name in config.block_fht_muon_latent_targets:
            requested_rows = int(config.block_fht_muon_latent_rows)
            if requested_rows <= 0:
                raise ValueError("block_fht_muon_latent_rows must be positive")
            rows = min(requested_rows, latent_dim)
            columns = math.ceil(latent_dim / rows)
            latent_dim = rows * columns
            latent_shape = (rows, columns)
        target_std = 0.02
        if is_residual_projection_target(target_name):
            target_std = 0.02 / math.sqrt(2 * config.n_layer)
        product_fht_factors = int(
            config.block_fht_cproj_product_fht_factors
        )
        if target_name == "mlp.c_proj" and product_fht_factors > 0:
            incompatible = {
                "affine_delta": (
                    target_name in config.block_fht_affine_delta_targets
                ),
                "quadratic": (
                    target_name in config.block_fht_quadratic_targets
                ),
                "output_gain": (
                    target_name in config.block_fht_output_gain_targets
                ),
                "input_gain": (
                    target_name in config.block_fht_input_gain_targets
                ),
                "residual_base": (
                    float(config.block_fht_residual_base_scale) != 0.0
                ),
                "cproj_lowrank": (
                    int(config.block_fht_cproj_lowrank_rank) != 0
                ),
                "cproj_quarter_diag": bool(
                    config.block_fht_cproj_quarter_diag
                ),
                "cproj_spectral_residual": (
                    int(config.block_fht_cproj_spectral_resid_rank) != 0
                ),
            }
            enabled = [
                name for name, active in incompatible.items() if active
            ]
            if enabled:
                raise ValueError(
                    "product-FHT c_proj cannot be combined with: "
                    + ", ".join(enabled)
                )
            return ProductFHTLinear(
                in_features,
                out_features,
                bias=bias,
                factors=product_fht_factors,
                seed=config.block_fht_seed + seed_offset,
                weight_std=target_std,
                diagonal_scale=(
                    config.block_fht_cproj_product_fht_diagonal_scale
                ),
                weight_space_muon=(
                    config
                    .block_fht_cproj_product_fht_weight_space_muon
                ),
                muon_momentum=(
                    config.block_fht_cproj_product_fht_muon_momentum
                ),
                muon_ns_steps=(
                    config.block_fht_cproj_product_fht_muon_ns_steps
                ),
                natural_gradient=(
                    config
                    .block_fht_cproj_product_fht_natural_gradient
                ),
                pullback_normalize=(
                    config
                    .block_fht_cproj_product_fht_pullback_normalize
                ),
                pullback_max_coordinate_update=(
                    config
                    .block_fht_cproj_product_fht_pullback_max_coordinate_update
                ),
                pullback_refresh_interval=(
                    config
                    .block_fht_cproj_product_fht_pullback_refresh_interval
                ),
                pullback_probe=(
                    config
                    .block_fht_cproj_product_fht_pullback_probe
                ),
            )
        if config.block_fht_weight_scale is not None:
            weight_scale = float(config.block_fht_weight_scale)
        elif config.block_fht_match_gpt_init:
            weight_scale = target_std / float(config.block_fht_latent_init_std)
        else:
            weight_scale = 1.0
        affine_delta = target_name in config.block_fht_affine_delta_targets
        if affine_delta and config.block_fht_residual_base_scale != 0.0:
            raise ValueError(
                "target-selective affine deltas cannot be combined with the "
                "legacy global residual base"
            )
        residual_base_scale = (
            float(config.block_fht_affine_delta_scale)
            if affine_delta
            else float(config.block_fht_residual_base_scale)
        )
        if not math.isfinite(residual_base_scale):
            raise ValueError("BlockFHT residual/affine delta scale must be finite")
        return BlockFHTLinear(
            in_features,
            out_features,
            bias=bias,
            latent_dim=latent_dim,
            latent_shape=latent_shape,
            latent_ratio=latent_ratio,
            layers=config.block_fht_layers,
            seed=config.block_fht_seed + seed_offset,
            latent_init_std=config.block_fht_latent_init_std,
            weight_scale=weight_scale,
            modulation_alpha=config.block_fht_modulation_alpha,
            modulation_centered=config.block_fht_modulation_centered,
            quadratic_scale=(
                config.block_fht_quadratic_scale
                if target_name in config.block_fht_quadratic_targets
                else 0.0
            ),
            quadratic_seed_offset=config.block_fht_quadratic_seed_offset,
            residual_base_scale=residual_base_scale,
            residual_base_std=target_std,
            residual_delta_zero_init=affine_delta,
            output_gain=target_name in config.block_fht_output_gain_targets,
            input_gain=target_name in config.block_fht_input_gain_targets,
            spectral_rank=config.block_fht_ffn_spectral_rank if target_name == "mlp.c_fc" else 0,
            spectral_out_groups=config.block_fht_ffn_spectral_out_groups if target_name == "mlp.c_fc" else 1,
            spectral_in_groups=config.block_fht_ffn_spectral_in_groups if target_name == "mlp.c_fc" else 1,
            global_output=config.block_fht_global_output,
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
        cayley_targets = set(config.block_fht_attn_cayley_targets)
        cayley_output_targets = set(
            config.block_fht_attn_cayley_output_targets
        )
        cayley_bilateral_targets = set(
            config.block_fht_attn_cayley_bilateral_targets
        )
        supported_cayley_targets = {
            "attn.c_attn.qk_headwise",
            "attn.c_attn.v",
            "attn.c_proj",
        }
        unknown_cayley_targets = cayley_targets - supported_cayley_targets
        if unknown_cayley_targets:
            raise ValueError(
                "unsupported attention Cayley targets: "
                + ", ".join(sorted(unknown_cayley_targets))
            )
        if not cayley_targets.issubset(set(config.block_fht_targets)):
            raise ValueError(
                "attention Cayley targets must also be BlockFHT targets"
            )
        if not cayley_output_targets.issubset(cayley_targets):
            raise ValueError(
                "attention Cayley output targets must also be enabled "
                "attention Cayley targets"
            )
        if not cayley_bilateral_targets.issubset(cayley_targets):
            raise ValueError(
                "attention Cayley bilateral targets must also be enabled "
                "attention Cayley targets"
            )
        default_cayley_rank = int(config.block_fht_attn_cayley_rank)
        cayley_rank_overrides = {
            str(target): int(rank)
            for target, rank in (
                config.block_fht_attn_cayley_ranks or {}
            ).items()
        }
        unknown_rank_targets = (
            set(cayley_rank_overrides) - supported_cayley_targets
        )
        if unknown_rank_targets:
            raise ValueError(
                "unsupported attention Cayley rank targets: "
                + ", ".join(sorted(unknown_rank_targets))
            )
        if not set(cayley_rank_overrides).issubset(cayley_targets):
            raise ValueError(
                "attention Cayley rank overrides must target enabled "
                "attention Cayley targets"
            )
        cayley_ranks = {
            target: cayley_rank_overrides.get(
                target, default_cayley_rank
            )
            for target in cayley_targets
        }
        invalid_cayley_ranks = {
            target: rank
            for target, rank in cayley_ranks.items()
            if rank <= 0
        }
        if invalid_cayley_ranks:
            raise ValueError(
                "attention Cayley ranks must be positive: "
                + ", ".join(
                    f"{target}={rank}"
                    for target, rank in sorted(
                        invalid_cayley_ranks.items()
                    )
                )
            )
        if (
            not cayley_targets
            and (
                default_cayley_rank != 0
                or cayley_rank_overrides
            )
        ):
            raise ValueError(
                "attention Cayley ranks must be empty/zero when no targets "
                "are enabled"
            )
        if (
            "attn.c_attn.qk_headwise" in cayley_targets
            and not self.qk_headwise_c_attn
        ):
            raise ValueError(
                "the shared QK Cayley chart requires "
                "attn.c_attn.qk_headwise"
            )

        def cayley_mix(
            features: int,
            seed_offset: int,
            target: str,
        ) -> LearnedLowRankCayleyMix:
            return LearnedLowRankCayleyMix(
                features,
                cayley_ranks[target],
                int(config.block_fht_attn_cayley_seed)
                + layer_id * 64
                + seed_offset,
                coordinate_scale=float(config.block_fht_attn_cayley_scale),
            )

        self.qk_input_cayley = (
            cayley_mix(
                config.n_embd, 0, "attn.c_attn.qk_headwise"
            )
            if (
                "attn.c_attn.qk_headwise" in cayley_targets
                and (
                    "attn.c_attn.qk_headwise"
                    not in cayley_output_targets
                    or "attn.c_attn.qk_headwise"
                    in cayley_bilateral_targets
                )
            )
            else None
        )
        self.qk_output_cayley = (
            cayley_mix(
                2 * config.n_embd,
                3,
                "attn.c_attn.qk_headwise",
            )
            if (
                "attn.c_attn.qk_headwise" in cayley_output_targets
                or "attn.c_attn.qk_headwise"
                in cayley_bilateral_targets
            )
            else None
        )
        self.v_input_cayley = (
            cayley_mix(config.n_embd, 1, "attn.c_attn.v")
            if (
                "attn.c_attn.v" in cayley_targets
                and (
                    "attn.c_attn.v" not in cayley_output_targets
                    or "attn.c_attn.v" in cayley_bilateral_targets
                )
            )
            else None
        )
        self.v_output_cayley = (
            cayley_mix(config.n_embd, 4, "attn.c_attn.v")
            if (
                "attn.c_attn.v" in cayley_output_targets
                or "attn.c_attn.v" in cayley_bilateral_targets
            )
            else None
        )
        self.cproj_input_cayley = (
            cayley_mix(config.n_embd, 2, "attn.c_proj")
            if (
                "attn.c_proj" in cayley_targets
                and (
                    "attn.c_proj" not in cayley_output_targets
                    or "attn.c_proj" in cayley_bilateral_targets
                )
            )
            else None
        )
        self.cproj_output_cayley = (
            cayley_mix(config.n_embd, 5, "attn.c_proj")
            if (
                "attn.c_proj" in cayley_output_targets
                or "attn.c_proj" in cayley_bilateral_targets
            )
            else None
        )
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
        qk_input = (
            self.qk_input_cayley(x)
            if self.qk_input_cayley is not None
            else x
        )
        v_input = (
            self.v_input_cayley(x)
            if self.v_input_cayley is not None
            else x
        )
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
            qk = self.c_attn_qk_headwise(qk_input)
            if self.qk_output_cayley is not None:
                qk = self.qk_output_cayley(qk)
            q, k = qk.split(self.n_embd, dim=2)
            v = self.c_attn_v(v_input)
            if self.v_output_cayley is not None:
                v = self.v_output_cayley(v)
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
        if self.cproj_input_cayley is not None:
            y = self.cproj_input_cayley(y)
        y = self.c_proj(y)
        if self.cproj_output_cayley is not None:
            y = self.cproj_output_cayley(y)
        return self.resid_dropout(y)


class SparsePermutationLinear(nn.Module):
    """A square map with procedural connectivity and learned edge gains.

    Each output coordinate receives one identity edge and ``paths - 1``
    globally permuted edges.  The integer connectivity is regenerated from the
    configured seed and is deliberately absent from checkpoints; only the
    length-``width`` gain vectors are persistent training state.
    """

    def __init__(
        self,
        width: int,
        paths: int,
        seed: int,
        init_std: float,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("sparse-permutation width must be positive")
        if paths < 1:
            raise ValueError("sparse-permutation paths must be positive")
        if not math.isfinite(init_std) or init_std <= 0.0:
            raise ValueError("sparse-permutation init_std must be positive and finite")
        self.width = int(width)
        self.paths = int(paths)
        self.seed = int(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        for path in range(self.paths):
            permutation = (
                torch.arange(self.width, dtype=torch.long)
                if path == 0
                else torch.randperm(self.width, generator=generator)
            )
            self.register_buffer(
                f"permutation_{path}", permutation, persistent=False
            )
        self.gains = nn.ParameterList(
            [nn.Parameter(torch.empty(self.width)) for _ in range(self.paths)]
        )
        for gain in self.gains:
            nn.init.normal_(gain, mean=0.0, std=float(init_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.width:
            raise ValueError(
                f"sparse-permutation input width {x.size(-1)} != {self.width}"
            )
        out = torch.zeros_like(x)
        for path, gain in enumerate(self.gains):
            permutation = getattr(self, f"permutation_{path}")
            out = out + x.index_select(-1, permutation) * gain.to(dtype=x.dtype)
        return out


def _sparse_permutation_map(
    x: torch.Tensor,
    permutations: torch.Tensor,
    gains: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    out = torch.zeros_like(x)
    for permutation, gain in zip(permutations, gains, strict=True):
        out = out + x.index_select(-1, permutation) * gain.to(dtype=x.dtype)
    return out


def _sparse_permutation_map_backward(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    permutations: torch.Tensor,
    gains: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    grad_x = torch.zeros_like(x)
    grad_gains = []
    reduce_dims = tuple(range(x.dim() - 1))
    for permutation, gain in zip(permutations, gains, strict=True):
        gathered = x.index_select(-1, permutation)
        grad_gains.append(
            (grad_y * gathered).sum(dim=reduce_dims).to(dtype=gain.dtype)
        )
        index = permutation.view(*([1] * (x.dim() - 1)), -1).expand_as(x)
        grad_x.scatter_add_(
            -1,
            index,
            grad_y * gain.to(dtype=grad_y.dtype),
        )
    return grad_x, grad_gains


class _SparsePermutationSwiGLUFunction(torch.autograd.Function):
    """Memory-bounded exact VJP for the frozen sparse SwiGLU topology.

    The forward graph does not retain every gathered edge activation.  The
    backward recomputes each branch from the saved layer input, then applies
    the exact sparse transpose with scatter-add.  This changes arithmetic but
    not the represented function or its gradients.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        permutations: torch.Tensor,
        branches: int,
        paths: int,
        branch_scale: float,
        *gains: torch.Tensor,
    ) -> torch.Tensor:
        expected = int(branches) * 3 * int(paths)
        if len(gains) != expected:
            raise ValueError(
                f"expected {expected} sparse gain tensors, got {len(gains)}"
            )
        ctx.branches = int(branches)
        ctx.paths = int(paths)
        ctx.branch_scale = float(branch_scale)
        ctx.save_for_backward(x, permutations, *gains)
        out = torch.zeros_like(x)
        cursor = 0
        for _ in range(ctx.branches):
            up_gains = tuple(gains[cursor : cursor + ctx.paths])
            cursor += ctx.paths
            gate_gains = tuple(gains[cursor : cursor + ctx.paths])
            cursor += ctx.paths
            down_gains = tuple(gains[cursor : cursor + ctx.paths])
            cursor += ctx.paths
            branch_index = cursor // (3 * ctx.paths) - 1
            base = branch_index * 3 * ctx.paths
            up = _sparse_permutation_map(
                x, permutations[base : base + ctx.paths], up_gains
            )
            gate = _sparse_permutation_map(
                x,
                permutations[
                    base + ctx.paths : base + 2 * ctx.paths
                ],
                gate_gains,
            )
            hidden = F.silu(up) * gate
            out = out + _sparse_permutation_map(
                hidden,
                permutations[
                    base + 2 * ctx.paths : base + 3 * ctx.paths
                ],
                down_gains,
            )
        return out * ctx.branch_scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, permutations, *saved_gains = ctx.saved_tensors
        gains = tuple(saved_gains)
        grad_x = torch.zeros_like(x)
        grad_gains: list[torch.Tensor] = []
        branch_grad = grad_output * ctx.branch_scale
        cursor = 0
        for branch_index in range(ctx.branches):
            up_gains = gains[cursor : cursor + ctx.paths]
            cursor += ctx.paths
            gate_gains = gains[cursor : cursor + ctx.paths]
            cursor += ctx.paths
            down_gains = gains[cursor : cursor + ctx.paths]
            cursor += ctx.paths
            base = branch_index * 3 * ctx.paths
            up_permutations = permutations[base : base + ctx.paths]
            gate_permutations = permutations[
                base + ctx.paths : base + 2 * ctx.paths
            ]
            down_permutations = permutations[
                base + 2 * ctx.paths : base + 3 * ctx.paths
            ]
            up = _sparse_permutation_map(x, up_permutations, up_gains)
            gate = _sparse_permutation_map(
                x, gate_permutations, gate_gains
            )
            silu_up = F.silu(up)
            hidden = silu_up * gate
            grad_hidden, down_gain_grads = _sparse_permutation_map_backward(
                hidden, branch_grad, down_permutations, down_gains
            )
            sigmoid_up = torch.sigmoid(up)
            grad_up = (
                grad_hidden
                * gate
                * sigmoid_up
                * (1.0 + up * (1.0 - sigmoid_up))
            )
            grad_gate = grad_hidden * silu_up
            grad_x_up, up_gain_grads = _sparse_permutation_map_backward(
                x, grad_up, up_permutations, up_gains
            )
            grad_x_gate, gate_gain_grads = _sparse_permutation_map_backward(
                x, grad_gate, gate_permutations, gate_gains
            )
            grad_x = grad_x + grad_x_up + grad_x_gate
            grad_gains.extend(
                [*up_gain_grads, *gate_gain_grads, *down_gain_grads]
            )
        return (
            grad_x,
            None,
            None,
            None,
            None,
            *grad_gains,
        )


class SparsePermutationSwiGLUMLP(nn.Module):
    """Compact-native four-branch SwiGLU with no generated dense weights."""

    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        branches = int(config.compact_native_mlp_branches)
        paths = int(config.compact_native_mlp_paths)
        if branches != 4 or paths != 3:
            raise ValueError(
                "the frozen sparse-permutation SwiGLU gate requires "
                "branches=4 and paths=3"
            )
        width = int(config.n_embd)
        up_std = 0.02 * math.sqrt(width / paths)
        down_std = (
            0.02
            / math.sqrt(2 * config.n_layer)
            * math.sqrt((4 * width) / paths)
        )
        base_seed = int(config.compact_native_mlp_seed) + layer_id * 1009
        self.branches = nn.ModuleList()
        for branch in range(branches):
            branch_seed = base_seed + branch * 101
            self.branches.append(
                nn.ModuleDict(
                    {
                        "up": SparsePermutationLinear(
                            width, paths, branch_seed + 0, up_std
                        ),
                        "gate": SparsePermutationLinear(
                            width, paths, branch_seed + 31, up_std
                        ),
                        "down": SparsePermutationLinear(
                            width, paths, branch_seed + 67, down_std
                        ),
                    }
                )
            )
        self.branch_scale = 1.0 / math.sqrt(branches)
        self.dropout = nn.Dropout(config.dropout)
        # Preserve the interface consumed by Block and GPT diagnostics.
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        permutations = []
        gains = []
        for branch in self.branches:
            for name in ("up", "gate", "down"):
                module = branch[name]
                permutations.extend(
                    getattr(module, f"permutation_{path}")
                    for path in range(module.paths)
                )
                gains.extend(module.gains)
        out = _SparsePermutationSwiGLUFunction.apply(
            x,
            torch.stack(permutations),
            len(self.branches),
            self.branches[0]["up"].paths,
            self.branch_scale,
            *gains,
        )
        return self.dropout(out)

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        return None

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return None

    def prepare_charted_cfc_cache(self) -> None:
        return None

    def prepare_charted_cproj_cache(self) -> None:
        return None

    def flush_charted_cfc_cache(self) -> None:
        return None

    def flush_charted_cproj_cache(self) -> None:
        return None

    def suspend_charted_cfc_cache(self) -> None:
        return None

    def suspend_charted_cproj_cache(self) -> None:
        return None

    def restore_charted_cfc_cache(self, _value: None) -> None:
        return None

    def restore_charted_cproj_cache(self, _value: None) -> None:
        return None


class SharedGroupTransport(nn.Module):
    """Two global residual-group transports shared by every compact MLP."""

    def __init__(self, groups: int) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError("shared group transport needs positive groups")
        self.groups = int(groups)
        self.pre_delta = nn.Parameter(torch.zeros(groups, groups))
        self.post_delta = nn.Parameter(torch.zeros(groups, groups))
        self.register_buffer(
            "identity",
            torch.eye(groups),
            persistent=False,
        )

    def _apply_transport(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.identity.to(device=x.device, dtype=x.dtype) + delta.to(
            dtype=x.dtype
        )
        return F.linear(x, weight)

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_transport(x, self.pre_delta)

    def post(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_transport(x, self.post_delta)


class SharedTransportGroupedGELUMLP(nn.Module):
    """Update-closed local GELU blocks with shared global group transport."""

    def __init__(
        self,
        config: GPTConfig,
        layer_id: int,
        shared_transport: SharedGroupTransport,
    ) -> None:
        super().__init__()
        del layer_id
        group_size = int(config.compact_native_mlp_group_size)
        if group_size != 4:
            raise ValueError("the frozen shared-transport gate requires group size 4")
        if config.n_embd % group_size:
            raise ValueError("n_embd must be divisible by compact MLP group size")
        groups = config.n_embd // group_size
        if shared_transport.groups != groups:
            raise ValueError("shared transport group count does not match n_embd")
        self.group_size = group_size
        self.groups = groups
        self.hidden_per_group = 4 * group_size
        # The shared module is owned and serialized once by GPT.  Keep only a
        # non-registering reference here so state_dict cannot duplicate it for
        # every layer.
        object.__setattr__(self, "_shared_transport", shared_transport)
        self.grouped_c_fc_weight = nn.Parameter(
            torch.empty(groups, self.hidden_per_group, group_size)
        )
        self.grouped_c_proj_weight = nn.Parameter(
            torch.empty(groups, group_size, self.hidden_per_group)
        )
        c_fc_std = 0.02 * math.sqrt(config.n_embd / group_size)
        c_proj_std = (
            0.02
            / math.sqrt(2 * config.n_layer)
            * math.sqrt(config.n_embd / group_size)
        )
        nn.init.normal_(self.grouped_c_fc_weight, mean=0.0, std=c_fc_std)
        nn.init.normal_(self.grouped_c_proj_weight, mean=0.0, std=c_proj_std)
        self.dropout = nn.Dropout(config.dropout)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    @property
    def shared_transport(self) -> SharedGroupTransport:
        return object.__getattribute__(self, "_shared_transport")

    def _group_transport(
        self,
        x: torch.Tensor,
        role: str,
    ) -> torch.Tensor:
        shape = x.shape
        grouped = x.reshape(*shape[:-1], self.groups, self.group_size)
        # Each within-group coordinate gets a global groups x groups GEMM.
        grouped = grouped.transpose(-1, -2)
        if role == "pre":
            grouped = self.shared_transport.pre(grouped)
        elif role == "post":
            grouped = self.shared_transport.post(grouped)
        else:
            raise ValueError(f"unsupported group transport role {role!r}")
        return grouped.transpose(-1, -2).reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = self._group_transport(x, "pre")
        grouped = x.reshape(-1, self.groups, self.group_size).transpose(0, 1)
        hidden = torch.bmm(
            grouped,
            self.grouped_c_fc_weight.transpose(-2, -1),
        )
        hidden = F.gelu(hidden)
        grouped_output = torch.bmm(
            hidden,
            self.grouped_c_proj_weight.transpose(-2, -1),
        )
        output = grouped_output.transpose(0, 1).reshape(shape)
        return self.dropout(self._group_transport(output, "post"))

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        return None

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return None

    def prepare_charted_cfc_cache(self) -> None:
        return None

    def prepare_charted_cproj_cache(self) -> None:
        return None

    def flush_charted_cfc_cache(self) -> None:
        return None

    def flush_charted_cproj_cache(self) -> None:
        return None

    def suspend_charted_cfc_cache(self) -> None:
        return None

    def suspend_charted_cproj_cache(self) -> None:
        return None

    def restore_charted_cfc_cache(self, _value: None) -> None:
        return None

    def restore_charted_cproj_cache(self, _value: None) -> None:
        return None


class SharedDenseResidualCore(nn.Module):
    """One learned dense residual core, serialized once across depth."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        width = int(config.compact_native_mlp_core_width)
        if width != 736:
            raise ValueError("the frozen shared dense-core gate requires width 736")
        if width > config.n_embd:
            raise ValueError("shared dense-core width cannot exceed n_embd")
        self.width = width
        self.weight = nn.Parameter(torch.empty(width, width))
        output_std = (
            0.02
            / math.sqrt(2 * config.n_layer)
            * math.sqrt((4 * config.n_embd) / width)
        )
        nn.init.normal_(self.weight, mean=0.0, std=output_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class SharedDenseCoreResidualMLP(nn.Module):
    """Layer-conjugated ``fc(GELU(x))`` with one shared dense core."""

    def __init__(
        self,
        config: GPTConfig,
        layer_id: int,
        shared_core: SharedDenseResidualCore,
    ) -> None:
        super().__init__()
        if shared_core.width != int(config.compact_native_mlp_core_width):
            raise ValueError("shared dense-core width does not match config")
        self.width = int(config.n_embd)
        self.core_width = int(shared_core.width)
        object.__setattr__(self, "_shared_core", shared_core)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(config.compact_native_mlp_seed) + int(layer_id) * 1009
        )
        permutation = torch.randperm(self.width, generator=generator)
        inverse_permutation = torch.argsort(permutation)
        sign = (
            torch.randint(
                0,
                2,
                (self.width,),
                generator=generator,
                dtype=torch.int64,
            )
            .mul(2)
            .sub(1)
            .float()
        )
        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer(
            "inverse_permutation",
            inverse_permutation,
            persistent=False,
        )
        self.register_buffer("sign", sign, persistent=False)
        pregelu_scale = 0.02 * math.sqrt(config.n_embd)
        self.input_gain = nn.Parameter(
            torch.full((self.width,), pregelu_scale)
        )
        output_gain = torch.ones(self.width)
        output_gain[self.core_width :] = 0.0
        self.output_gain = nn.Parameter(output_gain)
        self.dropout = nn.Dropout(config.dropout)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    @property
    def shared_core(self) -> SharedDenseResidualCore:
        return object.__getattribute__(self, "_shared_core")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sign = self.sign.to(dtype=x.dtype)
        signed = x.index_select(-1, self.permutation) * sign
        activated = F.gelu(
            signed * self.input_gain.to(dtype=x.dtype)
        )
        main = self.shared_core(activated[..., : self.core_width])
        tail = activated[..., self.core_width :]
        conjugated = torch.cat((main, tail), dim=-1)
        conjugated = conjugated * self.output_gain.to(dtype=x.dtype)
        output = (conjugated * sign).index_select(
            -1,
            self.inverse_permutation,
        )
        return self.dropout(output)

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        return None

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return None

    def prepare_charted_cfc_cache(self) -> None:
        return None

    def prepare_charted_cproj_cache(self) -> None:
        return None

    def flush_charted_cfc_cache(self) -> None:
        return None

    def flush_charted_cproj_cache(self) -> None:
        return None

    def suspend_charted_cfc_cache(self) -> None:
        return None

    def suspend_charted_cproj_cache(self) -> None:
        return None

    def restore_charted_cfc_cache(self, _value: None) -> None:
        return None

    def restore_charted_cproj_cache(self, _value: None) -> None:
        return None


class SharedRecurrentDenseCore(nn.Module):
    """One learned square core reused by every layer and recurrent step."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        width = int(config.compact_native_mlp_core_width)
        if config.n_embd == 768 and width != 720:
            raise ValueError("the frozen recurrent dense-core gate requires width 720")
        if width <= 0 or width > config.n_embd:
            raise ValueError("recurrent dense-core width must be in (0, n_embd]")
        self.width = width
        self.weight = nn.Parameter(torch.empty(width, width))
        # Two procedurally decorrelated writes are added inside one residual
        # block.  Give each half the single-write variance so their sum matches
        # the ordinary GPT MLP residual-write scale at initialization.
        output_std = (
            0.02
            / math.sqrt(2 * config.n_layer)
            * math.sqrt((4 * config.n_embd) / (2 * width))
        )
        nn.init.normal_(self.weight, mean=0.0, std=output_std)
        self.init_std = output_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class TwoStepRecurrentDenseCoreMLP(nn.Module):
    """Two residual writes through one shared core in private signed gauges."""

    depth = 2

    def __init__(
        self,
        config: GPTConfig,
        layer_id: int,
        shared_core: SharedRecurrentDenseCore,
    ) -> None:
        super().__init__()
        if shared_core.width != int(config.compact_native_mlp_core_width):
            raise ValueError("shared recurrent core width does not match config")
        self.width = int(config.n_embd)
        self.core_width = int(shared_core.width)
        object.__setattr__(self, "_shared_core", shared_core)

        permutations = []
        inverse_permutations = []
        signs = []
        for step in range(self.depth):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                int(config.compact_native_mlp_seed)
                + int(layer_id) * 1009
                + step * 104729
            )
            permutation = torch.randperm(self.width, generator=generator)
            permutations.append(permutation)
            inverse_permutations.append(torch.argsort(permutation))
            signs.append(
                torch.randint(
                    0,
                    2,
                    (self.width,),
                    generator=generator,
                    dtype=torch.int64,
                )
                .mul(2)
                .sub(1)
                .float()
            )
        self.register_buffer(
            "permutations",
            torch.stack(permutations),
            persistent=False,
        )
        self.register_buffer(
            "inverse_permutations",
            torch.stack(inverse_permutations),
            persistent=False,
        )
        self.register_buffer("signs", torch.stack(signs), persistent=False)

        pregelu_scale = 0.02 * math.sqrt(config.n_embd)
        self.input_gain = nn.ParameterList(
            [
                nn.Parameter(torch.full((self.core_width,), pregelu_scale))
                for _ in range(self.depth)
            ]
        )
        self.output_gain = nn.ParameterList(
            [
                nn.Parameter(torch.ones(self.core_width))
                for _ in range(self.depth)
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    @property
    def shared_core(self) -> SharedRecurrentDenseCore:
        return object.__getattribute__(self, "_shared_core")

    def _folded_step(self, state: torch.Tensor, step: int) -> torch.Tensor:
        """Evaluate one signed-conjugated update without activation gathers."""
        inverse = self.inverse_permutations[step]
        sign = self.signs[step].to(dtype=state.dtype)
        tail = self.width - self.core_width

        input_gain = F.pad(
            self.input_gain[step].to(dtype=state.dtype),
            (0, tail),
        )
        ambient_input_gain = (input_gain * sign).index_select(0, inverse)
        activated = F.gelu(state * ambient_input_gain)

        padded_core = F.pad(
            self.shared_core.weight,
            (0, tail, 0, tail),
        )
        folded_core = padded_core.index_select(0, inverse).index_select(
            1,
            inverse,
        )
        output_gain = F.pad(
            self.output_gain[step].to(dtype=folded_core.dtype),
            (0, tail),
        )
        ambient_output_gain = (output_gain * sign.to(folded_core.dtype)).index_select(
            0,
            inverse,
        )
        folded_core = folded_core * ambient_output_gain.unsqueeze(-1)
        return F.linear(activated, folded_core)

    def activation_space_step(
        self,
        state: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Unfolded reference formula used by exact equivalence tests."""
        permutation = self.permutations[step]
        inverse = self.inverse_permutations[step]
        sign = self.signs[step].to(dtype=state.dtype)
        signed = state.index_select(-1, permutation) * sign
        activated = F.gelu(
            signed[..., : self.core_width]
            * self.input_gain[step].to(dtype=state.dtype)
        )
        main = self.shared_core(activated)
        main = main * self.output_gain[step].to(dtype=main.dtype)
        conjugated = F.pad(main, (0, self.width - self.core_width))
        return (conjugated * sign).index_select(-1, inverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        for step in range(self.depth):
            state = state + self._folded_step(state, step)
        # Block.forward owns the outer skip, so this module returns only the
        # accumulated write.  Returning state would add x twice.
        return self.dropout(state - x)

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        return None

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return None

    def prepare_charted_cfc_cache(self) -> None:
        return None

    def prepare_charted_cproj_cache(self) -> None:
        return None

    def flush_charted_cfc_cache(self) -> None:
        return None

    def flush_charted_cproj_cache(self) -> None:
        return None

    def suspend_charted_cfc_cache(self) -> None:
        return None

    def suspend_charted_cproj_cache(self) -> None:
        return None

    def restore_charted_cfc_cache(self, _value: None) -> None:
        return None

    def restore_charted_cproj_cache(self, _value: None) -> None:
        return None


class SharedPairedAtomBank(nn.Module):
    """One shared nonlinear atom bank, serialized once across depth."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        width = int(config.n_embd)
        shared_width = int(config.compact_native_mlp_shared_width)
        private_width = int(config.compact_native_mlp_private_width)
        if config.n_embd == 768 and (shared_width, private_width) != (236, 10):
            raise ValueError(
                "the frozen paired-atom gate requires shared/private widths 236/10"
            )
        if shared_width <= 0 or private_width <= 0:
            raise ValueError("paired-atom widths must be positive")
        self.width = width
        self.shared_width = shared_width
        self.private_width = private_width
        self.input_weight = nn.Parameter(torch.empty(shared_width, width))
        self.output_weight = nn.Parameter(torch.empty(width, shared_width))
        nn.init.normal_(self.input_weight, mean=0.0, std=0.02)
        output_std = (
            0.02
            / math.sqrt(2 * config.n_layer)
            * math.sqrt((4 * width) / (shared_width + private_width))
        )
        nn.init.normal_(self.output_weight, mean=0.0, std=output_std)
        self.output_init_std = output_std


class PairedSharedPrivateAtomMLP(nn.Module):
    """Shared nonlinear atoms plus layer-private input/output atom pairs."""

    def __init__(
        self,
        config: GPTConfig,
        layer_id: int,
        shared_bank: SharedPairedAtomBank,
    ) -> None:
        super().__init__()
        if shared_bank.width != int(config.n_embd):
            raise ValueError("paired-atom bank width does not match config")
        if shared_bank.shared_width != int(config.compact_native_mlp_shared_width):
            raise ValueError("shared paired-atom width does not match config")
        if shared_bank.private_width != int(config.compact_native_mlp_private_width):
            raise ValueError("private paired-atom width does not match config")
        self.width = int(config.n_embd)
        self.shared_width = int(shared_bank.shared_width)
        self.private_width = int(shared_bank.private_width)
        object.__setattr__(self, "_shared_bank", shared_bank)

        self.private_input_weight = nn.Parameter(
            torch.empty(self.private_width, self.width)
        )
        self.private_output_weight = nn.Parameter(
            torch.empty(self.width, self.private_width)
        )
        nn.init.normal_(self.private_input_weight, mean=0.0, std=0.02)
        nn.init.normal_(
            self.private_output_weight,
            mean=0.0,
            std=shared_bank.output_init_std,
        )
        self.shared_hidden_gain = nn.Parameter(torch.ones(self.shared_width))
        self.output_gain = nn.Parameter(torch.ones(self.width))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(config.compact_native_mlp_seed) + int(layer_id) * 1009
        )
        permutation = torch.randperm(self.width, generator=generator)
        inverse_permutation = torch.argsort(permutation)
        sign = (
            torch.randint(
                0,
                2,
                (self.width,),
                generator=generator,
                dtype=torch.int64,
            )
            .mul(2)
            .sub(1)
            .float()
        )
        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer(
            "inverse_permutation",
            inverse_permutation,
            persistent=False,
        )
        self.register_buffer("sign", sign, persistent=False)
        self.dropout = nn.Dropout(config.dropout)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    @property
    def shared_bank(self) -> SharedPairedAtomBank:
        return object.__getattribute__(self, "_shared_bank")

    def _fold_input_weight(self, weight: torch.Tensor) -> torch.Tensor:
        inverse = self.inverse_permutation
        sign = self.sign.to(dtype=weight.dtype)
        return weight.index_select(-1, inverse) * sign.index_select(0, inverse)

    def _fold_output_weight(self, weight: torch.Tensor) -> torch.Tensor:
        inverse = self.inverse_permutation
        sign = self.sign.to(dtype=weight.dtype)
        return weight.index_select(0, inverse) * sign.index_select(
            0, inverse
        ).unsqueeze(-1)

    def activation_space_shared_write(self, x: torch.Tensor) -> torch.Tensor:
        """Unfolded reference formula used by exact equivalence tests."""
        sign = self.sign.to(dtype=x.dtype)
        signed = x.index_select(-1, self.permutation) * sign
        hidden = F.gelu(F.linear(signed, self.shared_bank.input_weight))
        hidden = hidden * self.shared_hidden_gain.to(dtype=hidden.dtype)
        conjugated = F.linear(hidden, self.shared_bank.output_weight)
        return (conjugated * sign).index_select(-1, self.inverse_permutation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fold the layer's procedural signed gauge into the shared matrices.
        # The live token path therefore contains only dense GEMMs and GELU.
        shared_hidden = F.gelu(
            F.linear(x, self._fold_input_weight(self.shared_bank.input_weight))
        )
        shared_hidden = shared_hidden * self.shared_hidden_gain.to(
            dtype=shared_hidden.dtype
        )
        shared_write = F.linear(
            shared_hidden,
            self._fold_output_weight(self.shared_bank.output_weight),
        )
        private_hidden = F.gelu(F.linear(x, self.private_input_weight))
        private_write = F.linear(private_hidden, self.private_output_weight)
        output = (shared_write + private_write) * self.output_gain.to(dtype=x.dtype)
        return self.dropout(output)

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        return None

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return None

    def prepare_charted_cfc_cache(self) -> None:
        return None

    def prepare_charted_cproj_cache(self) -> None:
        return None

    def flush_charted_cfc_cache(self) -> None:
        return None

    def flush_charted_cproj_cache(self) -> None:
        return None

    def suspend_charted_cfc_cache(self) -> None:
        return None

    def suspend_charted_cproj_cache(self) -> None:
        return None

    def restore_charted_cfc_cache(self, _value: None) -> None:
        return None

    def restore_charted_cproj_cache(self, _value: None) -> None:
        return None


class SharedHouseholderAtomBank(nn.Module):
    """One complete nonlinear atom bank shared across transformer depth."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        width = int(config.n_embd)
        atom_width = int(config.compact_native_mlp_shared_width)
        reflectors = int(config.compact_native_mlp_reflectors_per_side)
        if config.n_embd == 768 and (atom_width, reflectors) != (312, 4):
            raise ValueError(
                "the frozen Householder-frame gate requires width 312 and "
                "four reflectors per side"
            )
        if atom_width <= 0 or reflectors <= 0 or reflectors % 2:
            raise ValueError(
                "Householder atom width must be positive and reflector count "
                "must be a positive even number"
            )
        self.width = width
        self.atom_width = atom_width
        self.reflectors_per_side = reflectors
        self.input_weight = nn.Parameter(torch.empty(atom_width, width))
        self.output_weight = nn.Parameter(torch.empty(width, atom_width))
        nn.init.normal_(self.input_weight, mean=0.0, std=0.02)
        output_std = (
            0.02
            / math.sqrt(2 * config.n_layer)
            * math.sqrt((4 * width) / atom_width)
        )
        nn.init.normal_(self.output_weight, mean=0.0, std=output_std)
        self.output_init_std = output_std


class HouseholderFrameSharedAtomMLP(nn.Module):
    """Shared atom bank with layer-private global input/output frame motion."""

    def __init__(
        self,
        config: GPTConfig,
        layer_id: int,
        shared_bank: SharedHouseholderAtomBank,
    ) -> None:
        super().__init__()
        if shared_bank.width != int(config.n_embd):
            raise ValueError("shared Householder bank width does not match config")
        if shared_bank.atom_width != int(config.compact_native_mlp_shared_width):
            raise ValueError("shared Householder atom width does not match config")
        if shared_bank.reflectors_per_side != int(
            config.compact_native_mlp_reflectors_per_side
        ):
            raise ValueError("Householder reflector count does not match config")
        epsilon = float(config.compact_native_mlp_householder_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("Householder epsilon must be positive and finite")
        self.width = int(shared_bank.width)
        self.atom_width = int(shared_bank.atom_width)
        self.reflectors_per_side = int(shared_bank.reflectors_per_side)
        self.householder_epsilon = epsilon
        object.__setattr__(self, "_shared_bank", shared_bank)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(config.compact_native_mlp_seed) + int(layer_id) * 1009
        )

        def identity_paired_reflectors() -> nn.ParameterList:
            vectors: list[nn.Parameter] = []
            for _ in range(self.reflectors_per_side // 2):
                vector = torch.randn(self.width, generator=generator)
                vector = vector / torch.linalg.vector_norm(vector)
                vectors.append(nn.Parameter(vector.clone()))
                vectors.append(nn.Parameter(vector.clone()))
            return nn.ParameterList(vectors)

        self.input_reflectors = identity_paired_reflectors()
        self.output_reflectors = identity_paired_reflectors()
        self.hidden_gain = nn.Parameter(torch.ones(self.atom_width))
        self.output_gain = nn.Parameter(torch.ones(self.width))
        self.dropout = nn.Dropout(config.dropout)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    @property
    def shared_bank(self) -> SharedHouseholderAtomBank:
        return object.__getattribute__(self, "_shared_bank")

    def _apply_householders(
        self,
        values: torch.Tensor,
        reflectors,
    ) -> torch.Tensor:
        for vector in reflectors:
            vector = vector.to(dtype=values.dtype)
            denominator = vector.square().sum().clamp_min(
                self.householder_epsilon
            )
            projection = torch.matmul(values, vector).unsqueeze(-1)
            values = values - (2.0 * projection / denominator) * vector
        return values

    def folded_input_weight(self) -> torch.Tensor:
        # If row activations use x H0 H1 ... then the equivalent linear weight
        # is A ... H1 H0, hence the reversed reflector order here.
        return self._apply_householders(
            self.shared_bank.input_weight,
            reversed(self.input_reflectors),
        )

    def folded_output_weight(self) -> torch.Tensor:
        # Applying output reflectors after B is equivalent to left-folding the
        # reverse product into B; transposing exposes the row-wise operation.
        return self._apply_householders(
            self.shared_bank.output_weight.transpose(0, 1),
            self.output_reflectors,
        ).transpose(0, 1)

    def activation_space_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unfolded reference formula for values and gradient equivalence."""
        transformed = self._apply_householders(x, self.input_reflectors)
        hidden = F.gelu(F.linear(transformed, self.shared_bank.input_weight))
        hidden = hidden * self.hidden_gain.to(dtype=hidden.dtype)
        output = F.linear(hidden, self.shared_bank.output_weight)
        output = self._apply_householders(output, self.output_reflectors)
        return output * self.output_gain.to(dtype=output.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(F.linear(x, self.folded_input_weight()))
        hidden = hidden * self.hidden_gain.to(dtype=hidden.dtype)
        output = F.linear(hidden, self.folded_output_weight())
        output = output * self.output_gain.to(dtype=output.dtype)
        return self.dropout(output)

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        return None

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return None

    def prepare_charted_cfc_cache(self) -> None:
        return None

    def prepare_charted_cproj_cache(self) -> None:
        return None

    def flush_charted_cfc_cache(self) -> None:
        return None

    def flush_charted_cproj_cache(self) -> None:
        return None

    def suspend_charted_cfc_cache(self) -> None:
        return None

    def suspend_charted_cproj_cache(self) -> None:
        return None

    def restore_charted_cfc_cache(self, _value: None) -> None:
        return None

    def restore_charted_cproj_cache(self, _value: None) -> None:
        return None


class SharedSpectralPhaseAtomBank(nn.Module):
    """One shared atom bank moved by layer-private FFT phase frames."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        width = int(config.n_embd)
        atom_width = int(config.compact_native_mlp_shared_width)
        if width == 768 and atom_width != 353:
            raise ValueError(
                "the frozen spectral-phase frame gate requires width 353"
            )
        if width < 4 or width % 2:
            raise ValueError("spectral-phase width must be even and at least four")
        if atom_width <= 0:
            raise ValueError("spectral-phase atom width must be positive")
        self.width = width
        self.atom_width = atom_width
        self.phase_bins = width // 2 - 1
        self.input_weight = nn.Parameter(torch.empty(atom_width, width))
        self.output_weight = nn.Parameter(torch.empty(width, atom_width))
        nn.init.normal_(self.input_weight, mean=0.0, std=0.02)
        output_std = (
            0.02
            / math.sqrt(2 * config.n_layer)
            * math.sqrt((4 * width) / atom_width)
        )
        nn.init.normal_(self.output_weight, mean=0.0, std=output_std)
        self.output_init_std = output_std


class SpectralPhaseFrameSharedAtomMLP(nn.Module):
    """Shared nonlinear atoms with full-rank, basis-free spectral frames."""

    def __init__(
        self,
        config: GPTConfig,
        layer_id: int,
        shared_bank: SharedSpectralPhaseAtomBank,
    ) -> None:
        super().__init__()
        if shared_bank.width != int(config.n_embd):
            raise ValueError("shared spectral-phase width does not match config")
        if shared_bank.atom_width != int(config.compact_native_mlp_shared_width):
            raise ValueError("shared spectral-phase atom width does not match config")
        self.width = int(shared_bank.width)
        self.atom_width = int(shared_bank.atom_width)
        self.phase_bins = int(shared_bank.phase_bins)
        object.__setattr__(self, "_shared_bank", shared_bank)

        # This changes the fixed Fourier gauge for each layer without adding
        # any persistent state.  Buffers move with the module but are omitted
        # from state_dict/checkpoints by construction.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(config.compact_native_mlp_seed) + int(layer_id) * 1009
        )
        permutation = torch.randperm(self.width, generator=generator)
        inverse_permutation = torch.argsort(permutation)
        sign = torch.randint(0, 2, (self.width,), generator=generator)
        sign = sign.to(torch.float32).mul_(2.0).sub_(1.0)
        self.register_buffer("frame_permutation", permutation, persistent=False)
        self.register_buffer(
            "frame_inverse_permutation", inverse_permutation, persistent=False
        )
        self.register_buffer("frame_sign", sign, persistent=False)

        self.input_phase = nn.Parameter(torch.zeros(self.phase_bins))
        self.output_phase = nn.Parameter(torch.zeros(self.phase_bins))
        self.hidden_gain = nn.Parameter(torch.ones(self.atom_width))
        self.output_gain = nn.Parameter(torch.ones(self.width))
        self.dropout = nn.Dropout(config.dropout)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    @property
    def shared_bank(self) -> SharedSpectralPhaseAtomBank:
        return object.__getattribute__(self, "_shared_bank")

    def _signed_permute(self, values: torch.Tensor) -> torch.Tensor:
        sign = self.frame_sign.to(dtype=values.dtype)
        return values.index_select(-1, self.frame_permutation) * sign

    def _inverse_signed_permute(self, values: torch.Tensor) -> torch.Tensor:
        inverse = self.frame_inverse_permutation
        sign = self.frame_sign.index_select(0, inverse).to(dtype=values.dtype)
        return values.index_select(-1, inverse) * sign

    def _apply_spectral_frame(
        self,
        values: torch.Tensor,
        phase: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        if values.shape[-1] != self.width:
            raise ValueError("spectral frame input width mismatch")
        if phase.numel() != self.phase_bins:
            raise ValueError("spectral frame phase count mismatch")
        original_dtype = values.dtype
        # Weight folding is FP32 in production.  Promote tiny CPU tests too so
        # torch.fft never relies on reduced-precision backend support.
        transformed = self._signed_permute(values.float())
        phase_values = phase.float()
        if inverse:
            phase_values = -phase_values
        interior = torch.complex(torch.cos(phase_values), torch.sin(phase_values))
        edge = torch.ones(1, dtype=interior.dtype, device=interior.device)
        multiplier = torch.cat((edge, interior, edge), dim=0)
        spectrum = torch.fft.rfft(transformed, n=self.width, dim=-1)
        transformed = torch.fft.irfft(
            spectrum * multiplier,
            n=self.width,
            dim=-1,
        )
        transformed = self._inverse_signed_permute(transformed)
        return transformed.to(dtype=original_dtype)

    def folded_input_weight(self) -> torch.Tensor:
        # A P is obtained by applying P^{-1} to every row of A.
        return self._apply_spectral_frame(
            self.shared_bank.input_weight,
            self.input_phase,
            inverse=True,
        )

    def folded_output_weight(self) -> torch.Tensor:
        # Q B is obtained by applying Q to every column of B.
        return self._apply_spectral_frame(
            self.shared_bank.output_weight.transpose(0, 1),
            self.output_phase,
        ).transpose(0, 1)

    def activation_space_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unfolded reference formula for values and gradient equivalence."""
        transformed = self._apply_spectral_frame(x, self.input_phase)
        hidden = F.gelu(F.linear(transformed, self.shared_bank.input_weight))
        hidden = hidden * self.hidden_gain.to(dtype=hidden.dtype)
        output = F.linear(hidden, self.shared_bank.output_weight)
        output = self._apply_spectral_frame(output, self.output_phase)
        return output * self.output_gain.to(dtype=output.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(F.linear(x, self.folded_input_weight()))
        hidden = hidden * self.hidden_gain.to(dtype=hidden.dtype)
        output = F.linear(hidden, self.folded_output_weight())
        output = output * self.output_gain.to(dtype=output.dtype)
        return self.dropout(output)

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        return None

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return None

    def prepare_charted_cfc_cache(self) -> None:
        return None

    def prepare_charted_cproj_cache(self) -> None:
        return None

    def flush_charted_cfc_cache(self) -> None:
        return None

    def flush_charted_cproj_cache(self) -> None:
        return None

    def suspend_charted_cfc_cache(self) -> None:
        return None

    def suspend_charted_cproj_cache(self) -> None:
        return None

    def restore_charted_cfc_cache(self, _value: None) -> None:
        return None

    def restore_charted_cproj_cache(self, _value: None) -> None:
        return None


class SharedTensorProductRidgeField(nn.Module):
    """Shared factor and write map for the semiprivate tensor-product MLP."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        rank = int(config.compact_native_mlp_factor_rank)
        if rank != 20:
            raise ValueError("the frozen tensor-product ridge gate requires rank 20")
        self.rank = rank
        self.width = int(config.n_embd)
        self.feature_width = rank * rank + 2 * rank
        self.shared_input_weight = nn.Parameter(torch.empty(rank, self.width))
        self.write_weight = nn.Parameter(
            torch.empty(self.width, self.feature_width)
        )
        nn.init.normal_(self.shared_input_weight, mean=0.0, std=0.02)

        # Match the second moment of the ordinary 4d GELU MLP at GPT init.
        # The quadrature is deterministic and creates no persistent state.
        pre_std = 0.02 * math.sqrt(self.width)
        grid = torch.linspace(-8.0, 8.0, 16_385, dtype=torch.float64)
        density = torch.exp(-0.5 * grid.square()) / math.sqrt(2.0 * math.pi)
        gelu_second_moment = float(
            torch.trapezoid(F.gelu(pre_std * grid).square() * density, grid)
        )
        feature_second_moment = (
            2 * rank * gelu_second_moment
            + rank * rank * gelu_second_moment * gelu_second_moment
        ) / self.feature_width
        dense_output_std = 0.02 / math.sqrt(2 * config.n_layer)
        write_std = dense_output_std * math.sqrt(
            (4 * self.width * gelu_second_moment)
            / (self.feature_width * feature_second_moment)
        )
        nn.init.normal_(self.write_weight, mean=0.0, std=write_std)
        self.gelu_second_moment = gelu_second_moment
        self.feature_second_moment = feature_second_moment
        self.write_init_std = write_std

    def shared_features(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(F.linear(x, self.shared_input_weight))

    def write(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.write_weight)


class SemiprivateTensorProductRidgeMLP(nn.Module):
    """Layer-private ridge planes with a shared multiplicative write field."""

    def __init__(
        self,
        config: GPTConfig,
        layer_id: int,
        shared_field: SharedTensorProductRidgeField,
    ) -> None:
        super().__init__()
        if shared_field.rank != int(config.compact_native_mlp_factor_rank):
            raise ValueError("shared tensor-product rank does not match config")
        self.width = int(config.n_embd)
        self.rank = int(shared_field.rank)
        self.feature_width = int(shared_field.feature_width)
        object.__setattr__(self, "_shared_field", shared_field)
        self.private_input_weight = nn.Parameter(
            torch.empty(self.rank, self.width)
        )
        nn.init.normal_(self.private_input_weight, mean=0.0, std=0.02)
        self.feature_gain = nn.Parameter(torch.ones(self.feature_width))
        self.output_gain = nn.Parameter(torch.ones(self.width))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(config.compact_native_mlp_seed) + int(layer_id) * 1009
        )
        permutation = torch.randperm(self.width, generator=generator)
        inverse_permutation = torch.argsort(permutation)
        sign = (
            torch.randint(
                0,
                2,
                (self.width,),
                generator=generator,
                dtype=torch.int64,
            )
            .mul(2)
            .sub(1)
            .float()
        )
        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer(
            "inverse_permutation",
            inverse_permutation,
            persistent=False,
        )
        self.register_buffer("sign", sign, persistent=False)
        self.dropout = nn.Dropout(config.dropout)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    @property
    def shared_field(self) -> SharedTensorProductRidgeField:
        return object.__getattribute__(self, "_shared_field")

    def tensor_product_features(self, x: torch.Tensor) -> torch.Tensor:
        private = F.gelu(F.linear(x, self.private_input_weight))
        shared = self.shared_field.shared_features(x)
        products = (private.unsqueeze(-1) * shared.unsqueeze(-2)).flatten(-2)
        return torch.cat((private, shared, products), dim=-1)

    def _fold_input_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Move the fixed signed input permutation from activations to a weight."""
        inverse = self.inverse_permutation
        sign = self.sign.to(dtype=weight.dtype)
        return weight.index_select(-1, inverse) * sign.index_select(0, inverse)

    def _fold_write_weight(self) -> torch.Tensor:
        """Move the fixed signed output inverse-permutation into the write map."""
        inverse = self.inverse_permutation
        sign = self.sign.to(dtype=self.shared_field.write_weight.dtype)
        row_gain = self.output_gain.to(dtype=sign.dtype) * sign
        return self.shared_field.write_weight.index_select(0, inverse) * (
            row_gain.index_select(0, inverse).unsqueeze(-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Algebraically fold the procedural conjugation into the much smaller
        # weights.  This is exactly equivalent to gathering every token before
        # and after the MLP, while avoiding two activation-sized gathers.
        private = F.gelu(F.linear(x, self._fold_input_weight(self.private_input_weight)))
        shared = F.gelu(
            F.linear(
                x,
                self._fold_input_weight(self.shared_field.shared_input_weight),
            )
        )
        products = (private.unsqueeze(-1) * shared.unsqueeze(-2)).flatten(-2)
        features = torch.cat((private, shared, products), dim=-1)
        features = features * self.feature_gain.to(dtype=x.dtype)
        output = F.linear(features, self._fold_write_weight())
        return self.dropout(output)

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        return None

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return None

    def prepare_charted_cfc_cache(self) -> None:
        return None

    def prepare_charted_cproj_cache(self) -> None:
        return None

    def flush_charted_cfc_cache(self) -> None:
        return None

    def flush_charted_cproj_cache(self) -> None:
        return None

    def suspend_charted_cfc_cache(self) -> None:
        return None

    def suspend_charted_cproj_cache(self) -> None:
        return None

    def restore_charted_cfc_cache(self, _value: None) -> None:
        return None

    def restore_charted_cproj_cache(self, _value: None) -> None:
        return None


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        compact_rank = int(config.compact_mlp_gradient_seeded_rank)
        if compact_rank < 0:
            raise ValueError("compact_mlp_gradient_seeded_rank must be non-negative")
        grouped_targets = [target for target in MLP_C_FC_GROUP_TARGETS if target in config.block_fht_targets]
        if len(grouped_targets) > 1:
            raise ValueError("Use exactly one grouped mlp.c_fc target per run")
        if grouped_targets and "mlp.c_fc" in config.block_fht_targets:
            raise ValueError("Use either plain mlp.c_fc or grouped mlp.c_fc, not both")
        if compact_rank and grouped_targets:
            raise ValueError(
                "gradient-seeded compact MLP requires plain mlp.c_fc"
            )
        if compact_rank:
            self.c_fc = GradientSeededLowRankLinear(
                config.n_embd,
                4 * config.n_embd,
                bias=config.bias,
                rank=compact_rank,
                scale=float(config.compact_mlp_gradient_seeded_scale),
                target_name="mlp.c_fc",
            )
        elif grouped_targets:
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
        pregelu_rotation_stages = int(
            config.block_fht_mlp_pregelu_block_rotation_stages
        )
        if pregelu_rotation_stages < 0:
            raise ValueError(
                "block_fht_mlp_pregelu_block_rotation_stages must be "
                "non-negative"
            )
        if pregelu_rotation_stages and not isinstance(
            self.c_fc, (nn.Linear, BlockFHTLinear)
        ):
            raise ValueError(
                "pre-GELU block rotation requires a plain mlp.c_fc linear"
            )
        self.pregelu_block_rotation = (
            LearnedFHTBlockOrthogonalOutputMix(
                features=4 * config.n_embd,
                stages=pregelu_rotation_stages,
                rotation_block_size=int(
                    config.block_fht_mlp_pregelu_block_rotation_size
                ),
                basis_block_size=int(
                    config.block_fht_mlp_pregelu_block_rotation_basis_size
                ),
                seed=(
                    int(config.block_fht_mlp_pregelu_block_rotation_seed)
                    + layer_id * 64
                ),
                coordinate_scale=float(
                    config.block_fht_mlp_pregelu_block_rotation_coordinate_scale
                ),
            )
            if pregelu_rotation_stages
            else None
        )
        self.pregelu_cache_retain_graph = bool(
            config.block_fht_mlp_pregelu_cache_retain_graph
        )
        if (
            self.pregelu_cache_retain_graph
            and self.pregelu_block_rotation is None
        ):
            raise ValueError(
                "pre-GELU cache graph retention requires a pre-GELU frame"
            )
        self._cached_charted_cfc_weight: torch.Tensor | None = None
        self._cached_charted_cfc_graph_weight: torch.Tensor | None = None
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
        if compact_rank and structured_proj_count:
            raise ValueError(
                "gradient-seeded compact MLP requires plain mlp.c_proj"
            )
        if structured_proj_count and "mlp.c_proj" in config.block_fht_targets:
            raise ValueError("Use either plain mlp.c_proj or grouped mlp.c_proj, not both")
        muon_matched_cproj = bool(
            config.block_fht_mlp_cproj_muon_matched_givens
        )
        if muon_matched_cproj and structured_proj_count:
            raise ValueError(
                "Muon-matched Givens c_proj requires the plain "
                "mlp.c_proj target"
            )
        if (
            muon_matched_cproj
            and "mlp.c_proj" not in config.block_fht_targets
        ):
            raise ValueError(
                "Muon-matched Givens c_proj requires mlp.c_proj in "
                "block_fht_targets"
            )
        if compact_rank and muon_matched_cproj:
            raise ValueError(
                "gradient-seeded compact MLP is incompatible with Muon-matched c_proj"
            )
        if compact_rank:
            self.c_proj = GradientSeededLowRankLinear(
                4 * config.n_embd,
                config.n_embd,
                bias=config.bias,
                rank=compact_rank,
                scale=float(config.compact_mlp_gradient_seeded_scale),
                target_name="mlp.c_proj",
            )
        elif muon_matched_cproj:
            self.c_proj = MuonMatchedGivensLinear(
                4 * config.n_embd,
                config.n_embd,
                bias=config.bias,
                stages=int(
                    config
                    .block_fht_mlp_cproj_muon_matched_givens_stages
                ),
                residual_stages=int(
                    config
                    .block_fht_mlp_cproj_muon_matched_givens_residual_stages
                ),
                neighbors=int(
                    config
                    .block_fht_mlp_cproj_muon_matched_givens_neighbors
                ),
                refresh_interval=int(
                    config
                    .block_fht_mlp_cproj_muon_matched_givens_refresh_interval
                ),
                fast_fresh_matching=bool(
                    config
                    .block_fht_mlp_cproj_muon_matched_givens_fast_fresh
                ),
                matching_seed=(
                    int(
                        config
                        .block_fht_mlp_cproj_muon_matched_givens_seed
                    )
                    + layer_id * 64
                ),
                weight_std=(
                    0.02 / math.sqrt(2 * config.n_layer)
                ),
                layer_id=layer_id,
            )
        elif grouped_proj_targets:
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
            if (
                config.block_fht_cproj_spectral_resid_full_core
                and config.block_fht_cproj_spectral_resid_muon_matrix
            ):
                raise ValueError(
                    "block_fht_cproj_spectral_resid_full_core and "
                    "block_fht_cproj_spectral_resid_muon_matrix are mutually exclusive"
                )
            if config.block_fht_cproj_spectral_resid_full_core:
                spectral_shape = (spectral_rank, spectral_rank)
            elif config.block_fht_cproj_spectral_resid_muon_matrix:
                spectral_shape = (1, spectral_rank)
            else:
                spectral_shape = (spectral_rank,)
            self.cproj_spectral_resid_diag = nn.Parameter(torch.zeros(spectral_shape))
            self.cproj_spectral_resid_scale = nn.Parameter(torch.tensor(float(config.block_fht_cproj_spectral_resid_scale_init)))
            spectral_seed = int(config.block_fht_cproj_spectral_resid_seed)
            self.cproj_spectral_resid_in_mix = FixedFHTMix(4 * config.n_embd, spectral_seed + layer_id * 64 + 129)
            self.cproj_spectral_resid_out_mix = FixedFHTMix(config.n_embd, spectral_seed + layer_id * 64 + 130)
            self.register_buffer(
                "cproj_spectral_resid_in_basis",
                self.cproj_spectral_resid_in_mix.basis_columns(spectral_rank),
                persistent=False,
            )
            self.register_buffer(
                "cproj_spectral_resid_out_basis",
                self.cproj_spectral_resid_out_mix.basis_columns(spectral_rank),
                persistent=False,
            )
        else:
            self.cproj_spectral_resid_diag = None
            self.cproj_spectral_resid_scale = None
            self.cproj_spectral_resid_in_mix = None
            self.cproj_spectral_resid_out_mix = None
            self.cproj_spectral_resid_in_basis = None
            self.cproj_spectral_resid_out_basis = None
        self.postgelu_std_target = float(config.block_fht_ffn_postgelu_std_target)
        self.shared_hidden_log_gain = (
            nn.Parameter(torch.zeros(4 * config.n_embd))
            if config.block_fht_mlp_shared_hidden_gain
            else None
        )
        self.shared_hidden_gain_scale = float(config.block_fht_mlp_shared_hidden_gain_scale)
        if not math.isfinite(self.shared_hidden_gain_scale) or self.shared_hidden_gain_scale <= 0.0:
            raise ValueError("block_fht_mlp_shared_hidden_gain_scale must be positive and finite")
        if (
            config.block_fht_mlp_activation_chart
            and self.shared_hidden_log_gain is not None
        ):
            raise ValueError(
                "activation chart and legacy shared hidden gain are "
                "mutually exclusive"
            )
        self.activation_chart_channel_log_gain = (
            nn.Parameter(torch.zeros(4 * config.n_embd))
            if config.block_fht_mlp_activation_chart
            else None
        )
        self.activation_chart_common_log_gain = (
            nn.Parameter(torch.zeros(()))
            if config.block_fht_mlp_activation_chart
            else None
        )
        self.activation_chart_gauge_log_gain = (
            nn.Parameter(torch.zeros(()))
            if config.block_fht_mlp_activation_chart
            else None
        )
        self.activation_chart_channel_scale = float(
            config.block_fht_mlp_activation_chart_channel_scale
        )
        self.activation_chart_common_scale = float(
            config.block_fht_mlp_activation_chart_common_scale
        )
        self.activation_chart_gauge_scale = float(
            config.block_fht_mlp_activation_chart_gauge_scale
        )
        for name, scale in (
            ("channel", self.activation_chart_channel_scale),
            ("common", self.activation_chart_common_scale),
            ("gauge", self.activation_chart_gauge_scale),
        ):
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "block_fht_mlp_activation_chart_"
                    f"{name}_scale must be positive and finite"
                )
        rotation_stages = int(config.block_fht_mlp_output_rotation_stages)
        if rotation_stages < 0:
            raise ValueError("block_fht_mlp_output_rotation_stages must be non-negative")
        self.output_rotation = (
            LearnedGivensOutputMix(
                config.n_embd,
                rotation_stages,
                int(config.block_fht_mlp_output_rotation_seed) + layer_id * 64,
            )
            if rotation_stages
            else None
        )
        block_rotation_stages = int(
            config.block_fht_mlp_output_block_rotation_stages
        )
        if block_rotation_stages < 0:
            raise ValueError(
                "block_fht_mlp_output_block_rotation_stages "
                "must be non-negative"
            )
        if rotation_stages and block_rotation_stages:
            raise ValueError(
                "pairwise and block-orthogonal output rotations are "
                "mutually exclusive"
            )
        if block_rotation_stages and not isinstance(
            self.c_proj, (nn.Linear, BlockFHTLinear)
        ):
            raise ValueError(
                "block-orthogonal output rotation requires a plain "
                "mlp.c_proj linear"
            )
        incompatible_cproj_addon = any(
            value is not None
            for value in (
                self.cproj_lowrank_left,
                self.cproj_lowrank_right,
                self.cproj_tied_cfc_scale,
                self.cproj_quarter_diag_weight,
                self.cproj_spectral_resid_diag,
            )
        )
        hidden_block_rotation_stages = int(
            config.block_fht_mlp_hidden_block_rotation_stages
        )
        if hidden_block_rotation_stages < 0:
            raise ValueError(
                "block_fht_mlp_hidden_block_rotation_stages must be "
                "non-negative"
            )
        if hidden_block_rotation_stages and not isinstance(
            self.c_proj, (nn.Linear, BlockFHTLinear)
        ):
            raise ValueError(
                "block-orthogonal hidden rotation requires a plain "
                "mlp.c_proj linear"
            )
        if hidden_block_rotation_stages and incompatible_cproj_addon:
            raise ValueError(
                "block-orthogonal hidden rotation cannot be combined with "
                "a separate c_proj residual add-on"
            )
        self.hidden_block_rotation = (
            LearnedFHTBlockOrthogonalOutputMix(
                features=4 * config.n_embd,
                stages=hidden_block_rotation_stages,
                rotation_block_size=int(
                    config.block_fht_mlp_hidden_block_rotation_size
                ),
                basis_block_size=int(
                    config.block_fht_mlp_hidden_block_rotation_basis_size
                ),
                seed=(
                    int(config.block_fht_mlp_hidden_block_rotation_seed)
                    + layer_id * 64
                ),
                coordinate_scale=float(
                    config.block_fht_mlp_hidden_block_rotation_coordinate_scale
                ),
            )
            if hidden_block_rotation_stages
            else None
        )
        self.hidden_gain_scale = float(
            config.block_fht_mlp_hidden_gain_scale
        )
        if (
            not math.isfinite(self.hidden_gain_scale)
            or self.hidden_gain_scale <= 0.0
        ):
            raise ValueError(
                "block_fht_mlp_hidden_gain_scale must be positive and finite"
            )
        hidden_log_gain_init = float(
            config.block_fht_mlp_hidden_log_gain_init
        )
        if not math.isfinite(hidden_log_gain_init):
            raise ValueError(
                "block_fht_mlp_hidden_log_gain_init must be finite"
            )
        if (
            config.block_fht_mlp_hidden_gain
            and self.activation_chart_channel_log_gain is not None
        ):
            raise ValueError(
                "hidden gain and activation chart channel gain are "
                "mutually exclusive"
            )
        self.hidden_log_gain = (
            nn.Parameter(
                torch.full(
                    (4 * config.n_embd,),
                    hidden_log_gain_init / self.hidden_gain_scale,
                )
            )
            if config.block_fht_mlp_hidden_gain
            else None
        )
        if self.hidden_log_gain is not None and not isinstance(
            self.c_proj, (nn.Linear, BlockFHTLinear)
        ):
            raise ValueError(
                "hidden gain requires a plain mlp.c_proj linear"
            )
        if self.hidden_log_gain is not None and incompatible_cproj_addon:
            raise ValueError(
                "hidden gain cannot be combined with a separate c_proj "
                "residual add-on"
            )
        if block_rotation_stages and incompatible_cproj_addon:
            raise ValueError(
                "block-orthogonal output rotation cannot be combined with "
                "a separate c_proj residual add-on"
            )
        self.output_block_rotation = (
            LearnedFHTBlockOrthogonalOutputMix(
                features=config.n_embd,
                stages=block_rotation_stages,
                rotation_block_size=int(
                    config.block_fht_mlp_output_block_rotation_size
                ),
                basis_block_size=int(
                    config.block_fht_mlp_output_block_rotation_basis_size
                ),
                seed=(
                    int(config.block_fht_mlp_output_rotation_seed)
                    + layer_id * 64
                ),
                coordinate_scale=float(
                    config.block_fht_mlp_output_block_rotation_coordinate_scale
                ),
            )
            if block_rotation_stages
            else None
        )
        self.residual_output_gain_scale = float(
            config.block_fht_mlp_residual_output_gain_scale
        )
        if (
            not math.isfinite(self.residual_output_gain_scale)
            or self.residual_output_gain_scale <= 0.0
        ):
            raise ValueError(
                "block_fht_mlp_residual_output_gain_scale must be "
                "positive and finite"
            )
        residual_output_log_gain_init = float(
            config.block_fht_mlp_residual_output_log_gain_init
        )
        if not math.isfinite(residual_output_log_gain_init):
            raise ValueError(
                "block_fht_mlp_residual_output_log_gain_init must be finite"
            )
        self.residual_output_log_gain = (
            nn.Parameter(
                torch.full(
                    (config.n_embd,),
                    (
                        residual_output_log_gain_init
                        / self.residual_output_gain_scale
                    ),
                )
            )
            if config.block_fht_mlp_residual_output_gain
            else None
        )
        if self.residual_output_log_gain is not None and not isinstance(
            self.c_proj, (nn.Linear, BlockFHTLinear)
        ):
            raise ValueError(
                "residual output gain requires a plain mlp.c_proj linear"
            )
        if self.residual_output_log_gain is not None and incompatible_cproj_addon:
            raise ValueError(
                "residual output gain cannot be combined with a separate "
                "c_proj residual add-on"
            )
        self.residual_conditioned_output_gate_scale = float(
            config.block_fht_mlp_residual_conditioned_output_gate_scale
        )
        if (
            not math.isfinite(self.residual_conditioned_output_gate_scale)
            or self.residual_conditioned_output_gate_scale <= 0.0
        ):
            raise ValueError(
                "block_fht_mlp_residual_conditioned_output_gate_scale "
                "must be positive and finite"
            )
        conditioned_gate_layers = tuple(
            int(layer)
            for layer in config.block_fht_mlp_residual_conditioned_output_gate_layers
        )
        if any(
            layer < 0 or layer >= config.n_layer
            for layer in conditioned_gate_layers
        ):
            raise ValueError(
                "block_fht_mlp_residual_conditioned_output_gate_layers "
                "contains an invalid layer"
            )
        self.conditioned_output_gate_source = str(
            config.block_fht_mlp_conditioned_output_gate_source
        )
        if self.conditioned_output_gate_source not in (
            "residual",
            "postgelu",
        ):
            raise ValueError(
                "block_fht_mlp_conditioned_output_gate_source must be "
                "'residual' or 'postgelu'"
            )
        self.conditioned_output_gate_rms_epsilon = float(
            config.block_fht_mlp_conditioned_output_gate_rms_epsilon
        )
        if (
            not math.isfinite(self.conditioned_output_gate_rms_epsilon)
            or self.conditioned_output_gate_rms_epsilon <= 0.0
        ):
            raise ValueError(
                "block_fht_mlp_conditioned_output_gate_rms_epsilon must "
                "be positive and finite"
            )
        conditioned_gate_enabled = (
            config.block_fht_mlp_residual_conditioned_output_gate
            and (
                not conditioned_gate_layers
                or layer_id in conditioned_gate_layers
            )
        )
        self.residual_conditioned_output_slope = (
            nn.Parameter(torch.zeros(config.n_embd))
            if conditioned_gate_enabled
            else None
        )
        self.residual_conditioned_output_bias = (
            nn.Parameter(torch.zeros(config.n_embd))
            if (
                conditioned_gate_enabled
                and config.block_fht_mlp_residual_conditioned_output_gate_bias
            )
            else None
        )
        self.residual_conditioned_output_gate_fixed_basis = bool(
            conditioned_gate_enabled
            and config.block_fht_mlp_residual_conditioned_output_gate_fixed_basis
        )
        if (
            config.block_fht_mlp_residual_conditioned_output_gate_untied_bases
            and not config.block_fht_mlp_residual_conditioned_output_gate_fixed_basis
        ):
            raise ValueError(
                "block_fht_mlp_residual_conditioned_output_gate_"
                "untied_bases requires fixed_basis"
            )
        self.residual_conditioned_output_gate_untied_bases = bool(
            self.residual_conditioned_output_gate_fixed_basis
            and config.block_fht_mlp_residual_conditioned_output_gate_untied_bases
        )
        if (
            conditioned_gate_enabled
            and self.conditioned_output_gate_source == "postgelu"
            and (
                not self.residual_conditioned_output_gate_fixed_basis
                or not self.residual_conditioned_output_gate_untied_bases
            )
        ):
            raise ValueError(
                "postgelu conditioned output gate requires fixed_basis "
                "and untied_bases"
            )
        self.residual_conditioned_output_gate_basis_block_size = int(
            config.block_fht_mlp_residual_conditioned_output_gate_basis_block_size
        )
        if self.residual_conditioned_output_gate_fixed_basis:
            basis_block_size = (
                self.residual_conditioned_output_gate_basis_block_size
            )
            if (
                basis_block_size <= 0
                or basis_block_size & (basis_block_size - 1)
                or config.n_embd % basis_block_size
            ):
                raise ValueError(
                    "block_fht_mlp_residual_conditioned_output_gate_"
                    "basis_block_size must be a power of two dividing n_embd"
                )
            def make_basis_buffers(
                seed: int,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed) + layer_id * 64)
                selected_permutation = torch.randperm(
                    config.n_embd,
                    generator=generator,
                    device="cpu",
                )
                selected_signs = (
                    torch.randint(
                        0,
                        2,
                        (config.n_embd,),
                        generator=generator,
                        dtype=torch.float32,
                        device="cpu",
                    )
                    * 2.0
                    - 1.0
                )
                return (
                    selected_permutation,
                    torch.argsort(selected_permutation),
                    selected_signs,
                )

            (
                permutation,
                inverse_permutation,
                signs,
            ) = make_basis_buffers(
                config.block_fht_mlp_residual_conditioned_output_gate_basis_seed
            )
            if self.residual_conditioned_output_gate_untied_bases:
                (
                    update_permutation,
                    update_inverse_permutation,
                    update_signs,
                ) = make_basis_buffers(
                    config.block_fht_mlp_residual_conditioned_output_gate_update_basis_seed
                )
                (
                    output_permutation,
                    output_inverse_permutation,
                    output_signs,
                ) = make_basis_buffers(
                    config.block_fht_mlp_residual_conditioned_output_gate_output_basis_seed
                )
            else:
                update_permutation = None
                update_inverse_permutation = None
                update_signs = None
                output_permutation = None
                output_inverse_permutation = None
                output_signs = None
            hadamard = normalized_fht_last_dim(
                torch.eye(basis_block_size, dtype=torch.float32)
            )
        else:
            permutation = None
            inverse_permutation = None
            signs = None
            update_permutation = None
            update_inverse_permutation = None
            update_signs = None
            output_permutation = None
            output_inverse_permutation = None
            output_signs = None
            hadamard = None
        if (
            conditioned_gate_enabled
            and self.conditioned_output_gate_source == "postgelu"
        ):
            projection_generator = torch.Generator(device="cpu")
            projection_generator.manual_seed(
                int(
                    config.block_fht_mlp_conditioned_output_gate_projection_seed
                )
                + layer_id * 64
            )
            conditioned_output_projection_signs = (
                torch.randint(
                    0,
                    2,
                    (4, config.n_embd),
                    generator=projection_generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                * 2.0
                - 1.0
            )
        else:
            conditioned_output_projection_signs = None
        self.register_buffer(
            "residual_conditioned_output_permutation",
            permutation,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_inverse_permutation",
            inverse_permutation,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_signs",
            signs,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_update_permutation",
            update_permutation,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_update_inverse_permutation",
            update_inverse_permutation,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_update_signs",
            update_signs,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_output_permutation",
            output_permutation,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_output_inverse_permutation",
            output_inverse_permutation,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_output_signs",
            output_signs,
            persistent=True,
        )
        self.register_buffer(
            "residual_conditioned_output_hadamard",
            hadamard,
            persistent=True,
        )
        self.register_buffer(
            "conditioned_output_projection_signs",
            conditioned_output_projection_signs,
            persistent=True,
        )
        self.postgelu_hidden_self_gate_scale = float(
            config.block_fht_mlp_postgelu_hidden_self_gate_scale
        )
        if (
            not math.isfinite(self.postgelu_hidden_self_gate_scale)
            or self.postgelu_hidden_self_gate_scale <= 0.0
        ):
            raise ValueError(
                "block_fht_mlp_postgelu_hidden_self_gate_scale must be "
                "positive and finite"
            )
        postgelu_hidden_self_gate_layers = tuple(
            int(layer)
            for layer in config.block_fht_mlp_postgelu_hidden_self_gate_layers
        )
        if any(
            layer < 0 or layer >= config.n_layer
            for layer in postgelu_hidden_self_gate_layers
        ):
            raise ValueError(
                "block_fht_mlp_postgelu_hidden_self_gate_layers contains "
                "an invalid layer"
            )
        postgelu_hidden_self_gate_enabled = bool(
            config.block_fht_mlp_postgelu_hidden_self_gate
            and (
                not postgelu_hidden_self_gate_layers
                or layer_id in postgelu_hidden_self_gate_layers
            )
        )
        self.postgelu_hidden_self_gate_heads = int(
            config.block_fht_mlp_postgelu_hidden_self_gate_heads
        )
        if self.postgelu_hidden_self_gate_heads < 1:
            raise ValueError(
                "block_fht_mlp_postgelu_hidden_self_gate_heads must be "
                "at least one"
            )
        self.postgelu_hidden_self_gate_head_seed_stride = int(
            config.block_fht_mlp_postgelu_hidden_self_gate_head_seed_stride
        )
        if self.postgelu_hidden_self_gate_head_seed_stride <= 0:
            raise ValueError(
                "block_fht_mlp_postgelu_hidden_self_gate_head_seed_stride "
                "must be positive"
            )
        self.postgelu_hidden_self_slope = (
            nn.Parameter(
                torch.zeros(
                    (
                        self.postgelu_hidden_self_gate_heads,
                        4 * config.n_embd,
                    )
                    if self.postgelu_hidden_self_gate_heads > 1
                    else (4 * config.n_embd,)
                )
            )
            if postgelu_hidden_self_gate_enabled
            else None
        )
        self.postgelu_hidden_self_gate_basis_block_size = int(
            config.block_fht_mlp_postgelu_hidden_self_gate_basis_block_size
        )
        self.postgelu_hidden_self_gate_rms_epsilon = float(
            config.block_fht_mlp_postgelu_hidden_self_gate_rms_epsilon
        )
        if (
            not math.isfinite(self.postgelu_hidden_self_gate_rms_epsilon)
            or self.postgelu_hidden_self_gate_rms_epsilon <= 0.0
        ):
            raise ValueError(
                "block_fht_mlp_postgelu_hidden_self_gate_rms_epsilon must "
                "be positive and finite"
            )

        def make_postgelu_hidden_basis_buffers(
            seed: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            width = 4 * config.n_embd
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + layer_id * 64)
            selected_permutation = torch.randperm(
                width,
                generator=generator,
                device="cpu",
            )
            selected_signs = (
                torch.randint(
                    0,
                    2,
                    (width,),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                * 2.0
                - 1.0
            )
            return (
                selected_permutation,
                torch.argsort(selected_permutation),
                selected_signs,
            )

        def make_postgelu_hidden_basis_stack(
            seed: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            head_buffers = [
                make_postgelu_hidden_basis_buffers(
                    int(seed)
                    + head
                    * self.postgelu_hidden_self_gate_head_seed_stride
                )
                for head in range(self.postgelu_hidden_self_gate_heads)
            ]
            if self.postgelu_hidden_self_gate_heads == 1:
                return head_buffers[0]
            return tuple(
                torch.stack(
                    [head_buffers[head][index] for head in range(
                        self.postgelu_hidden_self_gate_heads
                    )],
                    dim=0,
                )
                for index in range(3)
            )

        if postgelu_hidden_self_gate_enabled:
            basis_block_size = (
                self.postgelu_hidden_self_gate_basis_block_size
            )
            hidden_width = 4 * config.n_embd
            if (
                basis_block_size <= 0
                or basis_block_size & (basis_block_size - 1)
                or hidden_width % basis_block_size
            ):
                raise ValueError(
                    "block_fht_mlp_postgelu_hidden_self_gate_"
                    "basis_block_size must be a power of two dividing "
                    "4 * n_embd"
                )
            (
                postgelu_hidden_condition_permutation,
                postgelu_hidden_condition_inverse_permutation,
                postgelu_hidden_condition_signs,
            ) = make_postgelu_hidden_basis_stack(
                config.block_fht_mlp_postgelu_hidden_self_gate_condition_basis_seed
            )
            (
                postgelu_hidden_update_permutation,
                postgelu_hidden_update_inverse_permutation,
                postgelu_hidden_update_signs,
            ) = make_postgelu_hidden_basis_stack(
                config.block_fht_mlp_postgelu_hidden_self_gate_update_basis_seed
            )
            (
                postgelu_hidden_output_permutation,
                postgelu_hidden_output_inverse_permutation,
                postgelu_hidden_output_signs,
            ) = make_postgelu_hidden_basis_stack(
                config.block_fht_mlp_postgelu_hidden_self_gate_output_basis_seed
            )
            postgelu_hidden_hadamard = normalized_fht_last_dim(
                torch.eye(basis_block_size, dtype=torch.float32)
            )
        else:
            postgelu_hidden_condition_permutation = None
            postgelu_hidden_condition_inverse_permutation = None
            postgelu_hidden_condition_signs = None
            postgelu_hidden_update_permutation = None
            postgelu_hidden_update_inverse_permutation = None
            postgelu_hidden_update_signs = None
            postgelu_hidden_output_permutation = None
            postgelu_hidden_output_inverse_permutation = None
            postgelu_hidden_output_signs = None
            postgelu_hidden_hadamard = None
        self.register_buffer(
            "postgelu_hidden_condition_permutation",
            postgelu_hidden_condition_permutation,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_condition_inverse_permutation",
            postgelu_hidden_condition_inverse_permutation,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_condition_signs",
            postgelu_hidden_condition_signs,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_update_permutation",
            postgelu_hidden_update_permutation,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_update_inverse_permutation",
            postgelu_hidden_update_inverse_permutation,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_update_signs",
            postgelu_hidden_update_signs,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_output_permutation",
            postgelu_hidden_output_permutation,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_output_inverse_permutation",
            postgelu_hidden_output_inverse_permutation,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_output_signs",
            postgelu_hidden_output_signs,
            persistent=True,
        )
        self.register_buffer(
            "postgelu_hidden_hadamard",
            postgelu_hidden_hadamard,
            persistent=True,
        )
        # Optional non-persistent teacher state is installed by the training
        # entry point. It is neither trainable nor part of a checkpoint.
        self.register_buffer(
            "cproj_teacher_weight", None, persistent=False
        )
        self.last_cproj_teacher_alignment_loss: torch.Tensor | None = None
        self.last_postgelu: torch.Tensor | None = None
        self._cached_charted_cproj_weight: torch.Tensor | None = None

    def set_cproj_teacher_weight(self, weight: torch.Tensor) -> None:
        expected = (self.c_proj.out_features, self.c_proj.in_features)
        if tuple(weight.shape) != expected:
            raise ValueError(
                "c_proj teacher weight shape mismatch: expected "
                f"{expected}, got {tuple(weight.shape)}"
            )
        if not self.has_charted_cproj():
            raise ValueError(
                "c_proj teacher alignment requires a charted c_proj"
            )
        if getattr(self.c_proj, "bias", None) is not None:
            raise ValueError(
                "c_proj teacher alignment currently requires bias-free c_proj"
            )
        if self.output_rotation is not None:
            raise ValueError(
                "c_proj teacher alignment requires output rotation folded "
                "into the c_proj weight"
            )
        reference = next(self.parameters())
        self.cproj_teacher_weight = weight.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def cproj_teacher_alignment_loss(self) -> torch.Tensor | None:
        return self.last_cproj_teacher_alignment_loss

    def has_charted_cproj(self) -> bool:
        return any(
            value is not None
            for value in (
                self.hidden_block_rotation,
                self.hidden_log_gain,
                self.output_block_rotation,
                self.residual_output_log_gain,
            )
        )

    def has_charted_cfc(self) -> bool:
        return self.pregelu_block_rotation is not None

    def _cfc_base_weight(self) -> torch.Tensor:
        weight = getattr(self.c_fc, "_cached_weight", None)
        if weight is not None:
            return weight
        if not hasattr(self.c_fc, "weight"):
            raise RuntimeError(
                "pre-GELU frame requires a plain c_fc weight"
            )
        return self.c_fc.weight

    def _materialize_charted_cfc_weight(
        self, weight: torch.Tensor
    ) -> torch.Tensor:
        if self.pregelu_block_rotation is None:
            return weight
        # Row activations use h' = h @ R.  Since h = x @ W^T, folding the
        # frame into F.linear requires W' = R^T @ W.  Apply R to W^T and
        # transpose the result instead of materializing the O(hidden^2)
        # rotation matrix.
        return self.pregelu_block_rotation(
            weight.transpose(0, 1)
        ).transpose(0, 1).contiguous()

    def prepare_charted_cfc_cache(self) -> None:
        if not self.has_charted_cfc():
            return
        if self._cached_charted_cfc_weight is not None:
            return
        if getattr(self.c_fc, "bias", None) is not None:
            raise ValueError(
                "pre-GELU frame currently requires bias-free c_fc"
            )
        base_weight = self._cfc_base_weight()
        chart_requires_grad = any(
            parameter.requires_grad
            for parameter in self.pregelu_block_rotation.parameters()
        )
        retain_graph = bool(
            self.pregelu_cache_retain_graph
            and (base_weight.requires_grad or chart_requires_grad)
        )
        if retain_graph:
            with torch.enable_grad():
                charted_weight = self._materialize_charted_cfc_weight(
                    base_weight
                )
            self._cached_charted_cfc_graph_weight = charted_weight
        else:
            with torch.no_grad():
                charted_weight = self._materialize_charted_cfc_weight(
                    base_weight
                )
            self._cached_charted_cfc_graph_weight = None
        self._cached_charted_cfc_weight = (
            charted_weight.detach().requires_grad_(
                bool(base_weight.requires_grad or chart_requires_grad)
            )
        )

    def flush_charted_cfc_cache(
        self, *, project_base_gradient: bool = True
    ) -> None:
        cached = self._cached_charted_cfc_weight
        graph_weight = self._cached_charted_cfc_graph_weight
        self._cached_charted_cfc_weight = None
        self._cached_charted_cfc_graph_weight = None
        if cached is None:
            if graph_weight is not None:
                raise RuntimeError(
                    "pre-GELU cached graph exists without cached weight"
                )
            return
        if cached.grad is None:
            return
        if self.pregelu_block_rotation is None:
            raise RuntimeError("cached pre-GELU frame is missing")
        base_weight = self._cfc_base_weight()
        chart_parameters = [
            parameter
            for parameter in self.pregelu_block_rotation.parameters()
            if parameter.requires_grad
        ]
        if graph_weight is not None:
            gradient_targets = (
                [base_weight, *chart_parameters]
                if project_base_gradient
                else chart_parameters
            )
            if not gradient_targets:
                return
            gradients = torch.autograd.grad(
                graph_weight,
                gradient_targets,
                grad_outputs=cached.grad.to(dtype=graph_weight.dtype),
                allow_unused=True,
            )
        else:
            with torch.enable_grad():
                base_proxy = base_weight.detach()
                if project_base_gradient:
                    base_proxy.requires_grad_(True)
                charted_proxy = self._materialize_charted_cfc_weight(
                    base_proxy
                )
                gradient_targets = (
                    [base_proxy, *chart_parameters]
                    if project_base_gradient
                    else chart_parameters
                )
                if not gradient_targets:
                    return
                gradients = torch.autograd.grad(
                    charted_proxy,
                    gradient_targets,
                    grad_outputs=cached.grad.to(dtype=charted_proxy.dtype),
                    allow_unused=True,
                )
        if project_base_gradient:
            base_gradient = gradients[0].to(dtype=base_weight.dtype)
            if base_weight.grad is None:
                base_weight.grad = base_gradient
            else:
                base_weight.grad.add_(base_gradient)
            chart_gradients = gradients[1:]
        else:
            chart_gradients = gradients
        for parameter, gradient in zip(
            chart_parameters, chart_gradients
        ):
            if gradient is None:
                continue
            gradient = gradient.to(dtype=parameter.dtype)
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)

    def suspend_charted_cfc_cache(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cached = self._cached_charted_cfc_weight
        graph_weight = self._cached_charted_cfc_graph_weight
        self._cached_charted_cfc_weight = None
        self._cached_charted_cfc_graph_weight = None
        return cached, graph_weight

    def restore_charted_cfc_cache(
        self,
        suspended: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> None:
        cached, graph_weight = suspended
        if cached is None:
            if graph_weight is not None:
                raise RuntimeError(
                    "cannot restore pre-GELU graph without cached weight"
                )
            return
        if (
            self._cached_charted_cfc_weight is not None
            or self._cached_charted_cfc_graph_weight is not None
        ):
            raise RuntimeError(
                "cannot restore a charted c_fc cache over a live cache"
            )
        self._cached_charted_cfc_weight = cached
        self._cached_charted_cfc_graph_weight = graph_weight

    def _charted_cfc(self, values: torch.Tensor) -> torch.Tensor | None:
        if not self.has_charted_cfc():
            return None
        if self._cached_charted_cfc_weight is not None:
            return F.linear(
                values, self._cached_charted_cfc_weight, None
            )
        if getattr(self.c_fc, "bias", None) is not None:
            raise ValueError(
                "pre-GELU frame currently requires bias-free c_fc"
            )
        return F.linear(
            values,
            self._materialize_charted_cfc_weight(
                self._cfc_base_weight()
            ),
            None,
        )

    def _materialize_charted_cproj_weight(
        self, weight: torch.Tensor
    ) -> torch.Tensor:
        charted = weight
        if self.hidden_block_rotation is not None:
            rotation = self.hidden_block_rotation.matrix(charted)
            charted = charted @ rotation.transpose(0, 1)
        if self.hidden_log_gain is not None:
            gain = (
                self.hidden_gain_scale * self.hidden_log_gain
            ).exp().to(device=weight.device, dtype=weight.dtype)
            charted = charted * gain
        transposed = charted.transpose(0, 1)
        if self.residual_output_log_gain is not None:
            gain = (
                self.residual_output_gain_scale
                * self.residual_output_log_gain
            ).exp().to(device=weight.device, dtype=weight.dtype)
            transposed = transposed * gain
        if self.output_block_rotation is not None:
            rotation = self.output_block_rotation.matrix(transposed)
            transposed = transposed @ rotation
        return transposed.transpose(0, 1).contiguous()

    def _cproj_chart_parameters(
        self, *, require_grad_only: bool
    ) -> list[torch.Tensor]:
        parameters: list[torch.Tensor] = []
        if self.hidden_block_rotation is not None:
            parameters.extend(
                parameter
                for parameter in self.hidden_block_rotation.parameters()
                if not require_grad_only or parameter.requires_grad
            )
        if self.hidden_log_gain is not None and (
            not require_grad_only or self.hidden_log_gain.requires_grad
        ):
            parameters.append(self.hidden_log_gain)
        if self.output_block_rotation is not None:
            parameters.extend(
                parameter
                for parameter in self.output_block_rotation.parameters()
                if not require_grad_only or parameter.requires_grad
            )
        if self.residual_output_log_gain is not None and (
            not require_grad_only
            or self.residual_output_log_gain.requires_grad
        ):
            parameters.append(self.residual_output_log_gain)
        return parameters

    def prepare_charted_cproj_cache(self) -> None:
        """Materialize the complete generated c_proj chart once per step.

        BlockFHT's normal cache makes the generated base weight a leaf and
        projects its accumulated gradient back to the latent after all
        microbatches.  The output chart needs the same treatment; otherwise
        its rotation and weight-folding GEMM are repeated for every
        accumulation microbatch.
        """
        if not self.has_charted_cproj():
            return
        if self._cached_charted_cproj_weight is not None:
            return
        base_weight = getattr(self.c_proj, "_cached_weight", None)
        # Biases also need output-chart VJPs.  Current scientific GPT configs
        # are bias-free; retain the live path for a biased c_proj.
        if base_weight is None or getattr(self.c_proj, "bias", None) is not None:
            return
        with torch.no_grad():
            charted_weight = self._materialize_charted_cproj_weight(
                base_weight
            )
        chart_requires_grad = any(
            parameter.requires_grad
            for parameter in self._cproj_chart_parameters(
                require_grad_only=False
            )
        )
        self._cached_charted_cproj_weight = (
            charted_weight.detach().requires_grad_(
                bool(base_weight.requires_grad or chart_requires_grad)
            )
        )

    def flush_charted_cproj_cache(
        self, *, project_base_gradient: bool = True
    ) -> None:
        """Project a cached charted-weight gradient to chart/base parameters.

        Endpoint chart diagnostics freeze the generated base completely.  In
        that case, omitting the unused base-weight VJP avoids both a dense
        gradient and a later cache-to-latent projection while preserving the
        exact chart-coordinate VJP.  Normal training retains the historical
        default and projects both.
        """
        cached = self._cached_charted_cproj_weight
        if cached is None:
            return
        self._cached_charted_cproj_weight = None
        if cached.grad is None:
            return
        base_weight = getattr(self.c_proj, "_cached_weight", None)
        if base_weight is None:
            raise RuntimeError(
                "charted c_proj cache cannot flush without its base cache"
            )
        chart_parameters = self._cproj_chart_parameters(
            require_grad_only=True
        )
        with torch.enable_grad():
            base_proxy = base_weight.detach()
            if project_base_gradient:
                base_proxy.requires_grad_(True)
            charted_proxy = self._materialize_charted_cproj_weight(base_proxy)
            gradient_targets = (
                [base_proxy, *chart_parameters]
                if project_base_gradient
                else chart_parameters
            )
            gradients = torch.autograd.grad(
                charted_proxy,
                gradient_targets,
                grad_outputs=cached.grad.to(dtype=charted_proxy.dtype),
                allow_unused=True,
            )
        if project_base_gradient:
            base_gradient = gradients[0]
            if base_weight.grad is None:
                base_weight.grad = base_gradient.to(dtype=base_weight.dtype)
            else:
                base_weight.grad.add_(
                    base_gradient.to(dtype=base_weight.dtype)
                )
            chart_gradients = gradients[1:]
        else:
            chart_gradients = gradients
        for parameter, gradient in zip(chart_parameters, chart_gradients):
            if gradient is None:
                continue
            gradient = gradient.to(dtype=parameter.dtype)
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)

    def suspend_charted_cproj_cache(self) -> torch.Tensor | None:
        cached = self._cached_charted_cproj_weight
        self._cached_charted_cproj_weight = None
        return cached

    def restore_charted_cproj_cache(
        self, cached: torch.Tensor | None
    ) -> None:
        if cached is None:
            return
        if self._cached_charted_cproj_weight is not None:
            raise RuntimeError(
                "cannot restore a charted c_proj cache over a live cache"
            )
        self._cached_charted_cproj_weight = cached

    def _charted_cproj(self, activated: torch.Tensor) -> torch.Tensor | None:
        if not self.has_charted_cproj():
            return None
        if self._cached_charted_cproj_weight is not None:
            return F.linear(
                activated,
                self._cached_charted_cproj_weight,
                None,
            )
        weight = getattr(self.c_proj, "_cached_weight", None)
        if weight is None:
            weight = self.c_proj.weight
        bias = getattr(self.c_proj, "bias", None)
        charted_weight = self._materialize_charted_cproj_weight(weight)
        if bias is not None:
            if self.residual_output_log_gain is not None:
                gain = (
                    self.residual_output_gain_scale
                    * self.residual_output_log_gain
                ).exp().to(device=bias.device, dtype=bias.dtype)
                bias = bias * gain
            if self.output_block_rotation is not None:
                bias = bias @ self.output_block_rotation.matrix(bias)
        return F.linear(activated, charted_weight, bias)

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

    def activation_chart_log_scales(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return paired pre/post-GELU log scales from manifold coordinates.

        ``common`` moves both paired matrix norms in the same direction,
        ``gauge`` moves ``c_fc`` and ``c_proj`` oppositely, and the centered
        channel vector supplies the layer-specific hidden-channel path without
        duplicating the layerwise mean coordinate.
        """

        channel = self.activation_chart_channel_log_gain
        common = self.activation_chart_common_log_gain
        gauge = self.activation_chart_gauge_log_gain
        if channel is None or common is None or gauge is None:
            return None
        centered = self.activation_chart_channel_scale * (
            channel - channel.mean()
        )
        common_value = self.activation_chart_common_scale * common
        gauge_value = self.activation_chart_gauge_scale * gauge
        return (
            common_value - gauge_value + centered,
            common_value + gauge_value + centered,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_cproj_teacher_alignment_loss = None
        hidden = self._charted_cfc(x)
        if hidden is None:
            hidden = self.c_fc(x)
        if self.lowrank_left is not None and self.lowrank_right is not None:
            residual = x.matmul(self.lowrank_left.to(dtype=x.dtype)).matmul(self.lowrank_right.to(dtype=x.dtype))
            hidden = hidden + self.lowrank_scale * residual
        if self.pregelu_gain is not None:
            hidden = hidden * self.pregelu_gain.to(dtype=hidden.dtype)
        if self.pregelu_bias is not None:
            hidden = hidden + self.pregelu_bias.to(dtype=hidden.dtype)
        activation_chart = self.activation_chart_log_scales()
        if activation_chart is not None:
            pre_log_gain, post_log_gain = activation_chart
            hidden = hidden * pre_log_gain.exp().to(dtype=hidden.dtype)
        shared_hidden_gain = None
        if self.shared_hidden_log_gain is not None:
            shared_hidden_gain = (
                self.shared_hidden_gain_scale * self.shared_hidden_log_gain
            ).exp().to(dtype=hidden.dtype)
            hidden = hidden * shared_hidden_gain
        activated = self.gelu(hidden)
        if activation_chart is not None:
            activated = activated * post_log_gain.exp().to(
                dtype=activated.dtype
            )
        if shared_hidden_gain is not None:
            activated = activated * shared_hidden_gain
        activated = self.apply_postgelu_hidden_self_gate(activated)
        if self.training and self.postgelu_std_target > 0.0:
            self.last_postgelu = activated
        else:
            self.last_postgelu = None
        out = self._charted_cproj(activated)
        if out is None:
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
            and self.cproj_spectral_resid_in_basis is not None
            and self.cproj_spectral_resid_out_basis is not None
        ):
            mixed_in = F.linear(activated, self.cproj_spectral_resid_in_basis.transpose(0, 1))
            if (
                self.cproj_spectral_resid_diag.ndim == 2
                and self.cproj_spectral_resid_diag.shape[0]
                == self.cproj_spectral_resid_diag.shape[1]
            ):
                spectral = F.linear(
                    mixed_in,
                    self.cproj_spectral_resid_diag.to(dtype=activated.dtype),
                )
            else:
                spectral = (
                    mixed_in
                    * self.cproj_spectral_resid_diag.reshape(-1).to(
                        dtype=activated.dtype
                    )
                )
            spectral = F.linear(spectral, self.cproj_spectral_resid_out_basis)
            out = out + self.cproj_spectral_resid_scale.to(dtype=out.dtype) * spectral
        if self.output_rotation is not None:
            out = self.output_rotation(out)
        if self.training and self.cproj_teacher_weight is not None:
            detached_activated = activated.detach()
            aligned_student_weight = self._cached_charted_cproj_weight
            if aligned_student_weight is None:
                base_weight = getattr(self.c_proj, "_cached_weight", None)
                if base_weight is None:
                    base_weight = self.c_proj.weight
                aligned_student_weight = (
                    self._materialize_charted_cproj_weight(base_weight)
                )
            teacher_weight = self.cproj_teacher_weight.to(
                device=aligned_student_weight.device,
                dtype=aligned_student_weight.dtype,
            )
            # F.linear(x, Ws) - F.linear(x, Wt) is exactly
            # F.linear(x, Ws-Wt), but costs one projection GEMM rather than
            # two. The detached activation keeps this diagnostic signal on
            # the generated c_proj base and chart only.
            alignment_residual = F.linear(
                detached_activated,
                aligned_student_weight - teacher_weight,
                None,
            )
            self.last_cproj_teacher_alignment_loss = (
                alignment_residual.float().square().mean()
            )
        if (
            self.residual_conditioned_output_slope is not None
            and self.conditioned_output_gate_source == "postgelu"
        ):
            out = self.apply_residual_conditioned_output_gate(
                self.postgelu_conditioned_output_condition(activated),
                out,
            )
        return self.dropout(out)

    def postgelu_conditioned_output_condition(
        self,
        activated: torch.Tensor,
    ) -> torch.Tensor:
        signs = self.conditioned_output_projection_signs
        if signs is None:
            raise RuntimeError(
                "post-GELU conditioned output projection is not configured"
            )
        expected = 4 * self.c_proj.out_features
        if activated.shape[-1] != expected:
            raise ValueError(
                "post-GELU condition width mismatch: expected "
                f"{expected}, got {activated.shape[-1]}"
            )
        grouped = activated.reshape(
            *activated.shape[:-1],
            4,
            self.c_proj.out_features,
        )
        signs = signs.to(
            device=activated.device,
            dtype=activated.dtype,
        )
        condition = (grouped * signs).sum(dim=-2) / 2.0
        rms = condition.float().square().mean(
            dim=-1,
            keepdim=True,
        ).add(self.conditioned_output_gate_rms_epsilon).sqrt()
        return condition / rms.to(dtype=condition.dtype)

    def _postgelu_hidden_self_basis(
        self,
        values: torch.Tensor,
        *,
        inverse: bool,
        role: str,
    ) -> torch.Tensor:
        if role == "condition":
            permutation = self.postgelu_hidden_condition_permutation
            inverse_permutation = (
                self.postgelu_hidden_condition_inverse_permutation
            )
            signs = self.postgelu_hidden_condition_signs
        elif role == "update":
            permutation = self.postgelu_hidden_update_permutation
            inverse_permutation = (
                self.postgelu_hidden_update_inverse_permutation
            )
            signs = self.postgelu_hidden_update_signs
        elif role == "output":
            permutation = self.postgelu_hidden_output_permutation
            inverse_permutation = (
                self.postgelu_hidden_output_inverse_permutation
            )
            signs = self.postgelu_hidden_output_signs
        else:
            raise ValueError(f"unsupported post-GELU basis role: {role}")
        if permutation is None:
            return values
        assert inverse_permutation is not None and signs is not None
        multihead = permutation.dim() == 2
        if inverse and not multihead:
            transformed = fixed_basis_transform(
                values.unsqueeze(0),
                permutation,
                signs,
                self.postgelu_hidden_self_gate_basis_block_size,
                inverse=True,
                shared_input=False,
            )
            return transformed[0]
        transformed = fixed_basis_transform(
            values,
            permutation,
            signs,
            self.postgelu_hidden_self_gate_basis_block_size,
            inverse=inverse,
            shared_input=not inverse,
        )
        if not multihead:
            return transformed[0]
        return transformed

    def apply_postgelu_hidden_self_gate(
        self,
        activated: torch.Tensor,
    ) -> torch.Tensor:
        slope = self.postgelu_hidden_self_slope
        if slope is None:
            return activated
        expected = 4 * self.c_proj.out_features
        if activated.shape[-1] != expected:
            raise ValueError(
                "post-GELU hidden self gate width mismatch: expected "
                f"{expected}, got {activated.shape[-1]}"
            )
        norm = torch.linalg.vector_norm(
            activated,
            ord=2,
            dim=-1,
            keepdim=True,
            dtype=torch.float32,
        )
        rms = norm.square().div(activated.shape[-1]).add(
            self.postgelu_hidden_self_gate_rms_epsilon
        ).sqrt()
        condition = activated / rms.to(dtype=activated.dtype)
        if slope.dim() == 2 and activated.is_cuda:
            assert (
                self.postgelu_hidden_condition_permutation is not None
                and self.postgelu_hidden_condition_signs is not None
                and self.postgelu_hidden_update_permutation is not None
                and self.postgelu_hidden_update_signs is not None
                and self.postgelu_hidden_output_permutation is not None
                and self.postgelu_hidden_output_signs is not None
            )
            return postgelu_multihead_mix(
                activated,
                condition,
                slope,
                self.postgelu_hidden_condition_permutation,
                self.postgelu_hidden_condition_signs,
                self.postgelu_hidden_update_permutation,
                self.postgelu_hidden_update_signs,
                self.postgelu_hidden_output_permutation,
                self.postgelu_hidden_output_signs,
                self.postgelu_hidden_self_gate_basis_block_size,
                self.postgelu_hidden_self_gate_scale,
            )
        spectral_condition = self._postgelu_hidden_self_basis(
            condition,
            inverse=False,
            role="condition",
        )
        slope_for_modulation = slope.to(dtype=spectral_condition.dtype)
        if slope_for_modulation.dim() == 2:
            slope_for_modulation = slope_for_modulation.reshape(
                slope_for_modulation.shape[0],
                *([1] * (spectral_condition.dim() - 2)),
                slope_for_modulation.shape[1],
            )
        modulation = (
            self.postgelu_hidden_self_gate_scale
            * spectral_condition
            * slope_for_modulation
        )
        spectral_update = self._postgelu_hidden_self_basis(
            activated,
            inverse=False,
            role="update",
        )
        correction = spectral_update * modulation
        transformed = self._postgelu_hidden_self_basis(
            correction,
            inverse=True,
            role="output",
        )
        if transformed.dim() == activated.dim() + 1:
            transformed = transformed.sum(dim=0)
        return activated + transformed

    def residual_conditioned_output_modulation(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.residual_conditioned_output_slope is None:
            return None
        condition = self._residual_conditioned_output_basis(
            condition,
            inverse=False,
            role="condition",
        )
        slope = self.residual_conditioned_output_slope.to(
            dtype=condition.dtype
        )
        if self.residual_conditioned_output_bias is None:
            modulation = condition * slope
        else:
            modulation = torch.addcmul(
                self.residual_conditioned_output_bias.to(
                    dtype=condition.dtype
                ),
                condition,
                slope,
            )
        if self.residual_conditioned_output_gate_scale != 1.0:
            modulation = (
                self.residual_conditioned_output_gate_scale * modulation
            )
        return modulation

    def _residual_conditioned_output_basis(
        self,
        values: torch.Tensor,
        *,
        inverse: bool,
        role: str,
    ) -> torch.Tensor:
        if role == "condition" or not self.residual_conditioned_output_gate_untied_bases:
            permutation = self.residual_conditioned_output_permutation
            inverse_permutation = (
                self.residual_conditioned_output_inverse_permutation
            )
            signs = self.residual_conditioned_output_signs
        elif role == "update":
            permutation = self.residual_conditioned_output_update_permutation
            inverse_permutation = (
                self.residual_conditioned_output_update_inverse_permutation
            )
            signs = self.residual_conditioned_output_update_signs
        elif role == "output":
            permutation = self.residual_conditioned_output_output_permutation
            inverse_permutation = (
                self.residual_conditioned_output_output_inverse_permutation
            )
            signs = self.residual_conditioned_output_output_signs
        else:
            raise ValueError(f"unsupported residual gate basis role: {role}")
        hadamard = self.residual_conditioned_output_hadamard
        if permutation is None:
            return values
        assert (
            inverse_permutation is not None
            and signs is not None
            and hadamard is not None
        )
        signs = signs.to(device=values.device, dtype=values.dtype)
        hadamard = hadamard.to(device=values.device, dtype=values.dtype)
        if inverse:
            values = values * signs
            grouped = values.reshape(
                *values.shape[:-1],
                values.shape[-1]
                // self.residual_conditioned_output_gate_basis_block_size,
                self.residual_conditioned_output_gate_basis_block_size,
            )
            values = F.linear(grouped, hadamard).reshape_as(values)
            return values.index_select(-1, inverse_permutation)
        values = values.index_select(-1, permutation)
        grouped = values.reshape(
            *values.shape[:-1],
            values.shape[-1]
            // self.residual_conditioned_output_gate_basis_block_size,
            self.residual_conditioned_output_gate_basis_block_size,
        )
        values = F.linear(grouped, hadamard).reshape_as(values)
        return values * signs

    def apply_residual_conditioned_output_gate(
        self,
        condition: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if condition.shape != update.shape:
            raise ValueError(
                "gate condition and update must be aligned, got "
                f"{tuple(condition.shape)} and {tuple(update.shape)}"
            )
        modulation = self.residual_conditioned_output_modulation(condition)
        if modulation is None:
            return update
        if not self.residual_conditioned_output_gate_fixed_basis:
            return torch.addcmul(
                update,
                update,
                modulation.to(dtype=update.dtype),
            )
        spectral_update = self._residual_conditioned_output_basis(
            update,
            inverse=False,
            role="update",
        )
        correction = spectral_update * modulation.to(
            dtype=spectral_update.dtype
        )
        return update + self._residual_conditioned_output_basis(
            correction,
            inverse=True,
            role="output",
        )

    def postgelu_spread_loss(self) -> torch.Tensor | None:
        if self.last_postgelu is None or self.postgelu_std_target <= 0.0:
            return None
        # Do not materialize a full FP32 [batch*sequence, 4*width] copy here:
        # at b64/1024 that is 1GiB *per layer* and can dominate the model
        # activation budget. CUDA's reduction accumulates the BF16 input
        # without that copy; only the per-channel result is promoted to FP32
        # for the small penalty calculation.
        std = self.last_postgelu.std(dim=(0, 1), unbiased=False).float()
        target = std.new_tensor(self.postgelu_std_target)
        return torch.relu(target - std).square().mean()


class Block(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        layer_id: int,
        shared_compact_mlp_transport: SharedGroupTransport | None = None,
        shared_compact_mlp_core: SharedDenseResidualCore | None = None,
        shared_compact_mlp_tensor_product: SharedTensorProductRidgeField | None = None,
        shared_compact_mlp_recurrent_core: SharedRecurrentDenseCore | None = None,
        shared_compact_mlp_paired_atom_bank: SharedPairedAtomBank | None = None,
        shared_compact_mlp_householder_bank: SharedHouseholderAtomBank | None = None,
        shared_compact_mlp_spectral_phase_bank: SharedSpectralPhaseAtomBank | None = None,
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if config.compact_native_mlp == "":
            self.mlp = MLP(config, layer_id)
        elif config.compact_native_mlp == "sparse_permutation_swiglu4":
            self.mlp = SparsePermutationSwiGLUMLP(config, layer_id)
        elif config.compact_native_mlp == "shared_transport_grouped_gelu4":
            if shared_compact_mlp_transport is None:
                raise ValueError("shared compact MLP transport is required")
            self.mlp = SharedTransportGroupedGELUMLP(
                config,
                layer_id,
                shared_compact_mlp_transport,
            )
        elif config.compact_native_mlp == "shared_dense_core_residual736":
            if shared_compact_mlp_core is None:
                raise ValueError("shared compact MLP core is required")
            self.mlp = SharedDenseCoreResidualMLP(
                config,
                layer_id,
                shared_compact_mlp_core,
            )
        elif config.compact_native_mlp == "semiprivate_tensorproduct_r20":
            if shared_compact_mlp_tensor_product is None:
                raise ValueError("shared tensor-product ridge field is required")
            self.mlp = SemiprivateTensorProductRidgeMLP(
                config,
                layer_id,
                shared_compact_mlp_tensor_product,
            )
        elif config.compact_native_mlp == "shared_recurrent_core_residual720x2":
            if shared_compact_mlp_recurrent_core is None:
                raise ValueError("shared recurrent compact MLP core is required")
            self.mlp = TwoStepRecurrentDenseCoreMLP(
                config,
                layer_id,
                shared_compact_mlp_recurrent_core,
            )
        elif config.compact_native_mlp == "paired_shared_private_atoms236x10":
            if shared_compact_mlp_paired_atom_bank is None:
                raise ValueError("shared paired-atom bank is required")
            self.mlp = PairedSharedPrivateAtomMLP(
                config,
                layer_id,
                shared_compact_mlp_paired_atom_bank,
            )
        elif config.compact_native_mlp == "shared_householder_frame312x4":
            if shared_compact_mlp_householder_bank is None:
                raise ValueError("shared Householder atom bank is required")
            self.mlp = HouseholderFrameSharedAtomMLP(
                config,
                layer_id,
                shared_compact_mlp_householder_bank,
            )
        elif config.compact_native_mlp == "shared_spectral_phase_frame353":
            if shared_compact_mlp_spectral_phase_bank is None:
                raise ValueError("shared spectral-phase atom bank is required")
            self.mlp = SpectralPhaseFrameSharedAtomMLP(
                config,
                layer_id,
                shared_compact_mlp_spectral_phase_bank,
            )
        else:
            raise ValueError(
                "unsupported compact_native_mlp: "
                f"{config.compact_native_mlp!r}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        mlp_input = self.ln_2(x)
        mlp_output = self.mlp(mlp_input)
        if (
            self.mlp.residual_conditioned_output_slope is None
            or self.mlp.conditioned_output_gate_source == "postgelu"
        ):
            return x + mlp_output
        return x + self.mlp.apply_residual_conditioned_output_gate(
            mlp_input,
            mlp_output,
        )


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        affine_delta_targets = set(config.block_fht_affine_delta_targets)
        if not affine_delta_targets.issubset(config.block_fht_targets):
            raise ValueError("affine-delta targets must also be BlockFHT targets")
        if affine_delta_targets and config.block_fht_residual_base_scale != 0.0:
            raise ValueError(
                "target-selective affine deltas cannot be combined with the "
                "legacy global residual base"
            )
        if affine_delta_targets and (
            not math.isfinite(float(config.block_fht_affine_delta_scale))
            or float(config.block_fht_affine_delta_scale) <= 0.0
        ):
            raise ValueError("block_fht_affine_delta_scale must be positive and finite")
        self.config = config
        if config.compact_native_mlp == "shared_transport_grouped_gelu4":
            group_size = int(config.compact_native_mlp_group_size)
            if config.n_embd % group_size:
                raise ValueError("n_embd must be divisible by compact MLP group size")
            self.shared_compact_mlp_transport = SharedGroupTransport(
                config.n_embd // group_size
            )
        else:
            self.shared_compact_mlp_transport = None
        if config.compact_native_mlp == "shared_dense_core_residual736":
            self.shared_compact_mlp_core = SharedDenseResidualCore(config)
        else:
            self.shared_compact_mlp_core = None
        if config.compact_native_mlp == "semiprivate_tensorproduct_r20":
            self.shared_compact_mlp_tensor_product = SharedTensorProductRidgeField(
                config
            )
        else:
            self.shared_compact_mlp_tensor_product = None
        if config.compact_native_mlp == "shared_recurrent_core_residual720x2":
            self.shared_compact_mlp_recurrent_core = SharedRecurrentDenseCore(config)
        else:
            self.shared_compact_mlp_recurrent_core = None
        if config.compact_native_mlp == "paired_shared_private_atoms236x10":
            self.shared_compact_mlp_paired_atom_bank = SharedPairedAtomBank(config)
        else:
            self.shared_compact_mlp_paired_atom_bank = None
        if config.compact_native_mlp == "shared_householder_frame312x4":
            self.shared_compact_mlp_householder_bank = SharedHouseholderAtomBank(
                config
            )
        else:
            self.shared_compact_mlp_householder_bank = None
        if config.compact_native_mlp == "shared_spectral_phase_frame353":
            self.shared_compact_mlp_spectral_phase_bank = SharedSpectralPhaseAtomBank(
                config
            )
        else:
            self.shared_compact_mlp_spectral_phase_bank = None
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            config,
                            layer_id,
                            self.shared_compact_mlp_transport,
                            self.shared_compact_mlp_core,
                            self.shared_compact_mlp_tensor_product,
                            self.shared_compact_mlp_recurrent_core,
                            self.shared_compact_mlp_paired_atom_bank,
                            self.shared_compact_mlp_householder_bank,
                            self.shared_compact_mlp_spectral_phase_bank,
                        )
                        for layer_id in range(config.n_layer)
                    ]
                ),
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
        for module in self.modules():
            if (
                isinstance(module, GradientSeededLowRankLinear)
                and module.target_name == "mlp.c_proj"
            ):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_compact_mlp_gradient_bootstrap(self) -> int:
        modules = [
            module
            for module in self.modules()
            if isinstance(module, GradientSeededLowRankLinear)
        ]
        for module in modules:
            module.enable_gradient_bootstrap()
        return len(modules)

    def finish_compact_mlp_gradient_bootstrap(
        self,
    ) -> list[dict[str, float | int | str]]:
        return [
            module.finish_gradient_bootstrap()
            for module in self.modules()
            if isinstance(module, GradientSeededLowRankLinear)
        ]

    def compact_mlp_gradient_seeded_stats(self) -> dict[str, int]:
        modules = [
            module
            for module in self.modules()
            if isinstance(module, GradientSeededLowRankLinear)
        ]
        return {
            "modules": len(modules),
            "fixed_seed_scalars": sum(
                module.in_features * module.out_features for module in modules
            ),
            "trainable_factor_scalars": sum(
                module.rank * (module.in_features + module.out_features)
                for module in modules
            ),
        }

    def forward(
        self,
        idx: torch.Tensor | None,
        targets: torch.Tensor | None = None,
        *,
        input_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (idx is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of idx or input_embeddings")
        if input_embeddings is None:
            assert idx is not None
            token_embeddings = self.transformer.wte(idx)
            device = idx.device
            bsz, seq_len = idx.size()
        else:
            if input_embeddings.ndim != 3:
                raise ValueError("input_embeddings must have shape [batch, sequence, width]")
            if input_embeddings.shape[-1] != self.config.n_embd:
                raise ValueError(
                    "input embedding width mismatch: expected "
                    f"{self.config.n_embd}, got {input_embeddings.shape[-1]}"
                )
            token_embeddings = input_embeddings
            device = input_embeddings.device
            bsz, seq_len, _ = input_embeddings.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        x = self.transformer.drop(token_embeddings + self.transformer.wpe(pos))
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

    def mlp_cproj_teacher_alignment_loss(self) -> torch.Tensor:
        losses = []
        for block in self.transformer.h:
            loss = block.mlp.cproj_teacher_alignment_loss()
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
        block_fht_attn_cayley_lr_scale: float = 1.0,
        block_fht_mlp_chart_lr_scale: float = 1.0,
        block_fht_mlp_pregelu_chart_lr_scale: float = 1.0,
    ):
        params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        decay = [param for _, param in params.items() if param.dim() >= 2]
        nodecay = [param for _, param in params.items() if param.dim() < 2]
        muon_matched_givens_modules = [
            module
            for module in self.modules()
            if isinstance(module, MuonMatchedGivensLinear)
        ]
        if muon_matched_givens_modules and optimizer != "muon":
            raise ValueError(
                "Muon-matched Givens c_proj requires optimizer='muon'"
            )
        if optimizer == "muon":
            product_factor_tokens = (
                "product_log_diagonals",
                "product_output_log_gain",
            )
            product_factors = [
                param
                for name, param in params.items()
                if any(
                    token in name for token in product_factor_tokens
                )
            ]
            product_factor_ids = {
                id(param) for param in product_factors
            }
            gradient_seeded_factors = [
                param
                for name, param in params.items()
                if "gradient_seeded_" in name
            ]
            gradient_seeded_factor_ids = {
                id(param) for param in gradient_seeded_factors
            }
            grouped_mlp_factors = [
                param
                for name, param in params.items()
                if (
                    "grouped_c_fc_weight" in name
                    or "grouped_c_proj_weight" in name
                )
            ]
            grouped_mlp_factor_ids = {
                id(param) for param in grouped_mlp_factors
            }
            matrix = [
                param
                for name, param in params.items()
                if param.dim() >= 2
                and "wte" not in name
                and "wpe" not in name
                and "lm_head" not in name
                and id(param) not in product_factor_ids
                and id(param) not in gradient_seeded_factor_ids
                and id(param) not in grouped_mlp_factor_ids
            ]
            other = [
                param
                for name, param in params.items()
                if (
                    param.dim() < 2
                    or "wte" in name
                    or "wpe" in name
                    or "lm_head" in name
                    or id(param) in gradient_seeded_factor_ids
                )
                and id(param) not in product_factor_ids
            ]
            chart_names = (
                "hidden_block_rotation.coordinates",
                ".hidden_log_gain",
                "output_block_rotation.coordinates",
                ".residual_output_log_gain",
                ".residual_conditioned_output_slope",
                ".residual_conditioned_output_bias",
                ".activation_chart_channel_log_gain",
                ".activation_chart_common_log_gain",
                ".activation_chart_gauge_log_gain",
            )
            other_parameter_ids = {id(param) for param in other}
            chart_other = [
                param
                for name, param in params.items()
                if id(param) in other_parameter_ids
                and any(token in name for token in chart_names)
            ]
            chart_parameter_ids = {id(param) for param in chart_other}
            pregelu_chart_other = [
                param
                for name, param in params.items()
                if id(param) in other_parameter_ids
                and "pregelu_block_rotation.coordinates" in name
            ]
            pregelu_chart_parameter_ids = {
                id(param) for param in pregelu_chart_other
            }
            attention_cayley_tokens = (
                ".qk_input_cayley.",
                ".qk_output_cayley.",
                ".v_input_cayley.",
                ".v_output_cayley.",
                ".cproj_input_cayley.",
                ".cproj_output_cayley.",
            )
            attention_cayley_other = [
                param
                for name, param in params.items()
                if id(param) in other_parameter_ids
                and any(token in name for token in attention_cayley_tokens)
            ]
            attention_cayley_parameter_ids = {
                id(param) for param in attention_cayley_other
            }
            regular_other = [
                param
                for param in other
                if id(param) not in chart_parameter_ids
                and id(param) not in pregelu_chart_parameter_ids
                and id(param) not in attention_cayley_parameter_ids
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
            if grouped_mlp_factors:
                optimizers.append(
                    GroupedMuon(
                        grouped_mlp_factors,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=muon_ns_steps,
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = 1.0
            if product_factors:
                optimizers.append(
                    torch.optim.SGD(
                        [
                            {
                                "params": product_factors,
                                "weight_decay": 0.0,
                                "lr_scale": 1.0,
                            }
                        ],
                        lr=learning_rate,
                    )
                )
            if muon_matched_givens_modules:
                optimizers.append(
                    MuonMatchedGivens(
                        muon_matched_givens_modules,
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
                fallback_groups = []
                if regular_other:
                    fallback_groups.append(
                        {
                            "params": regular_other,
                            "weight_decay": 0.0,
                            "lr_scale": float(muon_adamw_lr_scale),
                        }
                    )
                if chart_other:
                    fallback_groups.append(
                        {
                            "params": chart_other,
                            "weight_decay": 0.0,
                            "lr_scale": (
                                float(muon_adamw_lr_scale)
                                * float(block_fht_mlp_chart_lr_scale)
                            ),
                        }
                    )
                if pregelu_chart_other:
                    fallback_groups.append(
                        {
                            "params": pregelu_chart_other,
                            "weight_decay": 0.0,
                            "lr_scale": (
                                float(muon_adamw_lr_scale)
                                * float(
                                    block_fht_mlp_pregelu_chart_lr_scale
                                )
                            ),
                        }
                    )
                if attention_cayley_other:
                    fallback_groups.append(
                        {
                            "params": attention_cayley_other,
                            "weight_decay": 0.0,
                            "lr_scale": (
                                float(muon_adamw_lr_scale)
                                * float(block_fht_attn_cayley_lr_scale)
                            ),
                        }
                    )
                optimizers.append(
                    torch.optim.AdamW(
                        fallback_groups,
                        lr=adamw_lr,
                        betas=betas,
                        **extra_args,
                    )
                )
            else:
                adamw_lr = learning_rate * float(muon_adamw_lr_scale)
            print(
                f"optimizer=muon matrix_tensors={len(matrix)} adamw_other_tensors={len(other)} "
                f"product_fht_factor_tensors={len(product_factors)} "
                f"gradient_seeded_factor_tensors={len(gradient_seeded_factors)} "
                "muon_matched_givens_tensors="
                f"{len(muon_matched_givens_modules)} "
                f"mlp_chart_tensors={len(chart_other)} "
                f"mlp_pregelu_chart_tensors={len(pregelu_chart_other)} "
                f"momentum={muon_momentum} ns_steps={muon_ns_steps} "
                f"adamw_lr_scale={float(muon_adamw_lr_scale)} "
                f"mlp_chart_lr_scale={float(block_fht_mlp_chart_lr_scale)} "
                f"mlp_pregelu_chart_lr_scale="
                f"{float(block_fht_mlp_pregelu_chart_lr_scale)} "
                f"attn_cayley_tensors={len(attention_cayley_other)} "
                f"attn_cayley_lr_scale="
                f"{float(block_fht_attn_cayley_lr_scale)} "
                f"adamw_lr={adamw_lr}"
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
            elif isinstance(module, ProductFHTLinear):
                modules += 1
                generated += module.in_features * module.out_features
                latent += module.trainable_scalar_count
            elif isinstance(module, MuonMatchedGivensLinear):
                modules += 1
                generated += module.in_features * module.out_features
                latent += module.coordinate_count
        return {"modules": modules, "generated": generated, "latent": latent}

    def prepare_block_fht_cache(self, dtype: torch.dtype | None = None) -> None:
        prepare_block_fht_weight_cache(self, dtype=dtype)
        for block in self.transformer.h:
            block.mlp.prepare_charted_cfc_cache()
            block.mlp.prepare_charted_cproj_cache()

    def flush_block_fht_cache(self) -> None:
        for block in self.transformer.h:
            block.mlp.flush_charted_cfc_cache()
            block.mlp.flush_charted_cproj_cache()
        flush_block_fht_weight_cache(self)

    def set_product_fht_factor_learning_rate(
        self, learning_rate: float
    ) -> None:
        for module in self.modules():
            if isinstance(module, ProductFHTLinear):
                module.set_factor_learning_rate(learning_rate)

    def product_fht_clip_parameters(self) -> list[torch.Tensor]:
        excluded = {
            id(parameter)
            for module in self.modules()
            if (
                isinstance(module, ProductFHTLinear)
                and module.pullback_normalize
            )
            for parameter in (
                module.product_log_diagonals,
                module.product_output_log_gain,
            )
        }
        parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in excluded
        ]
        parameters.extend(
            module.weight
            for module in self.modules()
            if isinstance(module, MuonMatchedGivensLinear)
        )
        return parameters

    def finalize_product_fht_pullback_probes(
        self,
    ) -> list[dict[str, float]]:
        diagnostics = []
        for layer, block in enumerate(self.transformer.h):
            module = getattr(block.mlp, "c_proj", None)
            if not isinstance(module, ProductFHTLinear):
                continue
            row = module.finalize_pullback_probe()
            if row is not None:
                diagnostics.append({"layer": layer, **row})
        return diagnostics

    def suspend_block_fht_cache(self):
        charted = [
            (
                block.mlp,
                block.mlp.suspend_charted_cfc_cache(),
                block.mlp.suspend_charted_cproj_cache(),
            )
            for block in self.transformer.h
        ]
        return suspend_block_fht_weight_cache(self), charted

    def restore_block_fht_cache(self, suspended) -> None:
        block_fht, charted = suspended
        restore_block_fht_weight_cache(block_fht)
        for mlp, cfc_cached, cproj_cached in charted:
            mlp.restore_charted_cfc_cache(cfc_cached)
            mlp.restore_charted_cproj_cache(cproj_cached)


def freeze_non_block_fht(model: nn.Module, train_embeddings: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(
            module,
            (
                SparsePermutationSwiGLUMLP,
                SharedTransportGroupedGELUMLP,
                SharedGroupTransport,
                SharedDenseCoreResidualMLP,
                SharedDenseResidualCore,
                SemiprivateTensorProductRidgeMLP,
                SharedTensorProductRidgeField,
                TwoStepRecurrentDenseCoreMLP,
                SharedRecurrentDenseCore,
                PairedSharedPrivateAtomMLP,
                SharedPairedAtomBank,
                HouseholderFrameSharedAtomMLP,
                SharedHouseholderAtomBank,
                SpectralPhaseFrameSharedAtomMLP,
                SharedSpectralPhaseAtomBank,
            ),
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
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
        if isinstance(module, ProductFHTLinear):
            module.product_log_diagonals.requires_grad_(True)
            module.product_output_log_gain.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
        if isinstance(module, MLP) and module.shared_hidden_log_gain is not None:
            module.shared_hidden_log_gain.requires_grad_(True)
        if isinstance(module, MLP):
            for parameter in (
                module.activation_chart_channel_log_gain,
                module.activation_chart_common_log_gain,
                module.activation_chart_gauge_log_gain,
            ):
                if parameter is not None:
                    parameter.requires_grad_(True)
        if (
            isinstance(module, MLP)
            and module.pregelu_block_rotation is not None
        ):
            module.pregelu_block_rotation.coordinates.requires_grad_(True)
        if (
            isinstance(module, MLP)
            and module.hidden_block_rotation is not None
        ):
            module.hidden_block_rotation.coordinates.requires_grad_(True)
        if (
            isinstance(module, MLP)
            and module.hidden_log_gain is not None
        ):
            module.hidden_log_gain.requires_grad_(True)
        if (
            isinstance(module, MLP)
            and module.output_block_rotation is not None
        ):
            module.output_block_rotation.coordinates.requires_grad_(True)
        if (
            isinstance(module, MLP)
            and module.residual_output_log_gain is not None
        ):
            module.residual_output_log_gain.requires_grad_(True)
        if isinstance(module, MLP):
            for parameter in (
                module.residual_conditioned_output_slope,
                module.residual_conditioned_output_bias,
            ):
                if parameter is not None:
                    parameter.requires_grad_(True)
    if train_embeddings and isinstance(model, GPT):
        model.transformer.wte.weight.requires_grad_(True)
        model.transformer.wpe.weight.requires_grad_(True)
        for param in model.transformer.ln_f.parameters():
            param.requires_grad_(True)
