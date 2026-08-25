from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from examples.nanogpt.muon import Muon
from examples.nanogpt.muon_int8_lattice import (
    MuonInt8Lattice,
    MuonInt8LatticeLinear,
)
from examples.nanogpt.muon_pair_vq import MuonPairVQ, MuonPairVQLinear
from examples.nanogpt.muon_matched_givens import (
    MuonDirectedProduct,
    MuonDirectedProductLinear,
    MuonFunctionalShear,
    MuonFunctionalShearLinear,
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

    @property
    def compact_boundary_ready(self) -> bool:
        return all(
            bool(getattr(optimizer, "compact_boundary_ready", True))
            for optimizer in self.optimizers
        )

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
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_expert_hidden_multiplier: int = 2
    moe_load_balance_aux_coefficient: float = 0.01
    moe_router_z_loss_coefficient: float = 0.001
    moe_unpadded_expert_loop: bool = False
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
    block_fht_attn_pack_cached_qkv: bool = False
    block_fht_attn_cayley_atlas_start_steps: tuple[int, ...] = ()
    block_fht_attn_cayley_factor_optimizer: str = "adamw"
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
    block_fht_attn_muon_matched_givens_targets: tuple[str, ...] = ()
    block_fht_attn_muon_matched_givens_stages: int = 64
    block_fht_attn_muon_matched_givens_neighbors: int = 128
    block_fht_attn_muon_matched_givens_refresh_interval: int = 15
    block_fht_attn_muon_matched_givens_fast_matching: bool = True
    block_fht_attn_muon_matched_givens_seed: int = 161803
    block_fht_attn_muon_matched_givens_seed_step_stride: int = 8192
    block_fht_attn_muon_matched_givens_error_feedback: bool = False
    block_fht_attn_muon_matched_givens_error_feedback_decay: float = 0.5
    block_fht_attn_muon_matched_givens_error_feedback_max_nominal_steps: float | None = None
    block_fht_attn_cproj_int8_lattice: bool = False
    block_fht_attn_cproj_int8_lattice_block_size: int = 4096
    block_fht_attn_cproj_int8_lattice_seed: int = 271828
    block_fht_attn_cproj_int8_lattice_error_feedback: bool = False
    block_fht_attn_v_int8_lattice: bool = False
    block_fht_attn_v_int8_lattice_block_size: int = 4096
    block_fht_attn_v_int8_lattice_seed: int = 161804
    block_fht_attn_v_int8_lattice_error_feedback: bool = False
    block_fht_attn_pair_vq: bool = False
    block_fht_attn_pair_vq_seed: int = 20261121
    block_fht_mlp_int8_lattice_targets: tuple[str, ...] = ()
    block_fht_mlp_int8_lattice_block_size: int = 4096
    block_fht_mlp_int8_lattice_seed: int = 314159
    block_fht_mlp_int8_lattice_error_feedback: bool = False
    block_fht_mlp_pair_vq: bool = False
    block_fht_mlp_pair_vq_targets: tuple[str, ...] = (
        "mlp.c_fc",
        "mlp.c_proj",
    )
    block_fht_mlp_pair_vq_seed: int = 20261020
    block_fht_mlp_pair_vq_neighbor_candidates: int = 16
    block_fht_mlp_pair_vq_code_refresh_interval: int = 8
    block_fht_mlp_pair_vq_lazy_retraction_interval: int = 1
    block_fht_mlp_pair_vq_lazy_retraction_forced_steps: tuple[int, ...] = ()
    block_fht_mlp_pair_vq_error_feedback: bool = False
    block_fht_mlp_pair_vq_hierarchical_feedback_fit: bool = False
    block_fht_mlp_pair_vq_forward_visible_feedback: bool = False
    block_fht_mlp_pair_vq_fp16_ambient_momentum: bool = False
    block_fht_mlp_pair_vq_fp16_reserved_escape_granularity: str = ""
    block_fht_mlp_pair_vq_fp16_ambient_reference_probe_steps: tuple[int, ...] = ()
    block_fht_mlp_pair_vq_cproj_fast_residual: bool = False
    block_fht_mlp_pair_vq_stochastic_fast_retraction: bool = False
    block_fht_mlp_pair_vq_stochastic_fast_fht_block_size: int = 0
    block_fht_mlp_pair_vq_stochastic_fast_uniform_levels: bool = False
    block_fht_mlp_pair_vq_stochastic_fast_block_local_levels: bool = False
    block_fht_mlp_pair_vq_feedback_codec: str = "cartesian4x4"
    block_fht_mlp_pair_vq_feedback_output_group_size: int = 0
    block_fht_mlp_pair_vq_feedback_residual_probe_steps: tuple[int, ...] = ()
    block_fht_mlp_pair_vq_feedback_residual_probe_layers: tuple[int, ...] = ()
    block_fht_mlp_pair_vq_feedback_residual_probe_lloyd_iterations: tuple[int, ...] = ()
    block_fht_mlp_pair_vq_feedback_transform_probe_block_sizes: tuple[int, ...] = ()
    block_fht_mlp_pair_vq_feedback_lattice_probe_block_sizes: tuple[int, ...] = ()
    block_fht_mlp_pair_vq_feedback_lattice_probe_coordinate_bits: tuple[int, ...] = ()
    block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_block_size: int = 0
    block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_coordinate_bits: int = 7
    block_fht_mlp_pair_vq_feedback_fractional_probe_block_size: int = 0
    block_fht_mlp_pair_vq_feedback_fractional_probe_base_coordinate_bits: int = 7
    block_fht_mlp_pair_vq_feedback_fractional_probe_refinement_fractions: tuple[float, ...] = ()
    block_fht_mlp_cproj_muon_matched_givens: bool = False
    block_fht_mlp_cproj_muon_matched_givens_layers: tuple[int, ...] = ()
    block_fht_mlp_cproj_muon_matched_givens_stages: int = 32
    block_fht_mlp_cproj_muon_matched_givens_residual_stages: int = 0
    block_fht_mlp_cproj_muon_matched_givens_output_stages: int = 0
    block_fht_mlp_cproj_muon_matched_givens_neighbors: int = 64
    block_fht_mlp_cproj_muon_matched_givens_refresh_interval: int = 60
    block_fht_mlp_cproj_muon_matched_givens_fast_fresh: bool = False
    block_fht_mlp_cproj_muon_matched_givens_seed: int = 161803
    block_fht_mlp_cproj_muon_matched_givens_error_feedback: bool = False
    block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay: float = 1.0
    block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps: float | None = None
    block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay_after: float | None = None
    block_fht_mlp_cproj_muon_matched_givens_error_feedback_switch_fraction: float | None = None
    block_fht_mlp_cproj_activation_energy_metric: bool = False
    block_fht_mlp_cproj_activation_energy_metric_decay: float = 0.95
    block_fht_mlp_cproj_activation_energy_metric_minimum: float = 0.25
    block_fht_mlp_cproj_activation_energy_metric_maximum: float = 4.0
    block_fht_mlp_cproj_activation_energy_metric_epsilon: float = 1e-6
    block_fht_mlp_cproj_output_symmetric_shear_stages: int = 0
    block_fht_mlp_cproj_output_symmetric_shear_neighbors: int = 64
    block_fht_mlp_cproj_output_symmetric_shear_max_condition_number: float = 1.1
    block_fht_mlp_cproj_global_log_volume: bool = False
    block_fht_mlp_cproj_global_log_volume_max_abs: float = math.log(1.01)
    block_fht_mlp_cproj_hybrid_output: bool = False
    block_fht_mlp_cproj_hybrid_task_stages: int = 16
    block_fht_mlp_cproj_hybrid_directed_incoming: int = 8
    block_fht_mlp_cproj_hybrid_control_stages: int = 32
    block_fht_mlp_cproj_hybrid_ridge_ratio: float = 1e-6
    block_fht_mlp_cproj_hybrid_sample_cap: int = 2048
    block_fht_mlp_cfc_functional_shear: bool = False
    block_fht_mlp_cfc_functional_shear_parent_stages: int = 64
    block_fht_mlp_cfc_functional_shear_stages: int = 24
    block_fht_mlp_cfc_functional_shear_neighbors: int = 64
    block_fht_mlp_cfc_functional_shear_beta: float = 0.5
    block_fht_mlp_cfc_functional_shear_weight_norm_projection: bool = False
    block_fht_mlp_cfc_functional_shear_max_condition_number: float = 0.0
    block_fht_mlp_cfc_functional_shear_sample_cap: int = 2048
    block_fht_mlp_cfc_functional_shear_seed: int = 20260820
    block_fht_mlp_cfc_directed_product: bool = False
    block_fht_mlp_cfc_directed_product_schedule: tuple[int, ...] = (30, 29, 29)
    block_fht_mlp_cfc_directed_product_ridge_ratio: float = 1e-6
    block_fht_mlp_cfc_directed_product_chunk_size: int = 256
    block_fht_mlp_cfc_directed_product_family_radius_ratio: float = 0.6589686140591383
    block_fht_mlp_cfc_directed_product_error_feedback: bool = False
    block_fht_mlp_cfc_directed_product_error_feedback_decay: float = 1.0
    block_fht_mlp_muon_momentum_state_dtype: str = "float32"
    block_fht_mlp_error_feedback_state_codec: str = "float32"
    block_fht_mlp_error_feedback_state_block_size: int = 4096
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
    block_fht_mlp_paired_monarch_block_width: int = 0
    block_fht_mlp_paired_monarch_coordinate_scale: float = 1.0
    block_fht_mlp_paired_monarch_seed: int = 20260819
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
    mlp_shared_dense_trunk: bool = False
    mlp_shared_dense_trunk_groups: int = 1
    mlp_shared_dense_trunk_boundaries: tuple[int, ...] = ()
    mlp_shared_dense_tri_monarch_block_width: int = 0
    mlp_shared_dense_tri_monarch_coordinate_scale: float = 1.0
    mlp_shared_dense_tri_monarch_seed: int = 20260819
    mlp_shared_dense_block_fht_residual: bool = False
    mlp_shared_dense_block_fht_residual_scale: float = math.sqrt(0.5)
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


class LearnedMonarchHiddenMix(nn.Module):
    """Identity-initialized two-factor Monarch transport.

    The same hidden-space operator is folded into both MLP matrices.  For row
    activations ``h``, the paired function is

    ``c_proj(U.T @ gelu(U @ c_fc(x)))``.

    Each factor is block diagonal and the fixed permutation between factors
    provides global channel exchange.  Coordinates are stored as one vector
    so Muon does not mistake the small blocks for independent dense weights.
    """

    def __init__(
        self,
        features: int,
        block_width: int,
        seed: int,
        coordinate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.features = int(features)
        self.block_width = int(block_width)
        self.coordinate_scale = float(coordinate_scale)
        if self.features <= 0:
            raise ValueError("Monarch features must be positive")
        if (
            self.block_width <= 1
            or self.features % self.block_width
        ):
            raise ValueError(
                "Monarch block width must be > 1 and divide features"
            )
        if (
            not math.isfinite(self.coordinate_scale)
            or self.coordinate_scale <= 0.0
        ):
            raise ValueError(
                "Monarch coordinate scale must be positive and finite"
            )
        self.block_count = self.features // self.block_width
        self.coordinates = nn.Parameter(
            torch.zeros(
                2 * self.block_count * self.block_width * self.block_width
            )
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutation = torch.randperm(
            self.features, generator=generator, device="cpu"
        )
        self.register_buffer("permutation", permutation, persistent=True)
        self.register_buffer(
            "inverse_permutation",
            torch.argsort(permutation),
            persistent=True,
        )

    def _blocks(self, values: torch.Tensor) -> torch.Tensor:
        coordinates = self.coordinates.view(
            2,
            self.block_count,
            self.block_width,
            self.block_width,
        ).to(device=values.device, dtype=values.dtype)
        identity = torch.eye(
            self.block_width, device=values.device, dtype=values.dtype
        ).view(1, 1, self.block_width, self.block_width)
        return identity + self.coordinate_scale * coordinates

    def _apply_blocks(
        self, values: torch.Tensor, blocks: torch.Tensor
    ) -> torch.Tensor:
        grouped = values.reshape(
            *values.shape[:-1], self.block_count, self.block_width
        )
        return torch.einsum("...bi,bij->...bj", grouped, blocks).reshape_as(
            values
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.features:
            raise ValueError(
                f"expected last dimension {self.features}, "
                f"got {values.shape[-1]}"
            )
        first, second = self._blocks(values).unbind(0)
        result = values.index_select(-1, self.inverse_permutation)
        result = self._apply_blocks(result, first)
        result = result.index_select(-1, self.permutation)
        return self._apply_blocks(result, second)

    def apply_transpose(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the exact transpose of :meth:`forward` to row vectors."""

        if values.shape[-1] != self.features:
            raise ValueError(
                f"expected last dimension {self.features}, "
                f"got {values.shape[-1]}"
            )
        first, second = self._blocks(values).unbind(0)
        result = self._apply_blocks(values, second.transpose(-1, -2))
        result = result.index_select(-1, self.inverse_permutation)
        result = self._apply_blocks(result, first.transpose(-1, -2))
        return result.index_select(-1, self.permutation)


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
        matrix_parameters: bool = False,
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
            device="cpu",
        )
        right = F.normalize(right, dim=0)
        parameter_shape = (
            (self.features, self.rank)
            if matrix_parameters
            else (self.features * self.rank,)
        )
        self.left = nn.Parameter(
            torch.zeros(parameter_shape, device=right.device)
        )
        self.right = nn.Parameter(right.reshape(parameter_shape))

        identity = torch.eye(
            self.rank, dtype=torch.float32, device=right.device
        )
        zero = torch.zeros_like(identity)
        symplectic = torch.cat(
            (
                torch.cat((zero, identity), dim=1),
                torch.cat((-identity, zero), dim=1),
            ),
            dim=0,
        )
        self.register_buffer("symplectic", symplectic, persistent=False)

    def _factors_and_middle(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return factors.to(dtype=values.dtype), middle.to(dtype=values.dtype)

    def _apply_cayley(
        self, values: torch.Tensor, *, transpose: bool
    ) -> torch.Tensor:
        factors, middle = self._factors_and_middle(values)
        if transpose:
            middle = middle.transpose(0, 1)
        projected = values @ factors
        correction = (
            (projected @ middle)
            @ factors.transpose(0, 1)
        )
        return values + 2.0 * correction

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self._apply_cayley(values, transpose=False)

    def apply_transpose(self, values: torch.Tensor) -> torch.Tensor:
        """Right-multiply row vectors by the exact transposed Cayley map."""

        return self._apply_cayley(values, transpose=True)


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
        shared_mlp_residual = (
            config.mlp_shared_dense_block_fht_residual
            and target_name in {"mlp.c_fc", "mlp.c_proj"}
        )
        if affine_delta and config.block_fht_residual_base_scale != 0.0:
            raise ValueError(
                "target-selective affine deltas cannot be combined with the "
                "legacy global residual base"
            )
        if shared_mlp_residual:
            residual_base_scale = float(
                config.mlp_shared_dense_block_fht_residual_scale
            )
            if (
                not math.isfinite(residual_base_scale)
                or not 0.0 < residual_base_scale < 1.0
            ):
                raise ValueError(
                    "shared dense BlockFHT residual scale must be finite and in (0, 1)"
                )
            residual_base_std = target_std * math.sqrt(
                1.0 - residual_base_scale * residual_base_scale
            )
        else:
            residual_base_scale = (
                float(config.block_fht_affine_delta_scale)
                if affine_delta
                else float(config.block_fht_residual_base_scale)
            )
            residual_base_std = target_std
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
            residual_base_std=residual_base_std,
            residual_base_trainable=shared_mlp_residual,
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
        self.pack_cached_qkv = bool(config.block_fht_attn_pack_cached_qkv)
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
        attention_muon_targets = set(
            config.block_fht_attn_muon_matched_givens_targets
        )
        supported_attention_muon_targets = {
            "attn.c_attn.qk",
            "attn.c_attn.v",
            "attn.c_proj",
        }
        unknown_attention_muon_targets = (
            attention_muon_targets - supported_attention_muon_targets
        )
        if unknown_attention_muon_targets:
            raise ValueError(
                "unsupported attention Muon-matched Givens targets: "
                + ", ".join(sorted(unknown_attention_muon_targets))
            )
        if not attention_muon_targets.issubset(
            set(config.block_fht_targets)
        ):
            raise ValueError(
                "attention Muon-matched Givens targets must also be "
                "BlockFHT targets"
            )
        if (
            attention_muon_targets
            and not config.block_fht_attn_muon_matched_givens_fast_matching
        ):
            raise ValueError(
                "bilateral attention Muon-matched Givens requires the "
                "native fast matcher"
            )
        if config.block_fht_attn_cproj_int8_lattice:
            if "attn.c_proj" in attention_muon_targets:
                raise ValueError(
                    "attention c_proj cannot use both int8 lattice and "
                    "Muon-matched Givens"
                )
            if "attn.c_proj" in config.block_fht_targets:
                raise ValueError(
                    "attention c_proj int8 lattice is the c_proj target; "
                    "remove attn.c_proj from BlockFHT targets"
                )
        elif config.block_fht_attn_cproj_int8_lattice_error_feedback:
            raise ValueError(
                "attention c_proj int8 lattice error feedback requires "
                "the c_proj lattice"
            )
        if config.block_fht_attn_v_int8_lattice:
            if structured == 0:
                raise ValueError(
                    "attention V int8 lattice requires a split or structured "
                    "QKV attention family"
                )
            if "attn.c_attn.v" in attention_muon_targets:
                raise ValueError(
                    "attention V cannot use both int8 lattice and "
                    "Muon-matched Givens"
                )
            if "attn.c_attn.v" in config.block_fht_targets:
                raise ValueError(
                    "attention V int8 lattice is the V target; remove "
                    "attn.c_attn.v from BlockFHT targets"
                )

        attention_pair_vq = bool(config.block_fht_attn_pair_vq)
        if attention_pair_vq:
            if structured == 0:
                raise ValueError(
                    "attention Pair-VQ requires a split or structured QKV family"
                )
            if (
                config.block_fht_attn_v_int8_lattice
                or config.block_fht_attn_cproj_int8_lattice
            ):
                raise ValueError(
                    "attention Pair-VQ cannot be combined with attention int8 lattices"
                )
            conflicting_targets = {
                "attn.c_attn.v",
                "attn.c_proj",
            } & set(config.block_fht_targets)
            if conflicting_targets:
                raise ValueError(
                    "attention Pair-VQ is the complete V/output-projection "
                    "representation; remove: "
                    + ", ".join(sorted(conflicting_targets))
                )
            conflicting_muon_targets = {
                "attn.c_attn.v",
                "attn.c_proj",
            } & attention_muon_targets
            if conflicting_muon_targets:
                raise ValueError(
                    "attention Pair-VQ cannot be combined with attention "
                    "Muon-matched Givens for: "
                    + ", ".join(sorted(conflicting_muon_targets))
                )

        def attention_linear(
            in_features: int,
            out_features: int,
            bias: bool,
            target_name: str,
            seed_offset: int,
        ) -> nn.Module:
            if attention_pair_vq and target_name in {
                "attn.c_attn.v",
                "attn.c_proj",
            }:
                is_value = target_name == "attn.c_attn.v"
                return MuonPairVQLinear(
                    in_features,
                    out_features,
                    bias=bias,
                    stages=2 if is_value else 1,
                    base_seed=(
                        int(config.block_fht_attn_pair_vq_seed)
                        + layer_id * 8192
                        + (0 if is_value else 4096)
                    ),
                    weight_std=(
                        0.02
                        if is_value
                        else 0.02 / math.sqrt(2 * config.n_layer)
                    ),
                    layer_id=layer_id,
                    fast_residual=True,
                    stochastic_fast_retraction=bool(
                        config.block_fht_mlp_pair_vq_stochastic_fast_retraction
                    ),
                    stochastic_fast_fht_block_size=int(
                        config.block_fht_mlp_pair_vq_stochastic_fast_fht_block_size
                    ),
                    stochastic_fast_uniform_levels=bool(
                        config.block_fht_mlp_pair_vq_stochastic_fast_uniform_levels
                    ),
                    stochastic_fast_block_local_levels=bool(
                        config.block_fht_mlp_pair_vq_stochastic_fast_block_local_levels
                    ),
                    error_feedback=bool(
                        config.block_fht_mlp_pair_vq_error_feedback
                    ),
                    forward_visible_feedback=bool(
                        config.block_fht_mlp_pair_vq_forward_visible_feedback
                    ),
                    fp16_ambient_momentum=bool(
                        config.block_fht_mlp_pair_vq_fp16_ambient_momentum
                    ),
                    fp16_reserved_escape_granularity=str(
                        config.block_fht_mlp_pair_vq_fp16_reserved_escape_granularity
                    ),
                    reserved_escape_scope="c_fc" if is_value else "c_proj",
                    fp16_ambient_reference_probe_steps=tuple(
                        config.block_fht_mlp_pair_vq_fp16_ambient_reference_probe_steps
                    ),
                    feedback_codec=str(
                        config.block_fht_mlp_pair_vq_feedback_codec
                    ),
                    feedback_output_group_size=int(
                        config.block_fht_mlp_pair_vq_feedback_output_group_size
                    ),
                    feedback_residual_probe_steps=tuple(
                        config.block_fht_mlp_pair_vq_feedback_residual_probe_steps
                    ),
                    feedback_residual_probe_layers=tuple(
                        config.block_fht_mlp_pair_vq_feedback_residual_probe_layers
                    ),
                    feedback_residual_probe_lloyd_iterations=tuple(
                        config.block_fht_mlp_pair_vq_feedback_residual_probe_lloyd_iterations
                    ),
                    feedback_transform_probe_block_sizes=tuple(
                        config.block_fht_mlp_pair_vq_feedback_transform_probe_block_sizes
                    ),
                    feedback_lattice_probe_block_sizes=tuple(
                        config.block_fht_mlp_pair_vq_feedback_lattice_probe_block_sizes
                    ),
                    feedback_lattice_probe_coordinate_bits=tuple(
                        config.block_fht_mlp_pair_vq_feedback_lattice_probe_coordinate_bits
                    ),
                    feedback_axis_adaptation_probe_block_size=int(
                        config.block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_block_size
                    ),
                    feedback_axis_adaptation_probe_coordinate_bits=int(
                        config.block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_coordinate_bits
                    ),
                    feedback_fractional_probe_block_size=int(
                        config.block_fht_mlp_pair_vq_feedback_fractional_probe_block_size
                    ),
                    feedback_fractional_probe_base_coordinate_bits=int(
                        config.block_fht_mlp_pair_vq_feedback_fractional_probe_base_coordinate_bits
                    ),
                    feedback_fractional_probe_refinement_fractions=tuple(
                        config.block_fht_mlp_pair_vq_feedback_fractional_probe_refinement_fractions
                    ),
                    neighbor_candidates=int(
                        config.block_fht_mlp_pair_vq_neighbor_candidates
                    ),
                    code_refresh_interval=int(
                        config.block_fht_mlp_pair_vq_code_refresh_interval
                    ),
                    lazy_retraction_interval=int(
                        config.block_fht_mlp_pair_vq_lazy_retraction_interval
                    ),
                    lazy_retraction_forced_steps=tuple(
                        config.block_fht_mlp_pair_vq_lazy_retraction_forced_steps
                    ),
                )
            if (
                target_name == "attn.c_attn.v"
                and config.block_fht_attn_v_int8_lattice
            ):
                return MuonInt8LatticeLinear(
                    in_features,
                    out_features,
                    bias=bias,
                    block_size=int(
                        config.block_fht_attn_v_int8_lattice_block_size
                    ),
                    base_seed=(
                        int(config.block_fht_attn_v_int8_lattice_seed)
                        + layer_id * 4096
                    ),
                    weight_std=0.02,
                    layer_id=layer_id,
                    error_feedback=bool(
                        config.block_fht_attn_v_int8_lattice_error_feedback
                    ),
                )
            if (
                target_name == "attn.c_proj"
                and config.block_fht_attn_cproj_int8_lattice
            ):
                return MuonInt8LatticeLinear(
                    in_features,
                    out_features,
                    bias=bias,
                    block_size=int(
                        config.block_fht_attn_cproj_int8_lattice_block_size
                    ),
                    base_seed=(
                        int(config.block_fht_attn_cproj_int8_lattice_seed)
                        + layer_id * 4096
                    ),
                    weight_std=0.02 / math.sqrt(2 * config.n_layer),
                    layer_id=layer_id,
                    error_feedback=bool(
                        config
                        .block_fht_attn_cproj_int8_lattice_error_feedback
                    ),
                )
            if target_name not in attention_muon_targets:
                return make_linear(
                    in_features,
                    out_features,
                    bias,
                    config,
                    target_name,
                    seed_offset,
                )
            stages = int(
                config.block_fht_attn_muon_matched_givens_stages
            )
            target_seed_offset = {
                "attn.c_attn.qk": 0,
                "attn.c_attn.v": 256,
                # The c-proj oracle has output as side index zero.  The
                # module adds the shared output-side offset below.
                "attn.c_proj": 384,
            }[target_name]
            return MuonMatchedGivensLinear(
                in_features,
                out_features,
                bias=bias,
                stages=(0 if target_name == "attn.c_proj" else stages),
                residual_stages=0,
                output_stages=stages,
                neighbors=int(
                    config.block_fht_attn_muon_matched_givens_neighbors
                ),
                refresh_interval=int(
                    config
                    .block_fht_attn_muon_matched_givens_refresh_interval
                ),
                fast_fresh_matching=bool(
                    config
                    .block_fht_attn_muon_matched_givens_fast_matching
                ),
                matching_seed=(
                    int(config.block_fht_attn_muon_matched_givens_seed)
                    + layer_id * 4096
                    + target_seed_offset
                ),
                matching_seed_step_stride=int(
                    config
                    .block_fht_attn_muon_matched_givens_seed_step_stride
                ),
                output_matching_seed_offset=128,
                weight_std=(
                    0.02 / math.sqrt(2 * config.n_layer)
                    if target_name == "attn.c_proj"
                    else 0.02
                ),
                layer_id=layer_id,
            )
        if self.split_c_attn:
            if "attn.c_attn" in config.block_fht_targets:
                raise ValueError("Use either monolithic attn.c_attn or split attn.c_attn.{q,k,v}, not both")
            self.c_attn = None
            self.c_attn_q = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.q", layer_id * 8)
            self.c_attn_k = make_linear(config.n_embd, config.n_embd, config.bias, config, "attn.c_attn.k", layer_id * 8 + 1)
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
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
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
            self.c_attn_qk = attention_linear(config.n_embd, 2 * config.n_embd, config.bias, "attn.c_attn.qk", layer_id * 8)
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
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
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
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
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
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
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
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
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
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
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
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
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
            self.c_attn_v = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_attn.v", layer_id * 8 + 2)
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
        self.c_proj = attention_linear(config.n_embd, config.n_embd, config.bias, "attn.c_proj", layer_id * 4 + 1)
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
        cayley_factor_optimizer = str(
            config.block_fht_attn_cayley_factor_optimizer
        )
        if cayley_factor_optimizer not in {
            "adamw",
            "muon",
            "hybrid_left_muon",
        }:
            raise ValueError(
                "attention Cayley factor optimizer must be adamw, muon, "
                "or hybrid_left_muon"
            )
        atlas_start_steps = tuple(
            int(step)
            for step in config.block_fht_attn_cayley_atlas_start_steps
        )
        if atlas_start_steps:
            if not cayley_targets:
                raise ValueError(
                    "attention Cayley atlas requires enabled Cayley targets"
                )
            if atlas_start_steps[0] != 0:
                raise ValueError(
                    "attention Cayley atlas must start with step 0"
                )
            if any(step < 0 for step in atlas_start_steps) or any(
                later <= earlier
                for earlier, later in zip(
                    atlas_start_steps, atlas_start_steps[1:]
                )
            ):
                raise ValueError(
                    "attention Cayley atlas start steps must be strictly "
                    "increasing nonnegative integers"
                )
        self.cayley_atlas_start_steps = atlas_start_steps
        self.active_cayley_atlas_stage = max(len(atlas_start_steps) - 1, 0)
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
            stage: int = 0,
        ) -> LearnedLowRankCayleyMix:
            return LearnedLowRankCayleyMix(
                features,
                cayley_ranks[target],
                int(config.block_fht_attn_cayley_seed)
                + layer_id * 64
                + seed_offset
                + stage * 104729,
                coordinate_scale=float(config.block_fht_attn_cayley_scale),
                matrix_parameters=cayley_factor_optimizer != "adamw",
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

        def cayley_atlas(
            primary: LearnedLowRankCayleyMix | None,
            features: int,
            seed_offset: int,
            target: str,
        ) -> nn.ModuleList:
            if primary is None or len(atlas_start_steps) <= 1:
                return nn.ModuleList()
            return nn.ModuleList(
                [
                    cayley_mix(
                        features,
                        seed_offset,
                        target,
                        stage=stage,
                    )
                    for stage in range(1, len(atlas_start_steps))
                ]
            )

        self.qk_input_cayley_atlas = cayley_atlas(
            self.qk_input_cayley,
            config.n_embd,
            0,
            "attn.c_attn.qk_headwise",
        )
        self.qk_output_cayley_atlas = cayley_atlas(
            self.qk_output_cayley,
            2 * config.n_embd,
            3,
            "attn.c_attn.qk_headwise",
        )
        self.v_input_cayley_atlas = cayley_atlas(
            self.v_input_cayley,
            config.n_embd,
            1,
            "attn.c_attn.v",
        )
        self.v_output_cayley_atlas = cayley_atlas(
            self.v_output_cayley,
            config.n_embd,
            4,
            "attn.c_attn.v",
        )
        self.cproj_input_cayley_atlas = cayley_atlas(
            self.cproj_input_cayley,
            config.n_embd,
            2,
            "attn.c_proj",
        )
        self.cproj_output_cayley_atlas = cayley_atlas(
            self.cproj_output_cayley,
            config.n_embd,
            5,
            "attn.c_proj",
        )
        if self.pack_cached_qkv:
            if not self.qk_headwise_c_attn:
                raise ValueError(
                    "packed cached QKV requires attn.c_attn.qk_headwise"
                )
            if not isinstance(self.c_attn_v, nn.Linear):
                raise ValueError("packed cached QKV requires dense V")
            if self.v_input_cayley is not None or self.v_output_cayley is not None:
                raise ValueError("packed cached QKV requires uncharted dense V")
            if config.bias:
                raise ValueError("packed cached QKV currently requires bias=False")
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

    @staticmethod
    def _apply_cayley_atlas(
        values: torch.Tensor,
        primary: LearnedLowRankCayleyMix | None,
        atlas: nn.ModuleList,
        active_stage: int,
    ) -> torch.Tensor:
        result = primary(values) if primary is not None else values
        for stage, chart in enumerate(atlas, start=1):
            if stage > active_stage:
                break
            result = chart(result)
        return result

    @staticmethod
    def _active_cayley_charts(
        primary: LearnedLowRankCayleyMix | None,
        atlas: nn.ModuleList,
        active_stage: int,
    ) -> tuple[LearnedLowRankCayleyMix, ...]:
        charts = [] if primary is None else [primary]
        charts.extend(
            chart
            for stage, chart in enumerate(atlas, start=1)
            if stage <= active_stage
        )
        return tuple(charts)

    def _fold_cached_qk_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Legacy QK is x C_in W^T C_out.  F.linear therefore needs
        # W_eff = C_out^T W C_in^T.  Atlases compose in forward order.
        input_charts = self._active_cayley_charts(
            self.qk_input_cayley,
            self.qk_input_cayley_atlas,
            self.active_cayley_atlas_stage,
        )
        for chart in reversed(input_charts):
            weight = chart.apply_transpose(weight)

        output_charts = self._active_cayley_charts(
            self.qk_output_cayley,
            self.qk_output_cayley_atlas,
            self.active_cayley_atlas_stage,
        )
        transposed = weight.transpose(0, 1)
        for chart in output_charts:
            transposed = chart(transposed)
        return transposed.transpose(0, 1)

    def _project_packed_cached_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.c_attn_qk_headwise is not None
        assert isinstance(self.c_attn_v, nn.Linear)
        cached_weights = [
            getattr(head, "_cached_weight", None)
            for head in self.c_attn_qk_headwise.heads
        ]
        if not cached_weights or any(weight is None for weight in cached_weights):
            raise RuntimeError(
                "packed cached QKV requires a prepared BlockFHT weight cache"
            )
        qk_weight = torch.cat(cached_weights, dim=0)
        qk_weight = self._fold_cached_qk_weight(qk_weight)
        packed_weight = torch.cat((qk_weight, self.c_attn_v.weight), dim=0)
        qk, v = F.linear(x, packed_weight).split(
            (2 * self.n_embd, self.n_embd), dim=2
        )
        q, k = qk.split(self.n_embd, dim=2)
        return q, k, v

    def attention_cayley_stage_modules(
        self,
    ) -> tuple[tuple[LearnedLowRankCayleyMix, ...], ...]:
        if not self.cayley_atlas_start_steps:
            return ()
        primary = tuple(
            chart
            for chart in (
                self.qk_input_cayley,
                self.qk_output_cayley,
                self.v_input_cayley,
                self.v_output_cayley,
                self.cproj_input_cayley,
                self.cproj_output_cayley,
            )
            if chart is not None
        )
        atlases = (
            self.qk_input_cayley_atlas,
            self.qk_output_cayley_atlas,
            self.v_input_cayley_atlas,
            self.v_output_cayley_atlas,
            self.cproj_input_cayley_atlas,
            self.cproj_output_cayley_atlas,
        )
        return (primary,) + tuple(
            tuple(atlas[stage] for atlas in atlases if len(atlas) > stage)
            for stage in range(len(self.cayley_atlas_start_steps) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.size()
        if self.pack_cached_qkv:
            q, k, v = self._project_packed_cached_qkv(x)
        elif self.split_c_attn:
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
            qk_input = self._apply_cayley_atlas(
                x,
                self.qk_input_cayley,
                self.qk_input_cayley_atlas,
                self.active_cayley_atlas_stage,
            )
            v_input = self._apply_cayley_atlas(
                x,
                self.v_input_cayley,
                self.v_input_cayley_atlas,
                self.active_cayley_atlas_stage,
            )
            qk = self.c_attn_qk_headwise(qk_input)
            qk = self._apply_cayley_atlas(
                qk,
                self.qk_output_cayley,
                self.qk_output_cayley_atlas,
                self.active_cayley_atlas_stage,
            )
            q, k = qk.split(self.n_embd, dim=2)
            v = self.c_attn_v(v_input)
            v = self._apply_cayley_atlas(
                v,
                self.v_output_cayley,
                self.v_output_cayley_atlas,
                self.active_cayley_atlas_stage,
            )
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
        y = self._apply_cayley_atlas(
            y,
            self.cproj_input_cayley,
            self.cproj_input_cayley_atlas,
            self.active_cayley_atlas_stage,
        )
        y = self.c_proj(y)
        y = self._apply_cayley_atlas(
            y,
            self.cproj_output_cayley,
            self.cproj_output_cayley_atlas,
            self.active_cayley_atlas_stage,
        )
        return self.resid_dropout(y)


class SparseMoEMLP(nn.Module):
    """Dropless token-choice MoE with complete paired GELU experts.

    Both expert matrices are stored in batched parameters and executed with
    two batched GEMMs after deterministic token packing.  There is no Python
    expert loop and no shared post-mixture projection.
    """

    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.n_embd = int(config.n_embd)
        self.num_experts = int(config.moe_num_experts)
        self.top_k = int(config.moe_top_k)
        self.hidden_features = int(
            config.moe_expert_hidden_multiplier * config.n_embd
        )
        self.unpadded_expert_loop = bool(
            config.moe_unpadded_expert_loop
        )
        if self.num_experts < 2:
            raise ValueError("sparse MoE requires at least two experts")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts]")
        if self.hidden_features <= 0:
            raise ValueError("MoE expert hidden width must be positive")
        for name, value in (
            (
                "moe_load_balance_aux_coefficient",
                config.moe_load_balance_aux_coefficient,
            ),
            (
                "moe_router_z_loss_coefficient",
                config.moe_router_z_loss_coefficient,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)
        self.expert_c_fc = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_features, self.n_embd)
        )
        self.expert_c_proj = nn.Parameter(
            torch.empty(self.num_experts, self.n_embd, self.hidden_features)
        )
        # Muon must orthogonalize each expert matrix independently, not flatten
        # the expert axis into one unrelated rectangular matrix.
        self.expert_c_fc._muon_batched_matrices = True
        self.expert_c_proj._muon_batched_matrices = True
        nn.init.normal_(self.expert_c_fc, mean=0.0, std=0.02)
        nn.init.normal_(
            self.expert_c_proj,
            mean=0.0,
            std=0.02 / math.sqrt(2 * config.n_layer),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.last_load_balance_loss: torch.Tensor | None = None
        self.last_router_z_loss: torch.Tensor | None = None
        self.last_expert_counts: torch.Tensor | None = None
        self.last_assignment_count = 0
        self.last_max_tokens_per_expert: torch.Tensor | None = None

    def _route(
        self, flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(flat).float()
        # Use a selection-only expert-index tiebreak. Mixture probabilities are
        # still computed from the unmodified selected logits.
        tie = torch.arange(
            self.num_experts, device=logits.device, dtype=logits.dtype
        )
        selection_logits = logits - tie * torch.finfo(logits.dtype).eps
        _values, expert_indices = torch.topk(
            selection_logits,
            self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        selected_logits = logits.gather(-1, expert_indices)
        selected_probabilities = F.softmax(selected_logits, dim=-1).to(
            dtype=flat.dtype
        )
        return logits, expert_indices, selected_probabilities

    def _router_losses(
        self, logits: torch.Tensor, expert_indices: torch.Tensor
    ) -> None:
        probabilities = F.softmax(logits, dim=-1)
        importance = probabilities.mean(dim=0)
        load = F.one_hot(
            expert_indices, num_classes=self.num_experts
        ).float().mean(dim=(0, 1))
        self.last_load_balance_loss = self.num_experts * torch.sum(
            importance * load
        )
        self.last_router_z_loss = torch.logsumexp(logits, dim=-1).square().mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.n_embd)
        logits, expert_indices, selected_probabilities = self._route(flat)
        self._router_losses(logits, expert_indices)

        assignments = expert_indices.numel()
        token_indices = torch.arange(
            flat.shape[0], device=flat.device, dtype=torch.long
        ).repeat_interleave(self.top_k)
        flat_experts = expert_indices.reshape(-1)
        flat_probabilities = selected_probabilities.reshape(-1)
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts.index_select(0, order)
        sorted_tokens = token_indices.index_select(0, order)
        sorted_probabilities = flat_probabilities.index_select(0, order)
        counts = torch.bincount(flat_experts, minlength=self.num_experts)
        offsets = counts.cumsum(0) - counts
        positions = torch.arange(
            assignments, device=flat.device, dtype=torch.long
        ) - offsets.index_select(0, sorted_experts)
        max_tokens = counts.max()

        if self.unpadded_expert_loop:
            # stable=True above makes every expert's real assignments one
            # contiguous slice. The count transfer is one synchronization per
            # layer, after which no zero-capacity expert rows are executed.
            count_values = [int(value) for value in counts.tolist()]
            sorted_output_parts = []
            cursor = 0
            for expert, count in enumerate(count_values):
                if count == 0:
                    continue
                stop = cursor + count
                expert_input = flat.index_select(
                    0, sorted_tokens[cursor:stop]
                )
                hidden = F.gelu(
                    F.linear(expert_input, self.expert_c_fc[expert])
                )
                sorted_output_parts.append(
                    F.linear(hidden, self.expert_c_proj[expert])
                )
                cursor = stop
            if cursor != assignments:
                raise RuntimeError("sparse expert assignment accounting drift")
            sorted_output = torch.cat(sorted_output_parts, dim=0)
        else:
            # Zero padding is exact because experts are bias-free and
            # GELU(0)=0, but can execute far more rows than were routed.
            packed = flat.new_zeros(
                (self.num_experts, max_tokens, self.n_embd)
            )
            packed[sorted_experts, positions] = flat.index_select(
                0, sorted_tokens
            )
            hidden = torch.bmm(
                packed, self.expert_c_fc.transpose(1, 2)
            )
            hidden = F.gelu(hidden)
            expert_output = torch.bmm(
                hidden, self.expert_c_proj.transpose(1, 2)
            )
            sorted_output = expert_output[sorted_experts, positions]
        output = flat.new_zeros(flat.shape)
        output.index_add_(
            0,
            sorted_tokens,
            sorted_output * sorted_probabilities.unsqueeze(-1),
        )

        self.last_expert_counts = counts.detach()
        self.last_assignment_count = int(assignments)
        self.last_max_tokens_per_expert = max_tokens.detach()
        return self.dropout(output.reshape(original_shape))

    def router_auxiliary_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.last_load_balance_loss is None or self.last_router_z_loss is None:
            zero = self.expert_c_fc.new_zeros(())
            return zero, zero
        return self.last_load_balance_loss, self.last_router_z_loss

    def postgelu_spread_loss(self) -> None:
        return None

    def cproj_teacher_alignment_loss(self) -> None:
        return None


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        mlp_int8_lattice_targets = set(
            config.block_fht_mlp_int8_lattice_targets
        )
        pair_vq = bool(config.block_fht_mlp_pair_vq)
        pair_vq_targets = set(config.block_fht_mlp_pair_vq_targets)
        unsupported_pair_vq_targets = pair_vq_targets - {
            "mlp.c_fc",
            "mlp.c_proj",
        }
        if unsupported_pair_vq_targets:
            raise ValueError(
                "unsupported MLP pair-VQ targets: "
                + ", ".join(sorted(unsupported_pair_vq_targets))
            )
        pair_vq_cfc = pair_vq and "mlp.c_fc" in pair_vq_targets
        pair_vq_cproj = pair_vq and "mlp.c_proj" in pair_vq_targets
        unsupported_mlp_int8_targets = mlp_int8_lattice_targets - {
            "mlp.c_fc",
            "mlp.c_proj",
        }
        if unsupported_mlp_int8_targets:
            raise ValueError(
                "unsupported MLP int8 lattice targets: "
                + ", ".join(sorted(unsupported_mlp_int8_targets))
            )
        grouped_targets = [target for target in MLP_C_FC_GROUP_TARGETS if target in config.block_fht_targets]
        if len(grouped_targets) > 1:
            raise ValueError("Use exactly one grouped mlp.c_fc target per run")
        if grouped_targets and "mlp.c_fc" in config.block_fht_targets:
            raise ValueError("Use either plain mlp.c_fc or grouped mlp.c_fc, not both")
        functional_shear_cfc = bool(
            config.block_fht_mlp_cfc_functional_shear
        )
        directed_product_cfc = bool(
            config.block_fht_mlp_cfc_directed_product
        )
        int8_lattice_cfc = "mlp.c_fc" in mlp_int8_lattice_targets
        if pair_vq_cfc and (
            "mlp.c_fc" in mlp_int8_lattice_targets
            or functional_shear_cfc
            or directed_product_cfc
            or grouped_targets
            or "mlp.c_fc" in config.block_fht_targets
        ):
            raise ValueError(
                "MLP c_fc pair VQ cannot be combined with another c_fc "
                "representation"
            )
        if int8_lattice_cfc and (
            functional_shear_cfc
            or directed_product_cfc
            or grouped_targets
            or "mlp.c_fc" in config.block_fht_targets
        ):
            raise ValueError(
                "MLP c_fc int8 lattice cannot be combined with another "
                "c_fc representation"
            )
        if functional_shear_cfc and directed_product_cfc:
            raise ValueError(
                "functional-shear and directed-product c_fc are mutually exclusive"
            )
        if functional_shear_cfc and (
            grouped_targets or "mlp.c_fc" in config.block_fht_targets
        ):
            raise ValueError(
                "functional-shear c_fc requires the materialized plain "
                "c_fc path"
            )
        if pair_vq_cfc:
            self.c_fc = MuonPairVQLinear(
                config.n_embd,
                4 * config.n_embd,
                bias=config.bias,
                stages=2,
                base_seed=(
                    int(config.block_fht_mlp_pair_vq_seed)
                    + layer_id * 8192
                ),
                weight_std=0.02,
                layer_id=layer_id,
                fast_residual=True,
                stochastic_fast_retraction=bool(
                    config.block_fht_mlp_pair_vq_stochastic_fast_retraction
                ),
                stochastic_fast_fht_block_size=int(
                    config.block_fht_mlp_pair_vq_stochastic_fast_fht_block_size
                ),
                stochastic_fast_uniform_levels=bool(
                    config.block_fht_mlp_pair_vq_stochastic_fast_uniform_levels
                ),
                stochastic_fast_block_local_levels=bool(
                    config.block_fht_mlp_pair_vq_stochastic_fast_block_local_levels
                ),
                error_feedback=bool(
                    config.block_fht_mlp_pair_vq_error_feedback
                ),
                forward_visible_feedback=bool(
                    config.block_fht_mlp_pair_vq_forward_visible_feedback
                ),
                fp16_ambient_momentum=bool(
                    config.block_fht_mlp_pair_vq_fp16_ambient_momentum
                ),
                fp16_reserved_escape_granularity=str(
                    config.block_fht_mlp_pair_vq_fp16_reserved_escape_granularity
                ),
                reserved_escape_scope="c_fc",
                fp16_ambient_reference_probe_steps=tuple(
                    config.block_fht_mlp_pair_vq_fp16_ambient_reference_probe_steps
                ),
                feedback_codec=str(
                    config.block_fht_mlp_pair_vq_feedback_codec
                ),
                feedback_output_group_size=int(
                    config.block_fht_mlp_pair_vq_feedback_output_group_size
                ),
                feedback_residual_probe_steps=tuple(
                    config.block_fht_mlp_pair_vq_feedback_residual_probe_steps
                ),
                feedback_residual_probe_layers=tuple(
                    config.block_fht_mlp_pair_vq_feedback_residual_probe_layers
                ),
                feedback_residual_probe_lloyd_iterations=tuple(
                    config.block_fht_mlp_pair_vq_feedback_residual_probe_lloyd_iterations
                ),
                feedback_transform_probe_block_sizes=tuple(
                    config.block_fht_mlp_pair_vq_feedback_transform_probe_block_sizes
                ),
                feedback_lattice_probe_block_sizes=tuple(
                    config.block_fht_mlp_pair_vq_feedback_lattice_probe_block_sizes
                ),
                feedback_lattice_probe_coordinate_bits=tuple(
                    config.block_fht_mlp_pair_vq_feedback_lattice_probe_coordinate_bits
                ),
                feedback_axis_adaptation_probe_block_size=int(
                    config.block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_block_size
                ),
                feedback_axis_adaptation_probe_coordinate_bits=int(
                    config.block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_coordinate_bits
                ),
                feedback_fractional_probe_block_size=int(
                    config.block_fht_mlp_pair_vq_feedback_fractional_probe_block_size
                ),
                feedback_fractional_probe_base_coordinate_bits=int(
                    config.block_fht_mlp_pair_vq_feedback_fractional_probe_base_coordinate_bits
                ),
                feedback_fractional_probe_refinement_fractions=tuple(
                    config.block_fht_mlp_pair_vq_feedback_fractional_probe_refinement_fractions
                ),
                neighbor_candidates=int(
                    config.block_fht_mlp_pair_vq_neighbor_candidates
                ),
                code_refresh_interval=int(
                    config.block_fht_mlp_pair_vq_code_refresh_interval
                ),
                lazy_retraction_interval=int(
                    config.block_fht_mlp_pair_vq_lazy_retraction_interval
                ),
                lazy_retraction_forced_steps=tuple(
                    config.block_fht_mlp_pair_vq_lazy_retraction_forced_steps
                ),
            )
        elif int8_lattice_cfc:
            self.c_fc = MuonInt8LatticeLinear(
                config.n_embd,
                4 * config.n_embd,
                bias=config.bias,
                block_size=int(config.block_fht_mlp_int8_lattice_block_size),
                base_seed=(
                    int(config.block_fht_mlp_int8_lattice_seed)
                    + layer_id * 8192
                ),
                weight_std=0.02,
                layer_id=layer_id,
                error_feedback=(
                    config.block_fht_mlp_int8_lattice_error_feedback
                ),
            )
        elif directed_product_cfc:
            self.c_fc = MuonDirectedProductLinear(
                config.n_embd,
                4 * config.n_embd,
                bias=config.bias,
                incoming_schedule=tuple(
                    int(value)
                    for value in config.block_fht_mlp_cfc_directed_product_schedule
                ),
                ridge_ratio=float(
                    config.block_fht_mlp_cfc_directed_product_ridge_ratio
                ),
                chunk_size=int(
                    config.block_fht_mlp_cfc_directed_product_chunk_size
                ),
                family_radius_ratio=float(
                    config.block_fht_mlp_cfc_directed_product_family_radius_ratio
                ),
                error_feedback=bool(
                    config.block_fht_mlp_cfc_directed_product_error_feedback
                ),
                error_feedback_decay=float(
                    config
                    .block_fht_mlp_cfc_directed_product_error_feedback_decay
                ),
                weight_std=0.02,
                layer_id=layer_id,
            )
        elif functional_shear_cfc:
            self.c_fc = MuonFunctionalShearLinear(
                config.n_embd,
                4 * config.n_embd,
                bias=config.bias,
                parent_stages=int(
                    config.block_fht_mlp_cfc_functional_shear_parent_stages
                ),
                shear_stages=int(
                    config.block_fht_mlp_cfc_functional_shear_stages
                ),
                neighbors=int(
                    config.block_fht_mlp_cfc_functional_shear_neighbors
                ),
                coordinate_mix_beta=float(
                    config.block_fht_mlp_cfc_functional_shear_beta
                ),
                project_to_weight_norm=bool(
                    config
                    .block_fht_mlp_cfc_functional_shear_weight_norm_projection
                ),
                max_condition_number=(
                    float(
                        config
                        .block_fht_mlp_cfc_functional_shear_max_condition_number
                    )
                    or None
                ),
                functional_sample_cap=int(
                    config.block_fht_mlp_cfc_functional_shear_sample_cap
                ),
                matching_seed=(
                    int(config.block_fht_mlp_cfc_functional_shear_seed)
                    + layer_id * 1009
                ),
                weight_std=0.02,
                layer_id=layer_id,
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
        if structured_proj_count and "mlp.c_proj" in config.block_fht_targets:
            raise ValueError("Use either plain mlp.c_proj or grouped mlp.c_proj, not both")
        configured_cproj_layers = tuple(
            config.block_fht_mlp_cproj_muon_matched_givens_layers
        )
        muon_matched_cproj_enabled = bool(
            config.block_fht_mlp_cproj_muon_matched_givens
        )
        int8_lattice_cproj = "mlp.c_proj" in mlp_int8_lattice_targets
        if pair_vq_cproj and (
            int8_lattice_cproj
            or muon_matched_cproj_enabled
            or structured_proj_count
            or "mlp.c_proj" in config.block_fht_targets
        ):
            raise ValueError(
                "MLP c_proj pair VQ cannot be combined with another c_proj "
                "representation"
            )
        if int8_lattice_cproj and (
            muon_matched_cproj_enabled
            or structured_proj_count
            or "mlp.c_proj" in config.block_fht_targets
        ):
            raise ValueError(
                "MLP c_proj int8 lattice cannot be combined with another "
                "c_proj representation"
            )
        muon_matched_cproj = muon_matched_cproj_enabled and (
            not configured_cproj_layers
            or layer_id in configured_cproj_layers
        )
        dense_lwt_cproj = (
            muon_matched_cproj_enabled
            and bool(configured_cproj_layers)
            and layer_id not in configured_cproj_layers
        )
        if (
            config.block_fht_mlp_cproj_hybrid_output
            and not muon_matched_cproj
        ):
            raise ValueError(
                "hybrid c_proj output requires Muon-matched Givens"
            )
        if (
            config.block_fht_mlp_cproj_output_symmetric_shear_stages
            and not muon_matched_cproj
        ):
            raise ValueError(
                "symmetric-shear c_proj output requires Muon-matched Givens"
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
        if pair_vq_cproj:
            self.c_proj = MuonPairVQLinear(
                4 * config.n_embd,
                config.n_embd,
                bias=config.bias,
                stages=1,
                base_seed=(
                    int(config.block_fht_mlp_pair_vq_seed)
                    + layer_id * 8192
                    + 4096
                ),
                weight_std=0.02 / math.sqrt(2 * config.n_layer),
                layer_id=layer_id,
                fast_residual=bool(
                    config.block_fht_mlp_pair_vq_cproj_fast_residual
                ),
                stochastic_fast_retraction=bool(
                    config.block_fht_mlp_pair_vq_stochastic_fast_retraction
                ),
                stochastic_fast_fht_block_size=int(
                    config.block_fht_mlp_pair_vq_stochastic_fast_fht_block_size
                ),
                stochastic_fast_uniform_levels=bool(
                    config.block_fht_mlp_pair_vq_stochastic_fast_uniform_levels
                ),
                stochastic_fast_block_local_levels=bool(
                    config.block_fht_mlp_pair_vq_stochastic_fast_block_local_levels
                ),
                error_feedback=bool(
                    config.block_fht_mlp_pair_vq_error_feedback
                ),
                forward_visible_feedback=bool(
                    config.block_fht_mlp_pair_vq_forward_visible_feedback
                ),
                fp16_ambient_momentum=bool(
                    config.block_fht_mlp_pair_vq_fp16_ambient_momentum
                ),
                fp16_reserved_escape_granularity=str(
                    config.block_fht_mlp_pair_vq_fp16_reserved_escape_granularity
                ),
                reserved_escape_scope="c_proj",
                fp16_ambient_reference_probe_steps=tuple(
                    config.block_fht_mlp_pair_vq_fp16_ambient_reference_probe_steps
                ),
                feedback_codec=str(
                    config.block_fht_mlp_pair_vq_feedback_codec
                ),
                feedback_output_group_size=int(
                    config.block_fht_mlp_pair_vq_feedback_output_group_size
                ),
                feedback_residual_probe_steps=tuple(
                    config.block_fht_mlp_pair_vq_feedback_residual_probe_steps
                ),
                feedback_residual_probe_layers=tuple(
                    config.block_fht_mlp_pair_vq_feedback_residual_probe_layers
                ),
                feedback_residual_probe_lloyd_iterations=tuple(
                    config.block_fht_mlp_pair_vq_feedback_residual_probe_lloyd_iterations
                ),
                feedback_transform_probe_block_sizes=tuple(
                    config.block_fht_mlp_pair_vq_feedback_transform_probe_block_sizes
                ),
                feedback_lattice_probe_block_sizes=tuple(
                    config.block_fht_mlp_pair_vq_feedback_lattice_probe_block_sizes
                ),
                feedback_lattice_probe_coordinate_bits=tuple(
                    config.block_fht_mlp_pair_vq_feedback_lattice_probe_coordinate_bits
                ),
                feedback_axis_adaptation_probe_block_size=int(
                    config.block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_block_size
                ),
                feedback_axis_adaptation_probe_coordinate_bits=int(
                    config.block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_coordinate_bits
                ),
                feedback_fractional_probe_block_size=int(
                    config.block_fht_mlp_pair_vq_feedback_fractional_probe_block_size
                ),
                feedback_fractional_probe_base_coordinate_bits=int(
                    config.block_fht_mlp_pair_vq_feedback_fractional_probe_base_coordinate_bits
                ),
                feedback_fractional_probe_refinement_fractions=tuple(
                    config.block_fht_mlp_pair_vq_feedback_fractional_probe_refinement_fractions
                ),
                neighbor_candidates=int(
                    config.block_fht_mlp_pair_vq_neighbor_candidates
                ),
                code_refresh_interval=int(
                    config.block_fht_mlp_pair_vq_code_refresh_interval
                ),
                lazy_retraction_interval=int(
                    config.block_fht_mlp_pair_vq_lazy_retraction_interval
                ),
                lazy_retraction_forced_steps=tuple(
                    config.block_fht_mlp_pair_vq_lazy_retraction_forced_steps
                ),
            )
        elif int8_lattice_cproj:
            self.c_proj = MuonInt8LatticeLinear(
                4 * config.n_embd,
                config.n_embd,
                bias=config.bias,
                block_size=int(config.block_fht_mlp_int8_lattice_block_size),
                base_seed=(
                    int(config.block_fht_mlp_int8_lattice_seed)
                    + layer_id * 8192
                    + 4096
                ),
                weight_std=0.02 / math.sqrt(2 * config.n_layer),
                layer_id=layer_id,
                error_feedback=(
                    config.block_fht_mlp_int8_lattice_error_feedback
                ),
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
                output_stages=int(
                    config.block_fht_mlp_cproj_hybrid_task_stages
                    if config.block_fht_mlp_cproj_hybrid_output
                    else config
                    .block_fht_mlp_cproj_muon_matched_givens_output_stages
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
                hybrid_output=bool(
                    config.block_fht_mlp_cproj_hybrid_output
                ),
                hybrid_directed_incoming=int(
                    config.block_fht_mlp_cproj_hybrid_directed_incoming
                    if config.block_fht_mlp_cproj_hybrid_output
                    else 0
                ),
                hybrid_control_output_stages=int(
                    config.block_fht_mlp_cproj_hybrid_control_stages
                ),
                hybrid_ridge_ratio=float(
                    config.block_fht_mlp_cproj_hybrid_ridge_ratio
                ),
                hybrid_functional_sample_cap=int(
                    config.block_fht_mlp_cproj_hybrid_sample_cap
                ),
                activation_energy_metric=bool(
                    config.block_fht_mlp_cproj_activation_energy_metric
                ),
                activation_energy_metric_decay=float(
                    config
                    .block_fht_mlp_cproj_activation_energy_metric_decay
                ),
                activation_energy_metric_minimum=float(
                    config
                    .block_fht_mlp_cproj_activation_energy_metric_minimum
                ),
                activation_energy_metric_maximum=float(
                    config
                    .block_fht_mlp_cproj_activation_energy_metric_maximum
                ),
                activation_energy_metric_epsilon=float(
                    config
                    .block_fht_mlp_cproj_activation_energy_metric_epsilon
                ),
                output_symmetric_shear_stages=int(
                    config
                    .block_fht_mlp_cproj_output_symmetric_shear_stages
                ),
                output_symmetric_shear_neighbors=int(
                    config
                    .block_fht_mlp_cproj_output_symmetric_shear_neighbors
                ),
                output_symmetric_shear_max_condition_number=float(
                    config
                    .block_fht_mlp_cproj_output_symmetric_shear_max_condition_number
                ),
                global_log_volume=bool(
                    config.block_fht_mlp_cproj_global_log_volume
                ),
                global_log_volume_max_abs=float(
                    config.block_fht_mlp_cproj_global_log_volume_max_abs
                ),
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
        elif dense_lwt_cproj:
            self.c_proj = nn.Linear(
                4 * config.n_embd,
                config.n_embd,
                bias=config.bias,
            )
        else:
            self.c_proj = make_linear(4 * config.n_embd, config.n_embd, config.bias, config, "mlp.c_proj", layer_id * 4 + 3)
        paired_monarch_width = int(
            config.block_fht_mlp_paired_monarch_block_width
        )
        tri_monarch_width = int(
            config.mlp_shared_dense_tri_monarch_block_width
        )
        if paired_monarch_width < 0:
            raise ValueError(
                "block_fht_mlp_paired_monarch_block_width must be "
                "non-negative"
            )
        if tri_monarch_width < 0:
            raise ValueError(
                "mlp_shared_dense_tri_monarch_block_width must be "
                "non-negative"
            )
        if paired_monarch_width and tri_monarch_width:
            raise ValueError(
                "paired Monarch and shared-dense tri-Monarch are mutually "
                "exclusive"
            )
        if paired_monarch_width:
            if not (
                isinstance(self.c_fc, BlockFHTLinear)
                and isinstance(self.c_proj, BlockFHTLinear)
            ):
                raise ValueError(
                    "paired Monarch requires generated plain mlp.c_fc and "
                    "mlp.c_proj BlockFHT targets"
                )
            if self.pregelu_block_rotation is not None:
                raise ValueError(
                    "paired Monarch cannot be combined with a separate "
                    "pre-GELU rotation"
                )
        if tri_monarch_width:
            if not config.mlp_shared_dense_trunk:
                raise ValueError(
                    "shared-dense tri-Monarch requires the shared dense MLP "
                    "trunk"
                )
            if not (
                isinstance(self.c_fc, nn.Linear)
                and isinstance(self.c_proj, nn.Linear)
            ):
                raise ValueError(
                    "shared-dense tri-Monarch requires plain dense c_fc and "
                    "c_proj matrices"
                )
            if self.pregelu_block_rotation is not None:
                raise ValueError(
                    "shared-dense tri-Monarch cannot be combined with a "
                    "separate pre-GELU rotation"
                )
        monarch_width = paired_monarch_width or tri_monarch_width
        monarch_scale = (
            float(config.block_fht_mlp_paired_monarch_coordinate_scale)
            if paired_monarch_width
            else float(config.mlp_shared_dense_tri_monarch_coordinate_scale)
        )
        monarch_seed = (
            int(config.block_fht_mlp_paired_monarch_seed)
            if paired_monarch_width
            else int(config.mlp_shared_dense_tri_monarch_seed)
        )
        self.paired_monarch = (
            LearnedMonarchHiddenMix(
                features=4 * config.n_embd,
                block_width=monarch_width,
                seed=monarch_seed + layer_id * 64,
                coordinate_scale=monarch_scale,
            )
            if monarch_width
            else None
        )
        self.input_monarch = (
            LearnedMonarchHiddenMix(
                features=config.n_embd,
                block_width=tri_monarch_width,
                seed=monarch_seed + layer_id * 64 + 1,
                coordinate_scale=monarch_scale,
            )
            if tri_monarch_width
            else None
        )
        self.output_monarch = (
            LearnedMonarchHiddenMix(
                features=config.n_embd,
                block_width=tri_monarch_width,
                seed=monarch_seed + layer_id * 64 + 2,
                coordinate_scale=monarch_scale,
            )
            if tri_monarch_width
            else None
        )
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
            self.c_proj,
            (nn.Linear, BlockFHTLinear, MuonMatchedGivensLinear),
        ):
            raise ValueError(
                "block-orthogonal output rotation requires a materialized "
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
            self.c_proj,
            (nn.Linear, BlockFHTLinear, MuonMatchedGivensLinear),
        ):
            raise ValueError(
                "block-orthogonal hidden rotation requires a materialized "
                "mlp.c_proj linear"
            )
        if hidden_block_rotation_stages and incompatible_cproj_addon:
            raise ValueError(
                "block-orthogonal hidden rotation cannot be combined with "
                "a separate c_proj residual add-on"
            )
        if hidden_block_rotation_stages and self.paired_monarch is not None:
            raise ValueError(
                "paired Monarch cannot be combined with a separate hidden "
                "rotation"
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
                self.paired_monarch,
                self.output_monarch,
            )
        )

    def has_charted_cfc(self) -> bool:
        return any(
            value is not None
            for value in (
                self.pregelu_block_rotation,
                self.paired_monarch,
                self.input_monarch,
            )
        )

    def _cfc_chart_parameters(
        self, *, require_grad_only: bool
    ) -> list[torch.Tensor]:
        parameters: list[torch.Tensor] = []
        for module in (
            self.pregelu_block_rotation,
            self.paired_monarch,
            self.input_monarch,
        ):
            if module is None:
                continue
            parameters.extend(
                parameter
                for parameter in module.parameters()
                if not require_grad_only or parameter.requires_grad
            )
        return parameters

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
        charted = weight
        # Row activations use h' = h @ R.  Since h = x @ W^T, folding the
        # frame into F.linear requires W' = R^T @ W.  Apply R to W^T and
        # transpose the result instead of materializing the O(hidden^2)
        # rotation matrix.
        if self.pregelu_block_rotation is not None:
            charted = self.pregelu_block_rotation(
                charted.transpose(0, 1)
            ).transpose(0, 1)
        if self.paired_monarch is not None:
            charted = self.paired_monarch(
                charted.transpose(0, 1)
            ).transpose(0, 1)
        if self.input_monarch is not None:
            charted = self.input_monarch(charted)
        return charted.contiguous()

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
            for parameter in self._cfc_chart_parameters(
                require_grad_only=False
            )
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
        if not self.has_charted_cfc():
            raise RuntimeError("cached pre-GELU frame is missing")
        base_weight = self._cfc_base_weight()
        chart_parameters = self._cfc_chart_parameters(
            require_grad_only=True
        )
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
        if self.paired_monarch is not None:
            charted = self.paired_monarch(charted)
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
        if self.output_monarch is not None:
            transposed = self.output_monarch(transposed)
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
        if self.paired_monarch is not None:
            parameters.extend(
                parameter
                for parameter in self.paired_monarch.parameters()
                if not require_grad_only or parameter.requires_grad
            )
        if self.output_monarch is not None:
            parameters.extend(
                parameter
                for parameter in self.output_monarch.parameters()
                if not require_grad_only or parameter.requires_grad
            )
        return parameters

    def _cproj_base_weight(self) -> torch.Tensor:
        """Return the materialized c_proj weight used by an exact chart.

        BlockFHTLinear exposes its per-step materialization through
        ``_cached_weight``.  MuonMatchedGivensLinear instead owns a persistent
        folded ``weight`` buffer.  Treating both as materialized bases lets
        frozen-endpoint diagnostics reuse the exact cached chart VJP without
        changing the normal BlockFHT training path.
        """
        weight = getattr(self.c_proj, "_cached_weight", None)
        if weight is not None:
            return weight
        if not hasattr(self.c_proj, "weight"):
            raise RuntimeError(
                "charted c_proj requires a materialized base weight"
            )
        return self.c_proj.weight

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
        try:
            base_weight = self._cproj_base_weight()
        except RuntimeError:
            base_weight = None
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
        base_weight = self._cproj_base_weight()
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
        weight = self._cproj_base_weight()
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
        record_functional_context = getattr(
            self.c_fc, "record_functional_context", None
        )
        if record_functional_context is not None:
            record_functional_context(x, hidden)
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
        record_hybrid_output_context = getattr(
            self.c_proj, "record_hybrid_output_context", None
        )
        if record_hybrid_output_context is not None:
            record_hybrid_output_context(activated)
        record_activation_energy_context = getattr(
            self.c_proj, "record_activation_energy_context", None
        )
        if record_activation_energy_context is not None:
            record_activation_energy_context(activated)
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
    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = (
            SparseMoEMLP(config, layer_id)
            if config.moe_num_experts > 0
            else MLP(config, layer_id)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        mlp_input = self.ln_2(x)
        mlp_output = self.mlp(mlp_input)
        if isinstance(self.mlp, SparseMoEMLP) or (
            self.mlp.residual_conditioned_output_slope is None
            or self.mlp.conditioned_output_gate_source == "postgelu"
        ):
            return x + mlp_output
        return x + self.mlp.apply_residual_conditioned_output_gate(
            mlp_input,
            mlp_output,
        )


def validate_sparse_moe_block_fht_scope(
    *,
    moe_num_experts: int,
    block_fht: bool,
    block_fht_targets: tuple[str, ...] | list[str],
) -> None:
    """Keep dense-complete experts untouched in mixed MoE/BlockFHT runs."""
    if moe_num_experts <= 0 or not block_fht:
        return
    non_attention_targets = sorted(
        target
        for target in block_fht_targets
        if not target.startswith("attn.")
    )
    if non_attention_targets:
        raise ValueError(
            "dense-complete-expert MoE permits BlockFHT only for attention "
            "targets; unsupported targets: "
            + ", ".join(non_attention_targets)
        )


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        validate_sparse_moe_block_fht_scope(
            moe_num_experts=config.moe_num_experts,
            block_fht=config.block_fht,
            block_fht_targets=config.block_fht_targets,
        )
        cproj_layers = tuple(
            config.block_fht_mlp_cproj_muon_matched_givens_layers
        )
        if len(set(cproj_layers)) != len(cproj_layers):
            raise ValueError(
                "Muon-matched c_proj layer IDs must be unique"
            )
        if any(
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or layer >= config.n_layer
            for layer in cproj_layers
        ):
            raise ValueError(
                "Muon-matched c_proj layer IDs must be integers in "
                "[0, n_layer)"
            )
        if cproj_layers and not config.block_fht_mlp_cproj_muon_matched_givens:
            raise ValueError(
                "Muon-matched c_proj layer IDs require the c_proj chart"
            )
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
        if (
            config.mlp_shared_dense_trunk
            and config.mlp_shared_dense_block_fht_residual
        ):
            raise ValueError(
                "shared dense trunk and shared dense BlockFHT residual are mutually exclusive"
            )
        if config.mlp_shared_dense_trunk:
            self._tie_shared_dense_mlp_trunk()
        if config.mlp_shared_dense_block_fht_residual:
            self._tie_shared_dense_block_fht_residual()

    def _tie_shared_dense_mlp_trunk(self) -> None:
        """Share learned full-rank MLP pairs within contiguous depth groups.

        The layer-private pre-GELU and residual-output gains remain distinct.
        Tying after normal GPT initialization preserves the conventional
        c_proj residual scaling while avoiding unused per-layer dense weights.
        One group preserves the original all-layer sharing experiment.
        """

        if self.config.moe_num_experts > 0:
            raise ValueError("shared dense MLP trunk is incompatible with MoE")
        mlp_targets = sorted(
            target
            for target in self.config.block_fht_targets
            if target.startswith("mlp.")
        )
        if mlp_targets:
            raise ValueError(
                "shared dense MLP trunk requires dense MLP matrices; "
                "remove BlockFHT MLP targets: " + ", ".join(mlp_targets)
            )
        if not self.config.block_fht_ffn_pregelu_gain:
            raise ValueError(
                "shared dense MLP trunk requires layer-private pre-GELU gain"
            )
        if not self.config.block_fht_mlp_residual_output_gain:
            raise ValueError(
                "shared dense MLP trunk requires layer-private residual-output gain"
            )
        blocks = list(self.transformer.h)
        if not blocks:
            raise ValueError("shared dense MLP trunk requires at least one layer")
        groups = int(self.config.mlp_shared_dense_trunk_groups)
        boundaries = tuple(int(value) for value in self.config.mlp_shared_dense_trunk_boundaries)
        if boundaries:
            if groups != len(boundaries):
                raise ValueError(
                    "shared dense MLP trunk group count must equal the number "
                    "of explicit boundaries"
                )
            if (
                boundaries[-1] != len(blocks)
                or any(value <= 0 or value > len(blocks) for value in boundaries)
                or any(later <= earlier for earlier, later in zip(boundaries, boundaries[1:]))
            ):
                raise ValueError(
                    "shared dense MLP trunk boundaries must increase strictly "
                    "and terminate at n_layer"
                )
            ranges = list(zip((0, *boundaries[:-1]), boundaries, strict=True))
        else:
            if groups <= 0 or groups > len(blocks) or len(blocks) % groups:
                raise ValueError(
                    "shared dense MLP trunk groups must be positive, no greater "
                    "than n_layer, and evenly divide n_layer"
                )
            group_size = len(blocks) // groups
            ranges = [
                (start, start + group_size)
                for start in range(0, len(blocks), group_size)
            ]
        for block in blocks:
            if not isinstance(block.mlp, MLP):
                raise ValueError("shared dense MLP trunk requires the dense MLP module")
            if not isinstance(block.mlp.c_fc, nn.Linear) or not isinstance(
                block.mlp.c_proj, nn.Linear
            ):
                raise ValueError(
                    "shared dense MLP trunk requires plain dense c_fc and c_proj"
                )

        for start, stop in ranges:
            root = blocks[start].mlp
            for block in blocks[start + 1 : stop]:
                block.mlp.c_fc.weight = root.c_fc.weight
                block.mlp.c_proj.weight = root.c_proj.weight
                if self.config.bias:
                    assert root.c_fc.bias is not None and root.c_proj.bias is not None
                    block.mlp.c_fc.bias = root.c_fc.bias
                    block.mlp.c_proj.bias = root.c_proj.bias

    def _tie_shared_dense_block_fht_residual(self) -> None:
        """Tie the learned dense base while retaining private FHT residuals."""

        if self.config.mlp_shared_dense_trunk:
            raise ValueError(
                "shared dense trunk and shared dense BlockFHT residual are mutually exclusive"
            )
        if self.config.moe_num_experts > 0:
            raise ValueError(
                "shared dense BlockFHT residual is incompatible with MoE"
            )
        required_targets = {"mlp.c_fc", "mlp.c_proj"}
        if not required_targets.issubset(self.config.block_fht_targets):
            raise ValueError(
                "shared dense BlockFHT residual requires mlp.c_fc and mlp.c_proj targets"
            )
        if self.config.block_fht_residual_base_scale != 0.0:
            raise ValueError(
                "shared dense BlockFHT residual cannot use the legacy residual base"
            )
        residual_scale = float(
            self.config.mlp_shared_dense_block_fht_residual_scale
        )
        if not math.isfinite(residual_scale) or not 0.0 < residual_scale < 1.0:
            raise ValueError(
                "shared dense BlockFHT residual scale must be finite and in (0, 1)"
            )
        if not self.config.block_fht_match_gpt_init:
            raise ValueError(
                "shared dense BlockFHT residual requires GPT init matching"
            )
        if not self.config.block_fht_ffn_pregelu_gain:
            raise ValueError(
                "shared dense BlockFHT residual requires layer-private pre-GELU gain"
            )
        if not self.config.block_fht_mlp_residual_output_gain:
            raise ValueError(
                "shared dense BlockFHT residual requires layer-private residual-output gain"
            )
        blocks = list(self.transformer.h)
        if not blocks:
            raise ValueError(
                "shared dense BlockFHT residual requires at least one layer"
            )
        for block in blocks:
            if not isinstance(block.mlp, MLP) or not all(
                isinstance(module, BlockFHTLinear)
                for module in (block.mlp.c_fc, block.mlp.c_proj)
            ):
                raise ValueError(
                    "shared dense BlockFHT residual requires plain BlockFHT MLP matrices"
                )
            if not all(
                isinstance(module.residual_base_weight, nn.Parameter)
                for module in (block.mlp.c_fc, block.mlp.c_proj)
            ):
                raise ValueError(
                    "shared dense BlockFHT residual requires trainable dense bases"
                )

        root = blocks[0].mlp
        for block in blocks[1:]:
            block.mlp.c_fc.residual_base_weight = (
                root.c_fc.residual_base_weight
            )
            block.mlp.c_proj.residual_base_weight = (
                root.c_proj.residual_base_weight
            )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(
            module, (MuonFunctionalShearLinear, MuonDirectedProductLinear)
        ):
            nn.init.normal_(
                module.weight, mean=0.0, std=module.weight_std
            )
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
            if self.training and self.config.moe_num_experts > 0:
                load_balance, router_z = self.moe_router_losses()
                loss = (
                    loss
                    + float(self.config.moe_load_balance_aux_coefficient)
                    * load_balance
                    + float(self.config.moe_router_z_loss_coefficient)
                    * router_z
                )
        return logits, loss

    def moe_router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        losses = [
            block.mlp.router_auxiliary_loss()
            for block in self.transformer.h
            if isinstance(block.mlp, SparseMoEMLP)
        ]
        if not losses:
            zero = next(self.parameters()).new_zeros(())
            return zero, zero
        load_balance, router_z = zip(*losses, strict=True)
        return torch.stack(load_balance).mean(), torch.stack(router_z).mean()

    def moe_parameter_stats(self) -> dict[str, int]:
        if self.config.moe_num_experts <= 0:
            return {"active": 0, "stored": 0, "expert": 0, "router": 0}
        experts = [
            block.mlp
            for block in self.transformer.h
            if isinstance(block.mlp, SparseMoEMLP)
        ]
        if len(experts) != self.config.n_layer:
            raise RuntimeError("every transformer layer must own one sparse MoE")
        expert = sum(
            module.expert_c_fc.numel() + module.expert_c_proj.numel()
            for module in experts
        )
        router = sum(module.router.weight.numel() for module in experts)
        stored = sum(parameter.numel() for parameter in self.parameters())
        inactive_expert = sum(
            (module.num_experts - module.top_k)
            * (
                module.expert_c_fc[0].numel()
                + module.expert_c_proj[0].numel()
            )
            for module in experts
        )
        return {
            "active": stored - inactive_expert,
            "stored": stored,
            "expert": expert,
            "router": router,
        }

    def postgelu_spread_loss(self) -> torch.Tensor:
        losses = []
        for block in self.transformer.h:
            loss = block.mlp.postgelu_spread_loss()
            if loss is not None:
                losses.append(loss)
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def schedule_attention_cayley_atlas_gradients(
        self,
        iter_num: int,
    ) -> tuple[int | None, int, int]:
        """Train one local attention chart per registered trajectory phase.

        Every future chart is exactly identity initialized. At a phase
        boundary the preceding chart is frozen and the next chart becomes
        trainable, so the represented function is continuous while the local
        tangent receives a fresh fixed basis.
        """
        starts = tuple(self.config.block_fht_attn_cayley_atlas_start_steps)
        if not starts:
            return None, 0, 0
        active = max(
            index for index, start in enumerate(starts) if start <= iter_num
        )
        held_future = 0
        frozen_previous = 0
        for block in self.transformer.h:
            block.attn.active_cayley_atlas_stage = active
            for stage, modules in enumerate(
                block.attn.attention_cayley_stage_modules()
            ):
                for module in modules:
                    for parameter in module.parameters():
                        is_active = stage == active
                        parameter.requires_grad_(is_active)
                        if not is_active:
                            parameter.grad = None
                            if stage < active:
                                frozen_previous += 1
                            else:
                                held_future += 1
        return active, held_future, frozen_previous

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
        muon_mlp_polar_ridge: float = 0.0,
        muon_mlp_ns_steps: int = 0,
        muon_mlp_lr_scale: float = 1.0,
        muon_adamw_lr_scale: float = 1.0,
        muon_split_attention_qkv_rows: bool = False,
        block_fht_attn_cayley_lr_scale: float = 1.0,
        block_fht_attn_cayley_muon_lr_scale: float = 1.0,
        block_fht_mlp_chart_lr_scale: float = 1.0,
        block_fht_mlp_pregelu_chart_lr_scale: float = 1.0,
    ):
        params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        if (
            not math.isfinite(float(muon_mlp_polar_ridge))
            or float(muon_mlp_polar_ridge) < 0.0
        ):
            raise ValueError("MLP Muon polar ridge must be finite and non-negative")
        if int(muon_mlp_ns_steps) < 0:
            raise ValueError("MLP Muon NS steps must be non-negative")
        if (
            not math.isfinite(float(muon_mlp_lr_scale))
            or float(muon_mlp_lr_scale) <= 0.0
        ):
            raise ValueError("MLP Muon LR scale must be finite and positive")
        resolved_mlp_ns_steps = (
            int(muon_ns_steps)
            if int(muon_mlp_ns_steps) == 0
            else int(muon_mlp_ns_steps)
        )
        split_qkv_named: list[tuple[str, nn.Parameter]] = []
        if muon_split_attention_qkv_rows:
            if optimizer != "muon":
                raise ValueError("split-QKV row updates require optimizer='muon'")
            expected_shape = (3 * self.config.n_embd, self.config.n_embd)
            split_qkv_named = [
                (name, param)
                for name, param in params.items()
                if name.endswith(".attn.c_attn.weight")
            ]
            if len(split_qkv_named) != self.config.n_layer or any(
                tuple(param.shape) != expected_shape
                for _name, param in split_qkv_named
            ):
                raise ValueError(
                    "split-QKV row updates require one dense packed c_attn "
                    "weight with shape (3*n_embd, n_embd) per layer"
                )
            for _name, param in split_qkv_named:
                param._muon_row_splits = (
                    2 * self.config.n_embd,
                    self.config.n_embd,
                )
        decay = [param for _, param in params.items() if param.dim() >= 2]
        nodecay = [param for _, param in params.items() if param.dim() < 2]
        muon_matched_givens_named_modules = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, MuonMatchedGivensLinear)
        ]
        attention_muon_matched_givens_modules = [
            module
            for name, module in muon_matched_givens_named_modules
            if ".attn." in name
        ]
        mlp_muon_matched_givens_modules = [
            module
            for name, module in muon_matched_givens_named_modules
            if ".attn." not in name
        ]
        muon_matched_givens_modules = [
            module for _name, module in muon_matched_givens_named_modules
        ]
        int8_lattice_named_modules = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, MuonInt8LatticeLinear)
        ]
        attention_int8_lattice_modules = [
            module
            for name, module in int8_lattice_named_modules
            if ".attn." in name
        ]
        mlp_int8_lattice_modules = [
            module
            for name, module in int8_lattice_named_modules
            if ".mlp." in name
        ]
        int8_lattice_modules = [
            module for _name, module in int8_lattice_named_modules
        ]
        pair_vq_modules = [
            module
            for module in self.modules()
            if isinstance(module, MuonPairVQLinear)
        ]
        functional_shear_pairs = [
            (block.mlp.c_fc, block.mlp.c_proj)
            for block in self.transformer.h
            if isinstance(block.mlp, MLP)
            and isinstance(block.mlp.c_fc, MuonFunctionalShearLinear)
        ]
        directed_product_modules = [
            module
            for module in self.modules()
            if isinstance(module, MuonDirectedProductLinear)
        ]
        if (
            muon_matched_givens_modules
            or int8_lattice_modules
            or pair_vq_modules
            or functional_shear_pairs
            or directed_product_modules
        ) and optimizer != "muon":
            raise ValueError(
                "Muon chart optimizers require optimizer='muon'"
            )
        if optimizer == "muon":
            moe_router_named = [
                (name, param)
                for name, param in params.items()
                if ".mlp.router." in name
            ]
            moe_router_ids = {id(param) for _, param in moe_router_named}
            attention_cayley_tokens = (
                ".qk_input_cayley.",
                ".qk_output_cayley.",
                ".v_input_cayley.",
                ".v_output_cayley.",
                ".cproj_input_cayley.",
                ".cproj_output_cayley.",
                "_cayley_atlas.",
            )
            attention_cayley_matrix_named = [
                (name, param)
                for name, param in params.items()
                if param.dim() >= 2
                and any(
                    token in name for token in attention_cayley_tokens
                )
            ]
            attention_cayley_matrix_ids = {
                id(param) for _, param in attention_cayley_matrix_named
            }
            cayley_factor_optimizer = str(
                self.config.block_fht_attn_cayley_factor_optimizer
            )
            if cayley_factor_optimizer == "hybrid_left_muon":
                attention_cayley_muon_matrix_named = [
                    (name, param)
                    for name, param in attention_cayley_matrix_named
                    if name.endswith(".left")
                ]
                attention_cayley_adamw_matrix_named = [
                    (name, param)
                    for name, param in attention_cayley_matrix_named
                    if name.endswith(".right")
                ]
                routed_ids = {
                    id(param)
                    for _, param in (
                        attention_cayley_muon_matrix_named
                        + attention_cayley_adamw_matrix_named
                    )
                }
                if routed_ids != attention_cayley_matrix_ids:
                    raise ValueError(
                        "hybrid attention Cayley routing requires every "
                        "matrix factor to end in .left or .right"
                    )
            else:
                attention_cayley_muon_matrix_named = (
                    attention_cayley_matrix_named
                )
                attention_cayley_adamw_matrix_named = []
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
            use_special_mlp_muon = (
                float(muon_mlp_polar_ridge) > 0.0
                or resolved_mlp_ns_steps != int(muon_ns_steps)
                or float(muon_mlp_lr_scale) != 1.0
            )
            mlp_special_matrix_named = (
                [
                    (name, param)
                    for name, param in params.items()
                    if param.dim() >= 2
                    and (
                        name.endswith(".mlp.c_fc.weight")
                        or name.endswith(".mlp.c_proj.weight")
                    )
                ]
                if use_special_mlp_muon
                else []
            )
            mlp_special_matrix_ids = {
                id(param) for _name, param in mlp_special_matrix_named
            }
            matrix = [
                param
                for name, param in params.items()
                if param.dim() >= 2
                and "wte" not in name
                and "wpe" not in name
                and "lm_head" not in name
                and id(param) not in product_factor_ids
                and id(param) not in attention_cayley_matrix_ids
                and id(param) not in moe_router_ids
                and id(param) not in mlp_special_matrix_ids
            ]
            other = [
                param
                for name, param in params.items()
                if (
                    (
                        param.dim() < 2
                        or "wte" in name
                        or "wpe" in name
                        or "lm_head" in name
                    )
                    and id(param) not in product_factor_ids
                )
                or id(param) in moe_router_ids
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
                ".paired_monarch.coordinates",
                ".input_monarch.coordinates",
                ".output_monarch.coordinates",
            )
            if self.config.mlp_shared_dense_trunk:
                chart_names += (".pregelu_gain",)
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
            if mlp_special_matrix_named:
                optimizers.append(
                    Muon(
                        [param for _name, param in mlp_special_matrix_named],
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=resolved_mlp_ns_steps,
                        polar_ridge=float(muon_mlp_polar_ridge),
                        hierarchical_feedback_fit=bool(
                            self.config.block_fht_mlp_pair_vq_hierarchical_feedback_fit
                        ),
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = float(muon_mlp_lr_scale)
            if attention_cayley_muon_matrix_named:
                cayley_muon_groups = [
                    {
                        "params": [param],
                        "weight_decay": 0.0,
                        "lr_scale": (
                            float(block_fht_attn_cayley_muon_lr_scale)
                            * math.sqrt(float(param.shape[1]))
                        ),
                    }
                    for _name, param in attention_cayley_muon_matrix_named
                ]
                optimizers.append(
                    Muon(
                        cayley_muon_groups,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=0.0,
                        ns_steps=muon_ns_steps,
                    )
                )
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
            if functional_shear_pairs:
                optimizers.append(
                    MuonFunctionalShear(
                        functional_shear_pairs,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=muon_ns_steps,
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = 1.0
            if directed_product_modules:
                optimizers.append(
                    MuonDirectedProduct(
                        directed_product_modules,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=muon_ns_steps,
                        momentum_state_dtype=(
                            self.config.block_fht_mlp_muon_momentum_state_dtype
                        ),
                        feedback_state_codec=(
                            self.config.block_fht_mlp_error_feedback_state_codec
                        ),
                        feedback_state_block_size=(
                            self.config.block_fht_mlp_error_feedback_state_block_size
                        ),
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = 1.0
            if attention_muon_matched_givens_modules:
                optimizers.append(
                    MuonMatchedGivens(
                        attention_muon_matched_givens_modules,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=muon_ns_steps,
                        error_feedback=(
                            self.config
                            .block_fht_attn_muon_matched_givens_error_feedback
                        ),
                        error_feedback_decay=(
                            self.config
                            .block_fht_attn_muon_matched_givens_error_feedback_decay
                        ),
                        error_feedback_max_nominal_steps=(
                            self.config
                            .block_fht_attn_muon_matched_givens_error_feedback_max_nominal_steps
                        ),
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = 1.0
                    group["attention_error_feedback"] = True
            if int8_lattice_modules:
                optimizers.append(
                    MuonInt8Lattice(
                        int8_lattice_modules,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=muon_ns_steps,
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = 1.0
            if pair_vq_modules:
                optimizers.append(
                    MuonPairVQ(
                        pair_vq_modules,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=resolved_mlp_ns_steps,
                        polar_ridge=float(muon_mlp_polar_ridge),
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = float(muon_mlp_lr_scale)
            if mlp_muon_matched_givens_modules:
                optimizers.append(
                    MuonMatchedGivens(
                        mlp_muon_matched_givens_modules,
                        lr=learning_rate,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        ns_steps=muon_ns_steps,
                        error_feedback=(
                            self.config
                            .block_fht_mlp_cproj_muon_matched_givens_error_feedback
                        ),
                        error_feedback_decay=(
                            self.config
                            .block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay
                        ),
                        error_feedback_max_nominal_steps=(
                            self.config
                            .block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps
                        ),
                        momentum_state_dtype=(
                            self.config.block_fht_mlp_muon_momentum_state_dtype
                        ),
                        feedback_state_codec=(
                            self.config.block_fht_mlp_error_feedback_state_codec
                        ),
                        feedback_state_block_size=(
                            self.config.block_fht_mlp_error_feedback_state_block_size
                        ),
                    )
                )
                for group in optimizers[-1].param_groups:
                    group["lr_scale"] = 1.0
                    group["cproj_error_feedback_decay_schedule"] = True
            if other or attention_cayley_adamw_matrix_named:
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
                if attention_cayley_adamw_matrix_named:
                    fallback_groups.append(
                        {
                            "params": [
                                param
                                for _, param in (
                                    attention_cayley_adamw_matrix_named
                                )
                            ],
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
                f"muon_split_qkv_tensors={len(split_qkv_named)} "
                f"product_fht_factor_tensors={len(product_factors)} "
                "muon_matched_givens_tensors="
                f"{len(muon_matched_givens_modules)} "
                "attention_muon_matched_givens_tensors="
                f"{len(attention_muon_matched_givens_modules)} "
                "attention_int8_lattice_tensors="
                f"{len(attention_int8_lattice_modules)} "
                "mlp_int8_lattice_tensors="
                f"{len(mlp_int8_lattice_modules)} "
                "mlp_muon_matched_givens_tensors="
                f"{len(mlp_muon_matched_givens_modules)} "
                "muon_functional_shear_tensors="
                f"{len(functional_shear_pairs)} "
                f"mlp_chart_tensors={len(chart_other)} "
                f"mlp_pregelu_chart_tensors={len(pregelu_chart_other)} "
                f"momentum={muon_momentum} ns_steps={muon_ns_steps} "
                f"mlp_polar_ridge={float(muon_mlp_polar_ridge)} "
                f"mlp_ns_steps={resolved_mlp_ns_steps} "
                f"mlp_muon_lr_scale={float(muon_mlp_lr_scale)} "
                f"mlp_special_matrix_tensors={len(mlp_special_matrix_named)} "
                f"adamw_lr_scale={float(muon_adamw_lr_scale)} "
                f"mlp_chart_lr_scale={float(block_fht_mlp_chart_lr_scale)} "
                f"mlp_pregelu_chart_lr_scale="
                f"{float(block_fht_mlp_pregelu_chart_lr_scale)} "
                f"attn_cayley_tensors={len(attention_cayley_other)} "
                f"attn_cayley_lr_scale="
                f"{float(block_fht_attn_cayley_lr_scale)} "
                "attn_cayley_muon_tensors="
                f"{len(attention_cayley_muon_matrix_named)} "
                "attn_cayley_adamw_matrix_tensors="
                f"{len(attention_cayley_adamw_matrix_named)} "
                "attn_cayley_muon_lr_scale="
                f"{float(block_fht_attn_cayley_muon_lr_scale)} "
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
            elif isinstance(module, MuonInt8LatticeLinear):
                modules += 1
                generated += module.in_features * module.out_features
            elif isinstance(module, MuonPairVQLinear):
                modules += 1
                generated += module.in_features * module.out_features
                latent += module.codebooks.numel()
            elif isinstance(module, MuonFunctionalShearLinear):
                modules += 1
                generated += module.in_features * module.out_features
                latent += module.coordinate_count
            elif isinstance(module, MuonDirectedProductLinear):
                modules += 1
                generated += module.in_features * module.out_features
                latent += module.coordinate_count
        return {"modules": modules, "generated": generated, "latent": latent}

    def _int8_lattice_stats(
        self, *, scope: str
    ) -> dict[str, int | float]:
        named_modules = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, MuonInt8LatticeLinear)
        ]
        if scope == "attention":
            modules = [
                module for name, module in named_modules if ".attn." in name
            ]
        elif scope == "mlp":
            modules = [
                module for name, module in named_modules if ".mlp." in name
            ]
        elif scope == "all":
            modules = [module for _name, module in named_modules]
        else:
            raise ValueError(f"unknown int8 lattice scope: {scope}")
        codec_bytes = sum(module.persistent_codec_bytes for module in modules)
        fp32_bytes = sum(module.fp32_weight_bytes for module in modules)
        return {
            "modules": len(modules),
            "elements": sum(module.element_count for module in modules),
            "codec_bytes": codec_bytes,
            "fp32_weight_bytes": fp32_bytes,
            "storage_ratio": (
                codec_bytes / fp32_bytes if fp32_bytes else 0.0
            ),
        }

    def attention_int8_lattice_stats(self) -> dict[str, int | float]:
        return self._int8_lattice_stats(scope="attention")

    def mlp_int8_lattice_stats(self) -> dict[str, int | float]:
        return self._int8_lattice_stats(scope="mlp")

    def mlp_pair_vq_stats(self) -> dict[str, int | float | str]:
        modules = [
            module
            for module in self.modules()
            if isinstance(module, MuonPairVQLinear)
        ]
        elements = sum(module.element_count for module in modules)
        codec_bytes = sum(module.persistent_codec_bytes for module in modules)
        compact_momentum_bytes = sum(
            0 if module.fp16_ambient_momentum else module.compact_momentum_bytes
            for module in modules
        )
        ambient_momentum_bytes = sum(
            module.ambient_momentum_bytes for module in modules
        )
        momentum_bytes = compact_momentum_bytes + ambient_momentum_bytes
        feedback_bytes = sum(module.compact_feedback_bytes for module in modules)
        dense_bf16_bytes = 2 * elements
        dense_fp32_weight_momentum_bytes = 8 * elements
        persistent_training_bytes = codec_bytes + momentum_bytes + feedback_bytes
        return {
            "modules": len(modules),
            "elements": elements,
            "codec_bytes": codec_bytes,
            "compact_momentum_bytes": compact_momentum_bytes,
            "ambient_momentum_bytes": ambient_momentum_bytes,
            "optimizer_momentum_bytes": momentum_bytes,
            "compact_feedback_bytes": feedback_bytes,
            "persistent_training_bytes": persistent_training_bytes,
            "model_compression_vs_dense_bf16": (
                dense_bf16_bytes / codec_bytes if codec_bytes else 0.0
            ),
            "training_compression_vs_dense_fp32_weight_plus_momentum": (
                dense_fp32_weight_momentum_bytes / persistent_training_bytes
                if persistent_training_bytes
                else 0.0
            ),
            "dense_master_weight": "disabled",
            "dense_optimizer_momentum": (
                (
                    "fp16_reserved_escape_capacity_ceiling"
                    if any(
                        module.fp16_reserved_escape_granularity
                        for module in modules
                    )
                    else "fp16_ambient"
                )
                if ambient_momentum_bytes
                else "disabled"
            ),
            "dense_ambient_error_buffer": "disabled",
            "forward_visible_feedback": any(
                module.forward_visible_feedback for module in modules
            ),
            "compact_temporal_carry": (
                "uint8_cartesian_code_per_weight_pair"
                if feedback_bytes
                else "disabled"
            ),
        }

    def prepare_block_fht_cache(self, dtype: torch.dtype | None = None) -> None:
        prepare_block_fht_weight_cache(self, dtype=dtype)
        for block in self.transformer.h:
            if not isinstance(block.mlp, MLP):
                continue
            block.mlp.prepare_charted_cfc_cache()
            block.mlp.prepare_charted_cproj_cache()

    def flush_block_fht_cache(self) -> None:
        for block in self.transformer.h:
            if not isinstance(block.mlp, MLP):
                continue
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
            if isinstance(
                module,
                (
                    MuonMatchedGivensLinear,
                    MuonInt8LatticeLinear,
                    MuonPairVQLinear,
                    MuonFunctionalShearLinear,
                    MuonDirectedProductLinear,
                ),
            )
        )
        return parameters

    def finalize_product_fht_pullback_probes(
        self,
    ) -> list[dict[str, float]]:
        diagnostics = []
        for layer, block in enumerate(self.transformer.h):
            if not isinstance(block.mlp, MLP):
                continue
            module = block.mlp.c_proj
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
            if isinstance(block.mlp, MLP)
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
        if isinstance(module, LearnedLowRankCayleyMix):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
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
