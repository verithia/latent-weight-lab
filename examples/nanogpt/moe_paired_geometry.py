"""Same-gauge geometry primitives for complete sparse-MoE experts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _cosine_matrix(
    cross: torch.Tensor,
    left_norm: torch.Tensor,
    right_norm: torch.Tensor,
) -> torch.Tensor:
    denominator = left_norm[:, None] * right_norm[None, :]
    return cross / denominator.clamp_min(torch.finfo(cross.dtype).tiny)


def functional_atom_similarity(
    left_c_fc: torch.Tensor,
    left_c_proj: torch.Tensor,
    right_c_fc: torch.Tensor,
    right_c_proj: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity of all paired GELU feature/readout atoms.

    A hidden neuron is the rank-one function
    ``R_j(X) = GELU(X @ a_j) outer b_j``.  Its cross-inner-product factors as
    ``<h_j,h_k> * <b_j,b_k>``, avoiding materializing ``[N, d, hidden]``.
    """

    for name, value, rank in (
        ("left_c_fc", left_c_fc, 2),
        ("left_c_proj", left_c_proj, 2),
        ("right_c_fc", right_c_fc, 2),
        ("right_c_proj", right_c_proj, 2),
        ("activations", activations, 2),
    ):
        if value.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}")
    left_hidden, width = left_c_fc.shape
    right_hidden, right_width = right_c_fc.shape
    if width != right_width or activations.shape[1] != width:
        raise ValueError("c_fc and activation widths disagree")
    if left_c_proj.shape != (width, left_hidden):
        raise ValueError("left c_fc/c_proj paired shapes disagree")
    if right_c_proj.shape != (width, right_hidden):
        raise ValueError("right c_fc/c_proj paired shapes disagree")

    calculation_dtype = torch.float32
    x = activations.to(dtype=calculation_dtype)
    left_hidden_values = F.gelu(x @ left_c_fc.to(dtype=calculation_dtype).T)
    right_hidden_values = F.gelu(x @ right_c_fc.to(dtype=calculation_dtype).T)
    left_readout = left_c_proj.to(dtype=calculation_dtype)
    right_readout = right_c_proj.to(dtype=calculation_dtype)

    hidden_cross = left_hidden_values.T @ right_hidden_values
    readout_cross = left_readout.T @ right_readout
    cross = hidden_cross * readout_cross
    left_norm = (
        left_hidden_values.square().sum(dim=0)
        * left_readout.square().sum(dim=0)
    ).sqrt()
    right_norm = (
        right_hidden_values.square().sum(dim=0)
        * right_readout.square().sum(dim=0)
    ).sqrt()
    return _cosine_matrix(cross, left_norm, right_norm)


def maximum_weight_assignment(similarity: torch.Tensor) -> torch.Tensor:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("assignment similarity must be square")
    finite = torch.nan_to_num(similarity.detach().float(), nan=-1.0)
    rows, columns = linear_sum_assignment(-finite.cpu().numpy())
    if not np.array_equal(rows, np.arange(similarity.shape[0])):
        raise RuntimeError("assignment solver returned non-canonical row order")
    return torch.from_numpy(columns.astype(np.int64, copy=False))


@dataclass(frozen=True)
class PairedNeuronMatch:
    permutation: torch.Tensor
    mean_similarity: float
    minimum_similarity: float


def match_paired_neurons(
    left_c_fc: torch.Tensor,
    left_c_proj: torch.Tensor,
    right_c_fc: torch.Tensor,
    right_c_proj: torch.Tensor,
    activations: torch.Tensor,
) -> PairedNeuronMatch:
    similarity = functional_atom_similarity(
        left_c_fc,
        left_c_proj,
        right_c_fc,
        right_c_proj,
        activations,
    )
    permutation = maximum_weight_assignment(similarity)
    selected = similarity[
        torch.arange(similarity.shape[0], device=similarity.device),
        permutation.to(similarity.device),
    ]
    return PairedNeuronMatch(
        permutation=permutation,
        mean_similarity=float(selected.mean()),
        minimum_similarity=float(selected.min()),
    )


def apply_neuron_permutation(
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if c_fc.ndim != 2 or c_proj.ndim != 2:
        raise ValueError("expert matrices must be rank two")
    if c_fc.shape[0] != c_proj.shape[1] or permutation.numel() != c_fc.shape[0]:
        raise ValueError("permutation and paired expert shapes disagree")
    indices = permutation.to(c_fc.device)
    return c_fc.index_select(0, indices), c_proj.index_select(1, indices)


def complete_expert_similarity(
    left_router: torch.Tensor,
    left_c_fc: torch.Tensor,
    left_c_proj: torch.Tensor,
    right_router: torch.Tensor,
    right_c_fc: torch.Tensor,
    right_c_proj: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    """Joint router/output similarity for cross-run expert identity matching."""

    experts, hidden, width = left_c_fc.shape
    expected = (experts, hidden, width)
    if right_c_fc.shape != expected:
        raise ValueError("expert c_fc shapes disagree")
    if left_c_proj.shape != (experts, width, hidden) or right_c_proj.shape != (
        experts,
        width,
        hidden,
    ):
        raise ValueError("expert c_proj shapes disagree")
    if left_router.shape != (experts, width) or right_router.shape != (
        experts,
        width,
    ):
        raise ValueError("router shapes disagree")
    x = activations.float()
    left_logits = x @ left_router.float().T
    right_logits = x @ right_router.float().T
    router_similarity = _cosine_matrix(
        left_logits.T @ right_logits,
        left_logits.square().sum(dim=0).sqrt(),
        right_logits.square().sum(dim=0).sqrt(),
    )

    def outputs(c_fc: torch.Tensor, c_proj: torch.Tensor) -> torch.Tensor:
        values = []
        for expert in range(experts):
            hidden_values = F.gelu(x @ c_fc[expert].float().T)
            values.append((hidden_values @ c_proj[expert].float().T).flatten())
        return torch.stack(values)

    left_outputs = outputs(left_c_fc, left_c_proj)
    right_outputs = outputs(right_c_fc, right_c_proj)
    output_similarity = _cosine_matrix(
        left_outputs @ right_outputs.T,
        left_outputs.square().sum(dim=1).sqrt(),
        right_outputs.square().sum(dim=1).sqrt(),
    )
    return 0.5 * router_similarity + 0.5 * output_similarity
