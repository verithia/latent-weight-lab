from __future__ import annotations

import math

import torch


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor of a matrix gradient.

    This is the standard Muon-style Newton-Schulz orthogonalization used for
    matrix updates. It is intentionally applied only to tensors that can be
    viewed as matrices; vectors should use AdamW.
    """
    if grad.ndim < 2:
        raise ValueError("Muon orthogonalization requires at least 2D tensors")
    original_shape = grad.shape
    x = grad.float().reshape(grad.shape[0], -1)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x
    if transposed:
        x = x.T
    return x.reshape(original_shape).to(dtype=grad.dtype)


def ridge_smoothed_zeropower_via_newtonschulz5(
    grad: torch.Tensor,
    *,
    steps: int = 5,
    relative_rms_ridge: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return a fast, norm-restored ridge-smoothed polar update.

    For a Frobenius-normalized ``m x n`` request ``X`` with ``m <= n``, the
    exact polar factor of ``[X, tau I]`` contains

        ``(X X^T + tau^2 I)^(-1/2) X``

    in its first ``n`` columns.  Applying the ordinary Muon NS5 iteration to
    this augmented matrix therefore regularizes the angular singularity at
    rank loss without adding persistent state.  ``tau = rho / sqrt(m)`` is
    the same scale-free ridge convention as the sealed spectral oracle.  The
    retained block is restored to Frobenius norm ``sqrt(m)`` so the
    intervention changes direction rather than optimizer-step energy.

    A zero ridge deliberately dispatches to the native implementation so all
    existing recipes keep their exact code path.
    """
    ridge = float(relative_rms_ridge)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("Muon relative RMS polar ridge must be finite and non-negative")
    if ridge == 0.0:
        return zeropower_via_newtonschulz5(grad, steps=steps, eps=eps)
    if grad.ndim < 2:
        raise ValueError("Muon orthogonalization requires at least 2D tensors")
    original_shape = grad.shape
    x = grad.float().reshape(grad.shape[0], -1)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    rows, columns = x.shape
    x = x / (x.norm() + eps)
    tau = ridge / math.sqrt(float(rows))
    identity = torch.eye(rows, device=x.device, dtype=x.dtype).mul_(tau)
    augmented = torch.cat((x, identity), dim=1)
    augmented = augmented / (augmented.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = augmented @ augmented.T
        augmented = a * augmented + (b * gram + c * gram @ gram) @ augmented
    x = augmented[:, :columns]
    x = x * (math.sqrt(float(rows)) / x.norm().clamp_min(eps))
    if transposed:
        x = x.T
    return x.reshape(original_shape).to(dtype=grad.dtype)


def muon_update(
    update: torch.Tensor,
    *,
    steps: int,
    row_splits: tuple[int, ...] | None = None,
    polar_ridge: float = 0.0,
) -> torch.Tensor:
    """Return Muon's scaled polar update, optionally per contiguous row block.

    Packed QKV normally receives one polar map over its full ``3d x d``
    matrix.  A partial-QK model instead gives the dense value matrix its own
    ``d x d`` Muon update.  ``row_splits=(2d, d)`` is the exact packed-weight
    control for that optimizer geometry: momentum remains attached to the
    original parameter, while QK and V are orthogonalized and scaled as two
    independent matrices.
    """

    if row_splits is None:
        orthogonal = ridge_smoothed_zeropower_via_newtonschulz5(
            update,
            steps=steps,
            relative_rms_ridge=polar_ridge,
        )
        columns = max(1, orthogonal.numel() / orthogonal.shape[0])
        scale = max(1.0, orthogonal.shape[0] / columns) ** 0.5
        return orthogonal.mul(scale)
    if not row_splits or any(
        isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
        for rows in row_splits
    ):
        raise ValueError("Muon row splits must be positive integers")
    if sum(row_splits) != update.shape[0]:
        raise ValueError("Muon row splits must exactly cover the first dimension")
    blocks = []
    start = 0
    for rows in row_splits:
        block = update.narrow(0, start, rows)
        orthogonal = ridge_smoothed_zeropower_via_newtonschulz5(
            block,
            steps=steps,
            relative_rms_ridge=polar_ridge,
        )
        columns = max(1, orthogonal.numel() / orthogonal.shape[0])
        scale = max(1.0, orthogonal.shape[0] / columns) ** 0.5
        blocks.append(orthogonal.mul(scale))
        start += rows
    return torch.cat(blocks, dim=0)


def muon_update_batched(update: torch.Tensor, *, steps: int) -> torch.Tensor:
    """Apply the ordinary Muon polar update independently to a matrix batch."""
    if update.ndim != 3:
        raise ValueError("batched Muon expects [batch, rows, columns]")
    rows, columns = update.shape[-2:]
    x = update.float()
    transposed = rows > columns
    if transposed:
        x = x.transpose(-2, -1)
    x = x / (
        x.square().sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-7
    )
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        xx_t = torch.bmm(x, x.transpose(-2, -1))
        x = a * x + torch.bmm(b * xx_t + c * torch.bmm(xx_t, xx_t), x)
    if transposed:
        x = x.transpose(-2, -1)
    scale = max(1.0, rows / max(1, columns)) ** 0.5
    return x.mul(scale).to(dtype=update.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        polar_ridge: float = 0.0,
    ) -> None:
        if not math.isfinite(float(polar_ridge)) or float(polar_ridge) < 0.0:
            raise ValueError("Muon polar ridge must be finite and non-negative")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            polar_ridge=float(polar_ridge),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            polar_ridge = float(group["polar_ridge"])
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum)
                if getattr(param, "_muon_batched_matrices", False):
                    if polar_ridge != 0.0:
                        raise ValueError("ridge-smoothed Muon does not support batched matrices")
                    update = muon_update_batched(update, steps=ns_steps)
                else:
                    update = muon_update(
                        update,
                        steps=ns_steps,
                        row_splits=getattr(param, "_muon_row_splits", None),
                        polar_ridge=polar_ridge,
                    )
                param.add_(update, alpha=-lr)
        return loss
