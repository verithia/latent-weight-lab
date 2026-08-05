from __future__ import annotations

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


def muon_update(
    update: torch.Tensor,
    *,
    steps: int,
    row_splits: tuple[int, ...] | None = None,
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
        orthogonal = zeropower_via_newtonschulz5(update, steps=steps)
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
        orthogonal = zeropower_via_newtonschulz5(block, steps=steps)
        columns = max(1, orthogonal.numel() / orthogonal.shape[0])
        scale = max(1.0, orthogonal.shape[0] / columns) ** 0.5
        blocks.append(orthogonal.mul(scale))
        start += rows
    return torch.cat(blocks, dim=0)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
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
                update = muon_update(
                    update,
                    steps=ns_steps,
                    row_splits=getattr(param, "_muon_row_splits", None),
                )
                param.add_(update, alpha=-lr)
        return loss
