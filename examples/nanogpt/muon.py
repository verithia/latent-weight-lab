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


def zeropower_batched_via_newtonschulz5(
    grad: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Apply the Muon polar approximation independently to matrix blocks.

    ``grad`` has shape ``[groups, rows, columns]``.  Treating it as one
    ``groups x (rows*columns)`` matrix changes the optimizer geometry and
    couples otherwise independent local maps.  This routine is exactly the
    vectorized analogue of calling :func:`zeropower_via_newtonschulz5` once
    for every group, while using batched GEMMs instead of Python launches.
    """

    if grad.ndim != 3:
        raise ValueError("grouped Muon orthogonalization requires a 3D tensor")
    x = grad.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    norm = x.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    x = x / (norm + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        xx_t = x @ x.transpose(-2, -1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x.to(dtype=grad.dtype)


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
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = max(1.0, update.shape[0] / max(1, update.numel() / update.shape[0])) ** 0.5
                param.add_(update, alpha=-lr * scale)
        return loss


class GroupedMuon(torch.optim.Optimizer):
    """Muon with one independent polar update per leading-axis matrix."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
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
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if param.ndim != 3:
                    raise ValueError("GroupedMuon parameters must be 3D")
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum)
                update = zeropower_batched_via_newtonschulz5(
                    update,
                    steps=ns_steps,
                )
                rows = update.shape[-2]
                columns = update.shape[-1]
                scale = max(1.0, rows / max(1, columns)) ** 0.5
                param.add_(update, alpha=-lr * scale)
        return loss
