import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .common import RMSNorm

# Recurrence: h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t ;  y_t = C_t . h_t + D * u_t
# Shapes: u, delta (b, l, d) | A (d, n) | B, C (b, l, n) | D (d,)


def selective_scan_sequential(u, delta, A, B, C, D):
    """Reference implementation: one time step at a time. Slow, but easy to read and trust."""
    b, l, d = u.shape
    dA = torch.exp(delta.unsqueeze(-1) * A)  # (b, l, d, n)
    dBu = (delta * u).unsqueeze(-1) * B.unsqueeze(2)  # (b, l, d, n)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for t in range(l):
        h = dA[:, t] * h + dBu[:, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, dim=1) + u * D


def selective_scan_parallel(u, delta, A, B, C, D):
    """Parallel (Hillis-Steele) scan over the affine maps h -> a*h + b, which compose associatively:
    (a2, b2) o (a1, b1) = (a2*a1, a2*b1 + b2). O(log l) sequential steps instead of l.
    Memory-hungry (materialises (b, l, d, n)); fine for tests and small runs."""
    l = u.shape[1]
    a = torch.exp(delta.unsqueeze(-1) * A)
    h = (delta * u).unsqueeze(-1) * B.unsqueeze(2)
    shift = 1
    while shift < l:
        a_prev = torch.cat([torch.ones_like(a[:, :shift]), a[:, :-shift]], dim=1)
        h_prev = torch.cat([torch.zeros_like(h[:, :shift]), h[:, :-shift]], dim=1)
        h = a * h_prev + h
        a = a * a_prev
        shift *= 2
    return torch.einsum("bldn,bln->bld", h, C) + u * D


def selective_scan_step(h, u, delta, A, B, C, D):
    """One recurrent step for decoding. h (b, d, n) | u, delta (b, d) | B, C (b, n)."""
    h = torch.exp(delta.unsqueeze(-1) * A) * h + (delta * u).unsqueeze(-1) * B.unsqueeze(1)
    return h, torch.einsum("bdn,bn->bd", h, C) + u * D


SCANS = {"sequential": selective_scan_sequential, "parallel": selective_scan_parallel}


def _inv_softplus_dt(d: int, lo: float = 1e-3, hi: float = 1e-1):
    """Sample delta log-uniformly in [lo, hi], return softplus^-1(delta) so softplus(bias) = delta."""
    dt = torch.exp(torch.rand(d) * (math.log(hi) - math.log(lo)) + math.log(lo)).clamp(min=1e-4)
    return dt + torch.log(-torch.expm1(-dt))


class MambaBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, di, n = cfg.d_model, cfg.inner, cfg.d_state
        self.d_inner, self.d_state, self.d_conv, self.rank = di, n, cfg.d_conv, cfg.rank
        self.selective = cfg.selective
        self.scan = SCANS[cfg.scan]

        self.in_proj = nn.Linear(d, 2 * di, bias=False)  # -> (x, z)
        self.conv1d = nn.Conv1d(di, di, cfg.d_conv, groups=di, padding=cfg.d_conv - 1)
        self.out_proj = nn.Linear(di, d, bias=False)
        # A = -exp(A_log) is diagonal and negative real (S4D-real init: -1..-n), so exp(delta*A) in (0, 1)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float32)).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))

        dt_bias = _inv_softplus_dt(di)
        if cfg.selective:
            self.x_proj = nn.Linear(di, self.rank + 2 * n, bias=False)  # x -> (dt, B, C)
            self.dt_proj = nn.Linear(self.rank, di)
            self.dt_proj._no_reinit = True
            nn.init.uniform_(self.dt_proj.weight, -self.rank**-0.5, self.rank**-0.5)
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt_bias)
        else:  # ablation: delta, B, C are learned constants, so the SSM is linear time-invariant
            self.dt_bias = nn.Parameter(dt_bias)
            self.B_const = nn.Parameter(torch.ones(n))
            self.C_const = nn.Parameter(torch.randn(n) / math.sqrt(n))

    def _ssm_params(self, x):
        """x: (b, l, d_inner) -> delta (b, l, d_inner), B (b, l, n), C (b, l, n)."""
        if self.selective:
            dt, B, C = self.x_proj(x).split([self.rank, self.d_state, self.d_state], dim=-1)
            return F.softplus(self.dt_proj(dt)), B, C
        b, l, _ = x.shape
        return (
            F.softplus(self.dt_bias).expand(b, l, -1),
            self.B_const.expand(b, l, -1),
            self.C_const.expand(b, l, -1),
        )

    def forward(self, x):
        l = x.shape[1]
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xi = F.silu(self.conv1d(xi.transpose(1, 2))[..., :l].transpose(1, 2))  # causal depthwise conv
        delta, B, C = self._ssm_params(xi)
        A = -torch.exp(self.A_log.float())
        y = self.scan(xi.float(), delta.float(), A, B.float(), C.float(), self.D.float())
        return self.out_proj(y.to(z.dtype) * F.silu(z))

    def init_state(self, batch: int, device=None, dtype=torch.float32):
        return (
            torch.zeros(batch, self.d_inner, self.d_conv, device=device, dtype=dtype),  # conv window
            torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=torch.float32),  # SSM state
        )

    def step(self, x, state):
        """Recurrent decode, O(1) per token. x: (b, d_model) -> (y (b, d_model), new_state)."""
        conv_state, h = state
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        conv_state = torch.cat([conv_state[:, :, 1:], xi.unsqueeze(-1).to(conv_state.dtype)], dim=-1)
        xi = F.silu((conv_state * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias)
        delta, B, C = self._ssm_params(xi.unsqueeze(1))
        A = -torch.exp(self.A_log.float())
        h, y = selective_scan_step(
            h, xi.float(), delta[:, 0].float(), A, B[:, 0].float(), C[:, 0].float(), self.D.float()
        )
        return self.out_proj(y.to(z.dtype) * F.silu(z)), (conv_state, h)


class MambaLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.block = MambaBlock(cfg)

    def forward(self, x):
        return x + self.block(self.norm(x))

    def init_state(self, *args, **kwargs):
        return self.block.init_state(*args, **kwargs)

    def step(self, x, state):
        y, state = self.block.step(self.norm(x), state)
        return x + y, state
