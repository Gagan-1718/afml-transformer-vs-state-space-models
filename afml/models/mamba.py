"""Mamba-1 style selective state-space block with four interchangeable scan backends.

Recurrence (per channel d and state index n):
    h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
    y_t = C_t . h_t + D * u_t
Shapes: u, delta (b, l, d) | A (d, n) | B, C (b, l, n) | D (d,) | h (b, d, n)

Backends (all return (y, h_last) and must agree; see tests/test_scans.py):
  sequential  one step at a time. The readable reference.
  parallel    Hillis-Steele scan over the whole sequence. Fast for short inputs, but stores
              O(log L) tensors of size (b, l, d, n) for backward, so it cannot train at L=1024.
  chunked     splits the sequence into chunks, runs the parallel scan inside each chunk under
              activation checkpointing and carries the state across chunks. Only (b, l, d) inputs
              and (b, d, n) chunk-boundary states are kept for backward. Pure PyTorch; the default.
  cuda        the official fused kernel from `mamba-ssm` (recomputes states in backward).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .common import RMSNorm

def _load_cuda_scan():
    """The fused selective-scan kernel from the optional `mamba-ssm` package, or None.

    Only `mamba_ssm.ops.selective_scan_interface` is imported. The package's own __init__.py is skipped
    on purpose: it imports the whole library (models, generation utilities, transformers), which can
    break on a new environment even when the kernel itself works. Returns (fn, error message)."""
    import importlib
    import importlib.util
    import sys
    import types
    try:
        spec = importlib.util.find_spec("mamba_ssm")
    except (ImportError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"
    if spec is None or not spec.submodule_search_locations:
        return None, "mamba-ssm is not installed"
    added = "mamba_ssm" not in sys.modules
    if added:  # an empty package object, so importing a submodule does not run mamba_ssm/__init__.py
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = list(spec.submodule_search_locations)
        pkg.__spec__ = spec
        sys.modules["mamba_ssm"] = pkg
    try:
        mod = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        return mod.selective_scan_fn, None
    except Exception as e:  # noqa: BLE001 - any import failure means "not available"
        if added:
            for name in [m for m in sys.modules if m == "mamba_ssm" or m.startswith("mamba_ssm.")]:
                del sys.modules[name]
        return None, f"{type(e).__name__}: {e}"


_cuda_scan_fn, CUDA_SCAN_ERROR = _load_cuda_scan()
HAS_CUDA_SCAN = _cuda_scan_fn is not None


def _discretize(u, delta, A, B):
    """(b, l, d), (b, l, d), (d, n), (b, l, n) -> decay a and input x, both (b, l, d, n)."""
    a = torch.exp(delta.unsqueeze(-1) * A)
    x = (delta * u).unsqueeze(-1) * B.unsqueeze(2)
    return a, x


def selective_scan_sequential(u, delta, A, B, C, D, h0=None):
    b, l, d = u.shape
    a, x = _discretize(u, delta, A, B)
    h = u.new_zeros(b, d, A.shape[1]) if h0 is None else h0
    ys = []
    for t in range(l):
        h = a[:, t] * h + x[:, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, dim=1) + u * D, h


def _hillis_steele(a, x):
    """Inclusive scan of the affine maps h -> a*h + x along dim 1.
    (a2, x2) o (a1, x1) = (a2*a1, a2*x1 + x2). Returns (prod of a, state from h0 = 0)."""
    l = a.shape[1]
    shift = 1
    while shift < l:
        a_prev = F.pad(a[:, :-shift], (0, 0, 0, 0, shift, 0), value=1.0)
        x_prev = F.pad(x[:, :-shift], (0, 0, 0, 0, shift, 0), value=0.0)
        x = a * x_prev + x
        a = a * a_prev
        shift *= 2
    return a, x


def selective_scan_parallel(u, delta, A, B, C, D, h0=None):
    a, x = _discretize(u, delta, A, B)
    a_cum, h = _hillis_steele(a, x)
    if h0 is not None:
        h = h + a_cum * h0.unsqueeze(1)
    return torch.einsum("bldn,bln->bld", h, C) + u * D, h[:, -1]


def _scan_chunk(u, delta, A, B, C, h0):
    a, x = _discretize(u, delta, A, B)
    a_cum, h = _hillis_steele(a, x)
    h = h + a_cum * h0.unsqueeze(1)
    return torch.einsum("bldn,bln->bld", h, C), h[:, -1]


def selective_scan_chunked(u, delta, A, B, C, D, h0=None, chunk_size=64):
    b, l, d = u.shape
    h = u.new_zeros(b, d, A.shape[1]) if h0 is None else h0
    use_ckpt = torch.is_grad_enabled() and any(
        t.requires_grad for t in (u, delta, A, B, C, h))
    ys = []
    for s in range(0, l, chunk_size):
        args = (u[:, s:s + chunk_size], delta[:, s:s + chunk_size], A,
                B[:, s:s + chunk_size], C[:, s:s + chunk_size], h)
        if use_ckpt:
            y, h = checkpoint(_scan_chunk, *args, use_reentrant=False)
        else:
            y, h = _scan_chunk(*args)
        ys.append(y)
    return torch.cat(ys, dim=1) + u * D, h


def selective_scan_cuda(u, delta, A, B, C, D, h0=None):
    if _cuda_scan_fn is None:
        raise RuntimeError("scan='cuda' needs the mamba-ssm package (pip install mamba-ssm)")
    if h0 is not None:
        raise NotImplementedError("the fused kernel starts from a zero state")
    y, h = _cuda_scan_fn(
        u.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), A.contiguous(),
        B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(), D.contiguous(),
        return_last_state=True,
    )
    return y.transpose(1, 2), h


def selective_scan_step(h, u, delta, A, B, C, D):
    """One recurrent decoding step. h (b, d, n) | u, delta (b, d) | B, C (b, n)."""
    h = torch.exp(delta.unsqueeze(-1) * A) * h + (delta * u).unsqueeze(-1) * B.unsqueeze(1)
    return h, torch.einsum("bdn,bn->bd", h, C) + u * D


SCANS = {
    "sequential": selective_scan_sequential,
    "parallel": selective_scan_parallel,
    "chunked": selective_scan_chunked,
    "cuda": selective_scan_cuda,
}


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
        self.scan_name, self.chunk_size = cfg.scan, cfg.chunk_size
        if self.scan_name not in ("auto", *SCANS):
            raise ValueError(f"unknown scan {self.scan_name!r}")

        self.in_proj = nn.Linear(d, 2 * di, bias=False)  # -> (x, z)
        self.conv1d = nn.Conv1d(di, di, cfg.d_conv, groups=di, padding=cfg.d_conv - 1)
        self.out_proj = nn.Linear(di, d, bias=False)
        # A = -exp(A_log) is diagonal and negative real (S4D-real init: -1..-n), so exp(delta*A) in (0, 1)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float32)).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

        dt_bias = _inv_softplus_dt(di)
        if cfg.selective:
            self.x_proj = nn.Linear(di, self.rank + 2 * n, bias=False)  # x -> (dt, B, C)
            self.dt_proj = nn.Linear(self.rank, di)
            self.dt_proj._no_reinit = True
            nn.init.uniform_(self.dt_proj.weight, -self.rank**-0.5, self.rank**-0.5)
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt_bias)
            self.dt_proj.bias._no_weight_decay = True
        else:  # ablation: delta, B, C are learned constants, so the SSM is linear time-invariant
            self.dt_bias = nn.Parameter(dt_bias)
            self.B_const = nn.Parameter(torch.ones(n))
            self.C_const = nn.Parameter(torch.randn(n) / math.sqrt(n))
            for p in (self.dt_bias, self.B_const, self.C_const):
                p._no_weight_decay = True
        self.last_delta = None  # filled when record_delta is True (analysis only)
        self.record_delta = False

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

    def _pick_scan(self, x):
        if self.scan_name != "auto":
            return self.scan_name
        return "cuda" if (HAS_CUDA_SCAN and x.is_cuda) else "chunked"

    def _scan(self, name, *args):
        if name == "chunked":
            return selective_scan_chunked(*args, chunk_size=self.chunk_size)
        return SCANS[name](*args)

    def forward(self, x, return_state=False):
        l = x.shape[1]
        xz = self.in_proj(x)
        x_raw, z = xz.chunk(2, dim=-1)
        xi = F.silu(self.conv1d(x_raw.transpose(1, 2))[..., :l].transpose(1, 2))  # causal depthwise conv
        delta, B, C = self._ssm_params(xi)
        if self.record_delta:
            self.last_delta = delta.detach().float().cpu()
        with torch.autocast(device_type=x.device.type, enabled=False):  # the scan stays in fp32
            A = -torch.exp(self.A_log.float())
            y, h = self._scan(self._pick_scan(x), xi.float(), delta.float(), A, B.float(), C.float(),
                              self.D.float())
        out = self.out_proj(y.to(z.dtype) * F.silu(z))
        if not return_state:
            return out
        conv_state = x_raw.transpose(1, 2)[..., -self.d_conv:]
        if conv_state.shape[-1] < self.d_conv:
            conv_state = F.pad(conv_state, (self.d_conv - conv_state.shape[-1], 0))
        return out, (conv_state.float().contiguous(), h.float())

    def init_state(self, batch: int, device=None, dtype=torch.float32):
        return (
            torch.zeros(batch, self.d_inner, self.d_conv, device=device, dtype=torch.float32),
            torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=torch.float32),
        )

    def step(self, x, state):
        """Recurrent decode, O(1) memory per token. x: (b, d_model) -> (y (b, d_model), new_state)."""
        conv_state, h = state
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        conv_state = torch.cat([conv_state[:, :, 1:], xi.unsqueeze(-1).float()], dim=-1)
        w = self.conv1d.weight.squeeze(1).float()
        xi = F.silu((conv_state * w).sum(-1) + self.conv1d.bias.float())
        delta, B, C = self._ssm_params(xi.to(self.in_proj.weight.dtype).unsqueeze(1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            A = -torch.exp(self.A_log.float())
            h, y = selective_scan_step(h, xi.float(), delta[:, 0].float(), A, B[:, 0].float(),
                                       C[:, 0].float(), self.D.float())
        return self.out_proj(y.to(z.dtype) * F.silu(z)), (conv_state, h)

    def state_bytes(self, batch: int = 1, bytes_per=4) -> int:
        return batch * self.d_inner * (self.d_state + self.d_conv) * bytes_per


class MambaLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.block = MambaBlock(cfg)

    def forward(self, x, cache=None):
        if cache is None:
            return x + self.block(self.norm(x))
        y, state = self.block(self.norm(x), return_state=True)
        cache["state"] = state
        return x + y

    def init_cache(self, batch, max_len, device, dtype):
        return {"state": self.block.init_state(batch, device)}

    def step(self, x, cache):
        y, cache["state"] = self.block.step(self.norm(x), cache["state"])
        return x + y
