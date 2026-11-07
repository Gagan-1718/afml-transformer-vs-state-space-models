import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .common import RMSNorm


def _rope_table(start: int, length: int, head_dim: int, base: float, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(start, start + length, device=device).float()
    emb = torch.outer(pos, inv_freq).repeat(1, 2)
    return emb.cos(), emb.sin()


def _apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.rope_base = cfg.rope_base
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        cos, sin = _rope_table(0, cfg.max_seq_len, self.head_dim, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.record_attention = False  # analysis only
        self.last_attention = None

    def _rope(self, start, t, device, dtype):
        if start + t <= self.cos.shape[0]:
            cos, sin = self.cos[start:start + t], self.sin[start:start + t]
        else:  # beyond the cached length (length-extrapolation eval and long generation)
            cos, sin = _rope_table(start, t, self.head_dim, self.rope_base, device)
        return cos.to(dtype), sin.to(dtype)

    def _qkv(self, x, start):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = self._rope(start, t, x.device, q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin), v

    def forward(self, x, cache=None):
        b, t, d = x.shape
        q, k, v = self._qkv(x, 0)
        if self.record_attention:
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            self.last_attention = scores.masked_fill(mask, float("-inf")).softmax(-1).cpu()
        if cache is not None:  # prefill: store keys and values for later decoding
            cache["k"][:, :, :t] = k
            cache["v"][:, :, :t] = v
            cache["pos"] = t
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))

    def step(self, x, cache):
        """x: (b, d_model), one new token at position cache['pos']."""
        b, d = x.shape
        p = cache["pos"]
        if p >= cache["k"].shape[2]:
            raise RuntimeError("KV cache is full; allocate a larger max_len")
        q, k, v = self._qkv(x.unsqueeze(1), p)
        cache["k"][:, :, p:p + 1] = k
        cache["v"][:, :, p:p + 1] = v
        cache["pos"] = p + 1
        y = F.scaled_dot_product_attention(q, cache["k"][:, :, :p + 1], cache["v"][:, :, :p + 1])
        return self.out_proj(y.reshape(b, d))

    def init_cache(self, batch, max_len, device, dtype):
        shape = (batch, self.n_heads, max_len, self.head_dim)
        return {"k": torch.zeros(shape, device=device, dtype=dtype),
                "v": torch.zeros(shape, device=device, dtype=dtype), "pos": 0}


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        hidden = cfg.mlp_multiple_of * math.ceil(hidden / cfg.mlp_multiple_of)
        self.gate_up = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cache=None):
        x = x + self.attn(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))

    def init_cache(self, batch, max_len, device, dtype):
        return self.attn.init_cache(batch, max_len, device, dtype)

    def step(self, x, cache):
        x = x + self.attn.step(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))
