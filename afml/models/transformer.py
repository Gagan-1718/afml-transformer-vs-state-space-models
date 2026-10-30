import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .common import RMSNorm


def _rope_cache(seq_len: int, head_dim: int, base: float):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
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
        cos, sin = _rope_cache(cfg.max_seq_len, self.head_dim, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if t > self.cos.shape[0]:  # longer than max_seq_len (length-extrapolation eval)
            cos, sin = _rope_cache(t, self.head_dim, self.rope_base)
            cos, sin = cos.to(x.device), sin.to(x.device)
        else:
            cos, sin = self.cos[:t], self.sin[:t]
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))


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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
