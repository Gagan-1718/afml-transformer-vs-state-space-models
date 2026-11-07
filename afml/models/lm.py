import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .common import RMSNorm
from .mamba import MambaLayer
from .transformer import TransformerLayer

LAYERS = {"transformer": TransformerLayer, "mamba": MambaLayer}


class LM(nn.Module):
    """Embedding -> N x (Transformer | Mamba) layer -> RMSNorm -> tied LM head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(LAYERS[cfg.arch](cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # GPT-2 style: shrink the residual-branch outputs
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if getattr(m, "_no_reinit", False):
            return
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def hidden(self, idx, caches=None):
        x = self.embed(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i])
        return self.norm_f(x)

    def forward(self, idx, targets=None, reduction="mean"):
        logits = self.lm_head(self.hidden(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   reduction=reduction)
            if reduction == "none":
                loss = loss.view(targets.shape)
        return logits, loss

    # ---- cached inference: KV cache for attention, recurrent state for Mamba ----
    def init_cache(self, batch, max_len, device=None, dtype=torch.float32):
        device = device or self.embed.weight.device
        return [layer.init_cache(batch, max_len, device, dtype) for layer in self.layers]

    @torch.no_grad()
    def prefill(self, idx, caches):
        """Run the prompt once, filling the caches. Returns logits for the last position."""
        return self.lm_head(self.hidden(idx, caches)[:, -1])

    @torch.no_grad()
    def decode_step(self, token, caches):
        """token: (b,) ids -> logits (b, vocab) for the next position."""
        x = self.embed(token)
        for layer, cache in zip(self.layers, caches):
            x = layer.step(x, cache)
        return self.lm_head(self.norm_f(x))

    def cache_bytes(self, batch, seq_len, bytes_per=2):
        """Analytic inference-state size: KV cache grows with seq_len, the SSM state does not."""
        c = self.cfg
        if c.arch == "transformer":
            return 2 * c.n_layers * batch * seq_len * c.d_model * bytes_per
        return c.n_layers * batch * c.inner * (c.d_state + c.d_conv) * bytes_per


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40, generator=None, banned=()):
    """Sample with caching. idx: (b, t) prompt ids. Returns (b, t + max_new_tokens)."""
    model.eval()
    b, t = idx.shape
    caches = model.init_cache(b, t + max_new_tokens)
    logits = model.prefill(idx, caches)
    out = [idx]
    for i in range(max_new_tokens):
        scores = logits.float() / max(temperature, 1e-6)
        for tok in banned:
            scores[:, tok] = -float("inf")
        if top_k:
            cutoff = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
            scores = scores.masked_fill(scores < cutoff, -float("inf"))
        nxt = torch.multinomial(scores.softmax(-1), 1, generator=generator)
        out.append(nxt)
        if i + 1 < max_new_tokens:
            logits = model.decode_step(nxt[:, 0], caches)
    return torch.cat(out, dim=1)


def build_model(cfg: ModelConfig) -> LM:
    return LM(cfg)


def count_params(model: LM, exclude_embedding: bool = False) -> int:
    n = sum(p.numel() for p in model.parameters())  # tied head is counted once
    return n - model.embed.weight.numel() if exclude_embedding else n


def param_groups(model: nn.Module, weight_decay: float):
    """Decay weight matrices only. No decay on norms, biases, embeddings, A_log, D, delta bias."""
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim < 2 or getattr(p, "_no_weight_decay", False) or name.startswith("embed."):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def train_flops_per_token(cfg: ModelConfig, n_params_nonemb: int, seq_len: int) -> float:
    """Approximate training FLOPs per token (forward + backward = 3 x forward).
    Dense: 6N (Kaplan et al.). Attention adds 6 * layers * seq_len * d_model.
    The SSM scan adds about 3 * layers * 9 * d_inner * d_state (Gu & Dao, O(BLDN))."""
    flops = 6 * n_params_nonemb + 6 * cfg.vocab_size * cfg.d_model  # include the LM head matmul
    if cfg.arch == "transformer":
        flops += 6 * cfg.n_layers * seq_len * cfg.d_model
    else:
        flops += 3 * cfg.n_layers * 9 * cfg.inner * cfg.d_state
    return float(flops)
