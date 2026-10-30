import math

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

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def build_model(cfg: ModelConfig) -> LM:
    return LM(cfg)


def count_params(model: LM, exclude_embedding: bool = False) -> int:
    n = sum(p.numel() for p in model.parameters())  # tied head is counted once
    return n - model.embed.weight.numel() if exclude_embedding else n
