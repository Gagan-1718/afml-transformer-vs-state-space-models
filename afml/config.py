import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    arch: str = "transformer"  # "transformer" | "mamba"
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 8
    max_seq_len: int = 1024

    # transformer
    n_heads: int = 8
    mlp_multiple_of: int = 64
    rope_base: float = 10000.0

    # mamba
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    d_inner: Optional[int] = None  # overrides expand * d_model; the knob used to parameter-match
    dt_rank: Optional[int] = None  # defaults to ceil(d_model / 16)
    selective: bool = True  # False -> input-independent (S4-style) ablation
    scan: str = "parallel"  # "parallel" | "sequential"

    @property
    def inner(self) -> int:
        return self.d_inner or self.expand * self.d_model

    @property
    def rank(self) -> int:
        return self.dt_rank or math.ceil(self.d_model / 16)
