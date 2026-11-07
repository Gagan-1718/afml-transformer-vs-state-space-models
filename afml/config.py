import math
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class ModelConfig:
    arch: str = "transformer"  # "transformer" | "mamba"
    vocab_size: int = 16384
    d_model: int = 384
    n_layers: int = 6
    max_seq_len: int = 1024  # RoPE cache length; longer inputs are still supported

    # transformer
    n_heads: int = 6
    mlp_multiple_of: int = 64
    rope_base: float = 10000.0

    # mamba
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    d_inner: Optional[int] = None  # overrides expand * d_model; the knob used to parameter-match
    dt_rank: Optional[int] = None  # defaults to ceil(d_model / 16)
    selective: bool = True  # False -> input-independent (S4-style, LTI) ablation
    scan: str = "auto"  # "auto" | "chunked" | "cuda" | "parallel" | "sequential"
    chunk_size: int = 64  # chunked scan only

    @property
    def inner(self) -> int:
        return self.d_inner or self.expand * self.d_model

    @property
    def rank(self) -> int:
        return self.dt_rank or math.ceil(self.d_model / 16)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
        return cls(**d)
