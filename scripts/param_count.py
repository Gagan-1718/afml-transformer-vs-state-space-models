"""Parameter-match the Mamba models to the Transformer baseline by tuning d_inner.

    python scripts/param_count.py --vocab 16384 --d-model 512 --tf-layers 8 --mamba-layers 16
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afml.models import ModelConfig, build_model, count_params  # noqa: E402


def non_embedding(cfg: ModelConfig) -> int:
    return count_params(build_model(cfg), exclude_embedding=True)


def match_d_inner(base: ModelConfig, target: int, lo: int = 64, hi: int = 4096) -> int:
    """Binary-search d_inner (multiples of 8) so the non-embedding count is closest to target."""
    lo, hi = lo // 8, hi // 8
    while lo < hi:
        mid = (lo + hi) // 2
        if non_embedding(replace(base, d_inner=mid * 8)) < target:
            lo = mid + 1
        else:
            hi = mid
    return min((lo - 1, lo), key=lambda k: abs(non_embedding(replace(base, d_inner=k * 8)) - target)) * 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--tf-layers", type=int, default=8)
    ap.add_argument("--mamba-layers", type=int, default=16)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    a = ap.parse_args()

    common = dict(vocab_size=a.vocab, d_model=a.d_model, max_seq_len=a.seq_len)
    tf = ModelConfig(arch="transformer", n_layers=a.tf_layers, n_heads=a.n_heads, **common)
    target = non_embedding(tf)

    rows = [("transformer", tf, "")]
    for name, selective in [("mamba", True), ("mamba-LTI (ablation)", False)]:
        base = ModelConfig(arch="mamba", n_layers=a.mamba_layers, d_state=a.d_state, selective=selective, **common)
        d_inner = match_d_inner(base, target)
        rows.append((name, replace(base, d_inner=d_inner), f"d_inner={d_inner}"))

    print(f"{'model':<22}{'layers':>7}{'non-emb params':>16}{'total params':>14}{'vs transformer':>16}  notes")
    for name, cfg, note in rows:
        model = build_model(cfg)
        ne, tot = count_params(model, True), count_params(model)
        print(f"{name:<22}{cfg.n_layers:>7}{ne:>16,}{tot:>14,}{(ne - target) / target:>15.2%}  {note}")


if __name__ == "__main__":
    main()
