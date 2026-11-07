"""Write parameter-matched run configs: configs/{S,M,L}_{transformer,mamba,mamba_lti}.json.

Mamba uses 2 layers per Transformer layer (one Mamba block replaces attention + MLP), and d_inner
is binary-searched so its non-embedding parameter count matches the Transformer's.

    python scripts/make_configs.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afml.models import ModelConfig, build_model, count_params  # noqa: E402

VOCAB = 16384
SCALES = {
    #      d_model, tf_layers, heads, train settings
    "S": (384, 6, 6, {"micro_batch": 16, "grad_accum": 16, "total_tokens": 340_000_000, "lr": 1e-3,
                      "eval_every": 100}),
    "M": (512, 12, 8, {"micro_batch": 16, "grad_accum": 32, "total_tokens": 940_000_000, "lr": 6e-4,
                       "eval_every": 150}),
    "L": (768, 12, 12, {"micro_batch": 8, "grad_accum": 64, "total_tokens": 1_500_000_000, "lr": 4e-4,
                        "eval_every": 150}),
}


def non_embedding(cfg):
    return count_params(build_model(cfg), exclude_embedding=True)


def match_d_inner(base, target, lo=64, hi=8192):
    lo, hi = lo // 8, hi // 8
    while lo < hi:
        mid = (lo + hi) // 2
        if non_embedding(replace(base, d_inner=mid * 8)) < target:
            lo = mid + 1
        else:
            hi = mid
    return min((lo - 1, lo), key=lambda k: abs(non_embedding(replace(base, d_inner=k * 8)) - target)) * 8


def main():
    table = []
    for scale, (d, layers, heads, train) in SCALES.items():
        tf = ModelConfig(arch="transformer", vocab_size=VOCAB, d_model=d, n_layers=layers, n_heads=heads)
        target = non_embedding(tf)
        models = {"transformer": {"arch": "transformer", "d_model": d, "n_layers": layers, "n_heads": heads}}
        for name, selective in (("mamba", True), ("mamba_lti", False)):
            base = ModelConfig(arch="mamba", vocab_size=VOCAB, d_model=d, n_layers=2 * layers, selective=selective)
            di = match_d_inner(base, target)
            models[name] = {"arch": "mamba", "d_model": d, "n_layers": 2 * layers, "d_inner": di,
                            "d_state": 16, "selective": selective, "scan": "auto"}
        for name, m in models.items():
            cfg = {"model": m, "train": {"seq_len": 1024, **train}}
            (ROOT / "configs" / f"{scale}_{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
            mc = ModelConfig.from_dict({**m, "vocab_size": VOCAB})
            model = build_model(mc)
            ne, tot = count_params(model, True), count_params(model)
            table.append((scale, name, m["n_layers"], m.get("d_inner", "-"), ne, tot, (ne - target) / target))
    print(f"{'scale':<6}{'model':<13}{'layers':>7}{'d_inner':>9}{'non-emb':>13}{'total':>13}{'gap':>9}")
    for s, n, l, di, ne, tot, gap in table:
        print(f"{s:<6}{n:<13}{l:>7}{di!s:>9}{ne:>13,}{tot:>13,}{gap:>9.2%}")


if __name__ == "__main__":
    main()
