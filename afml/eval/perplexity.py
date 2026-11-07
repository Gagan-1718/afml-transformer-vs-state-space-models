"""Test-set perplexity, split into associative-recall hits and other tokens.

A target token is a recall hit when the bigram (previous token, target) already appeared earlier in
the same context window, following Zoology (Arora et al., 2023). Attention-free models are expected
to lose more on those tokens.

    python -m afml.eval.perplexity --runs runs/S_* --data DATA --out results/S/perplexity.csv
"""
import argparse
import csv
import math
from pathlib import Path

import torch

from ..data import TokenData
from .common import device_auto, load_model, run_label


def recall_hit_mask(x, y, vocab_size):
    """x, y: (b, l) inputs and targets. True where the pair (x_t, y_t) occurred at an earlier position."""
    b, l = x.shape
    pairs = x.long() * vocab_size + y.long()
    hits = torch.zeros(b, l, dtype=torch.bool, device=x.device)
    pos = torch.arange(l, device=x.device)
    for i in range(b):
        _, inv = torch.unique(pairs[i], return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), l, dtype=torch.long, device=x.device)
        first = first.scatter_reduce(0, inv, pos, reduce="amin")
        hits[i] = first[inv] < pos
    return hits


@torch.no_grad()
def evaluate_split(model, x, y, device, batch=8):
    sums = {"all": [0.0, 0], "hit": [0.0, 0], "other": [0.0, 0]}
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
        _, loss = model(xb, yb, reduction="none")
        hit = recall_hit_mask(xb, yb, model.cfg.vocab_size)
        for key, mask in (("all", torch.ones_like(hit)), ("hit", hit), ("other", ~hit)):
            sums[key][0] += loss[mask].sum().item()
            sums[key][1] += int(mask.sum())
    out = {}
    for key, (s, n) in sums.items():
        out[f"{key}_loss"] = s / max(n, 1)
        out[f"{key}_tokens"] = n
    out["ppl"] = math.exp(out["all_loss"])
    out["hit_fraction"] = out["hit_tokens"] / out["all_tokens"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tokens", type=int, default=5_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = device_auto(a.device)
    x, y = TokenData(a.data).eval_windows(a.split, a.seq_len, a.tokens)
    rows = []
    for run in a.runs:
        arch, summary = run_label(run)
        model = load_model(run, device=device)
        r = {"run": Path(run).name, "arch": arch, "params_total": summary["params_total"],
             "params_nonembedding": summary["params_nonembedding"], "seed": summary["train_config"]["seed"],
             "lr": summary["train_config"]["lr"], "train_tokens": summary["tokens"],
             "train_flops": summary["train_flops"], "split": a.split,
             **evaluate_split(model, x, y, device)}
        print(r, flush=True)
        rows.append(r)
        del model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
