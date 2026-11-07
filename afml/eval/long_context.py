"""Length extrapolation: loss at each token position on long PG-19 books.

Models are trained at 1,024 tokens. Each book's first `--max-len` tokens are scored in one forward
pass, and per-token losses are averaged into position bins. A model that uses (or survives) context
beyond its training length keeps a flat or falling curve past 1,024.

    python -m afml.eval.long_context --runs runs/M_* --data DATA --max-len 16384 --out results/M/long_context.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ..data import TokenData
from .common import device_auto, load_model, run_label


def position_bins(max_len, first=128):
    edges = [0]
    e = first
    while e < max_len:
        edges.append(e)
        e *= 2
    edges.append(max_len)
    return edges


@torch.no_grad()
def per_position_loss(model, books, max_len, device):
    total = torch.zeros(max_len, dtype=torch.float64)
    count = 0
    for book in books:
        if len(book) < max_len + 1:
            continue
        ids = torch.from_numpy(np.asarray(book[:max_len + 1], dtype=np.int64)).unsqueeze(0).to(device)
        _, loss = model(ids[:, :-1], ids[:, 1:], reduction="none")
        total += loss[0].double().cpu()
        count += 1
    if count == 0:
        raise ValueError(f"no book has {max_len + 1} tokens")
    return total / count, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = device_auto(a.device)
    books = TokenData(a.data).pg19_books()
    edges = position_bins(a.max_len)
    rows = []
    for run in a.runs:
        arch, s = run_label(run)
        model = load_model(run, device=device)
        try:
            losses, n_books = per_position_loss(model, books, a.max_len, device)
        except torch.cuda.OutOfMemoryError:
            print(f"{run}: out of memory at {a.max_len}", flush=True)
            torch.cuda.empty_cache()
            continue
        for lo, hi in zip(edges[:-1], edges[1:]):
            rows.append({"run": Path(run).name, "arch": arch, "params_total": s["params_total"],
                         "train_seq_len": s["train_config"]["seq_len"], "pos_start": lo, "pos_end": hi,
                         "loss": losses[lo:hi].mean().item(), "books": n_books})
            print(rows[-1], flush=True)
        del model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
