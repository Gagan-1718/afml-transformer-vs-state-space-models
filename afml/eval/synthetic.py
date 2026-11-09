"""Synthetic recall tasks that separate attention from fixed-size recurrent state.

MQAR (multi-query associative recall; Arora et al., 2023, "Zoology"): the sequence starts with K
key-value pairs; later, keys reappear as queries and the model must output the paired value.
Attention can look the key up; a recurrent model must have stored all K pairs in its state.

Selective copying (Gu & Dao, 2023): M data tokens are scattered among noise tokens; after a
separator the model must reproduce them in order. A time-invariant (LTI) SSM cannot choose which
inputs to keep, so it should fail; the selective model should succeed.

    python -m afml.eval.synthetic --task mqar --archs transformer mamba mamba_lti \
        --d-models 64 128 256 --seq-lens 64 128 256 512 --out results/synthetic/mqar.csv
"""
import argparse
import csv
import itertools
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import ModelConfig
from ..models import build_model, param_groups

IGNORE = -100


def make_mqar(n, seq_len, num_kv, vocab, generator):
    """Returns x (n, L) and y (n, L); y is IGNORE except at query positions."""
    half = vocab // 2
    keys = torch.stack([torch.randperm(half - 1, generator=generator)[:num_kv] + 1 for _ in range(n)])
    vals = torch.randint(half, vocab, (n, num_kv), generator=generator)
    x = torch.zeros(n, seq_len, dtype=torch.long)
    y = torch.full((n, seq_len), IGNORE, dtype=torch.long)
    x[:, 0:2 * num_kv:2], x[:, 1:2 * num_kv:2] = keys, vals
    slots = (seq_len - 2 * num_kv) // 2
    if slots < num_kv:
        raise ValueError(f"seq_len {seq_len} too short for {num_kv} queries")
    for i in range(n):
        pos = 2 * num_kv + 2 * torch.randperm(slots, generator=generator)[:num_kv]
        order = torch.randperm(num_kv, generator=generator)
        x[i, pos], x[i, pos + 1] = keys[i, order], vals[i, order]
        y[i, pos] = vals[i, order]
    return x, y


def make_selective_copy(n, ctx_len, num_tokens, data_vocab, generator):
    """Vocab: 0 noise, 1 separator, 2.. data. Returns x, y of length ctx_len + num_tokens."""
    data = torch.randint(2, 2 + data_vocab, (n, num_tokens), generator=generator)
    seq = torch.zeros(n, ctx_len + 1 + num_tokens, dtype=torch.long)
    for i in range(n):
        pos = torch.randperm(ctx_len, generator=generator)[:num_tokens].sort().values
        seq[i, pos] = data[i]
    seq[:, ctx_len] = 1
    seq[:, ctx_len + 1:] = data
    x, y = seq[:, :-1], seq[:, 1:].clone()
    y[:, :ctx_len] = IGNORE
    return x, y


def make_model(arch, d_model, vocab, seq_len, layers):
    if arch == "transformer":
        cfg = ModelConfig(arch="transformer", vocab_size=vocab, d_model=d_model, n_layers=layers,
                          n_heads=max(1, d_model // 64), max_seq_len=seq_len)
    else:  # one Mamba block replaces both attention and MLP, so use twice as many layers
        cfg = ModelConfig(arch="mamba", vocab_size=vocab, d_model=d_model, n_layers=2 * layers,
                          max_seq_len=seq_len, selective=arch == "mamba")
    return build_model(cfg)


@torch.no_grad()
def accuracy(model, x, y, device, batch=128):
    model.eval()
    correct = total = 0
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
        pred = model(xb)[0].argmax(-1)
        mask = yb != IGNORE
        correct += (pred[mask] == yb[mask]).sum().item()
        total += mask.sum().item()
    model.train()
    return correct / total


def train_task(model, sample, device, steps, batch, lr, eval_x, eval_y, log_every=500, on_eval=None):
    """Returns (best eval accuracy, last step). on_eval(step, loss, accuracy) is called at every evaluation."""
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    best = 0.0
    for step in range(1, steps + 1):
        x, y = sample(batch)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            acc = accuracy(model, eval_x, eval_y, device)
            best = max(best, acc)
            if on_eval is not None:
                on_eval(step, loss.item(), acc)
            if best > 0.995:
                break
    return best, step


CURVE_FIELDS = ["task", "arch", "d_model", "seq_len", "num_kv_or_tokens", "lr", "seed", "step", "loss", "accuracy"]


def _curve_writer(path, key):
    """Appends one row per evaluation. Rows left by an interrupted earlier attempt at the same grid
    point are dropped first, so a resumed run does not duplicate its curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    same = lambda r: all(str(r[f]) == str(v) for f, v in key.items())  # noqa: E731
    if path.exists():
        with open(path, newline="") as f:
            keep = [r for r in csv.DictReader(f) if not same(r)]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
            w.writeheader()
            w.writerows(keep)

    def write(step, loss, acc):
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
            if new:
                w.writeheader()
            w.writerow({**key, "step": step, "loss": round(loss, 5), "accuracy": acc})
    return write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mqar", "selective_copy"], required=True)
    ap.add_argument("--archs", nargs="+", default=["transformer", "mamba", "mamba_lti"])
    ap.add_argument("--d-models", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--kv-ratio", type=float, default=0.25, help="MQAR: key-value pairs = ratio * seq_len / 2")
    ap.add_argument("--num-tokens", type=int, default=16, help="selective copy: tokens to copy")
    ap.add_argument("--vocab", type=int, default=8192, help="MQAR vocabulary")
    ap.add_argument("--layers", type=int, default=2, help="Transformer layers (Mamba gets 2x)")
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-3])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--curve", default=None, help="optional CSV: eval accuracy at every evaluation step")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # resume: skip finished grid points
        with open(out) as f:
            done = {(r["arch"], int(r["d_model"]), int(r["seq_len"]), float(r["lr"])) for r in csv.DictReader(f)}
    fields = ["task", "arch", "d_model", "seq_len", "num_kv_or_tokens", "lr", "accuracy", "steps",
              "params", "seconds", "seed"]
    write_header = not out.exists()
    for seq_len, d_model, arch, lr in itertools.product(a.seq_lens, a.d_models, a.archs, a.lrs):
        if (arch, d_model, seq_len, lr) in done:
            continue
        g = torch.Generator().manual_seed(a.seed)
        eg = torch.Generator().manual_seed(a.seed + 10_000)
        if a.task == "mqar":
            k = max(1, int(a.kv_ratio * seq_len / 2))
            vocab = a.vocab
            sample = lambda n, k=k, L=seq_len: make_mqar(n, L, k, vocab, g)  # noqa: E731
            ex, ey = make_mqar(a.eval_n, seq_len, k, vocab, eg)
        else:
            k = a.num_tokens
            vocab = 2 + 16
            ctx = seq_len - k
            sample = lambda n, k=k, c=ctx: make_selective_copy(n, c, k, 16, g)  # noqa: E731
            ex, ey = make_selective_copy(a.eval_n, ctx, k, 16, eg)
        torch.manual_seed(a.seed)
        model = make_model(arch, d_model, vocab, seq_len, a.layers).to(device)
        on_eval = None
        if a.curve:
            key = {"task": a.task, "arch": arch, "d_model": d_model, "seq_len": seq_len, "num_kv_or_tokens": k,
                   "lr": lr, "seed": a.seed}
            on_eval = _curve_writer(Path(a.curve), key)
        t0 = time.time()
        acc, steps = train_task(model, sample, device, a.steps, a.batch, lr, ex, ey, a.eval_every, on_eval)
        row = {"task": a.task, "arch": arch, "d_model": d_model, "seq_len": seq_len, "num_kv_or_tokens": k,
               "lr": lr, "accuracy": acc, "steps": steps, "params": sum(p.numel() for p in model.parameters()),
               "seconds": round(time.time() - t0, 1), "seed": a.seed}
        print(row, flush=True)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)


if __name__ == "__main__":
    main()
