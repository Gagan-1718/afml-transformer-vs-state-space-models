"""Inference efficiency: prefill latency, decode latency, peak memory and throughput vs length.

Uses randomly initialised models of the study's sizes (speed does not depend on the weights).
Every Mamba measurement is repeated per scan backend so implementation effects are visible.
Protocol: one GPU, fixed precision, warm-up runs, median of repeated CUDA-event timings.
The CSV is rewritten after every measurement, so a stopped session keeps what it measured.
Once a prefill takes longer than --max-prefill-ms, longer lengths at that batch size are recorded
as "skipped_slow" for that model and backend (this bounds the slow pure-PyTorch scan).

    python -m afml.eval.bench --configs configs/S_transformer.json configs/S_mamba.json \
        --lengths 512 1024 2048 4096 8192 16384 --out results/bench/bench.csv
"""
import argparse
import csv
import json
import os
import statistics
from pathlib import Path

import torch

from ..config import ModelConfig
from ..models import build_model, count_params
from ..models.mamba import HAS_CUDA_SCAN
from .common import Timer, device_auto

DEFAULT_BACKENDS = ["cuda", "chunked"]  # the official kernel first, so the slow chunked scan cannot crowd it out
FIELDS = ["config", "arch", "backend", "params_total", "device", "precision", "seq_len", "batch", "status",
          "prefill_ms", "prefill_peak_mb", "prefill_tokens_per_s", "decode_ms_per_token", "decode_tokens_per_s",
          "decode_peak_mb", "cache_mb_analytic", "weights_mb"]


def write_rows(path, rows):
    # Write to a temporary file, then rename, so the CSV is never half-written.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _dtype(name):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _peak_reset(device):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak(device):
    return torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")


def _median_ms(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        with Timer(device) as t:
            fn()
        times.append(t.ms)
    return statistics.median(times)


@torch.no_grad()
def bench_model(model, label, lengths, batches, device, dtype, warmup, repeats, decode_tokens,
                max_prefill_ms=float("inf"), on_row=lambda row: None):
    rows = []
    vocab = model.cfg.vocab_size
    weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    too_slow = set()  # batch sizes whose prefill already took longer than max_prefill_ms
    for L in lengths:
        for b in batches:
            row = {**label, "seq_len": L, "batch": b}
            if b in too_slow:
                row["status"] = "skipped_slow"
                print(row, flush=True)
                rows.append(row)
                on_row(row)
                continue
            idx = torch.randint(0, vocab, (b, L), device=device)
            try:
                _peak_reset(device)
                row["prefill_ms"] = _median_ms(lambda: model(idx), device, warmup, repeats)
                row["prefill_peak_mb"] = _peak(device)
                row["prefill_tokens_per_s"] = b * L / (row["prefill_ms"] / 1000)

                _peak_reset(device)
                caches = model.init_cache(b, L + decode_tokens + warmup + 1, device, dtype)
                logits = model.prefill(idx, caches)
                tok = logits.argmax(-1)
                state = {"tok": tok}

                def one_step():
                    state["tok"] = model.decode_step(state["tok"], caches).argmax(-1)

                for _ in range(warmup):
                    one_step()
                with Timer(device) as t:
                    for _ in range(decode_tokens):
                        one_step()
                row["decode_ms_per_token"] = t.ms / decode_tokens
                row["decode_tokens_per_s"] = b * decode_tokens / (t.ms / 1000)
                row["decode_peak_mb"] = _peak(device)
                row["cache_mb_analytic"] = model.cache_bytes(b, L, torch.tensor([], dtype=dtype).element_size()) / 2**20
                row["weights_mb"] = weights_mb
                row["status"] = "ok"
                del caches
            except torch.cuda.OutOfMemoryError:
                row["status"] = "oom"
                torch.cuda.empty_cache()
            print(row, flush=True)
            rows.append(row)
            on_row(row)
            if row["status"] == "ok" and row["prefill_ms"] > max_prefill_ms:
                too_slow.add(b)
            if row["status"] == "oom":
                break  # larger batches will not fit either
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="run configs (model section is used)")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                    help="Mamba scan backends to time, in this order (cuda is skipped if mamba-ssm is missing)")
    ap.add_argument("--max-prefill-ms", type=float, default=20000.0,
                    help="after a prefill slower than this, skip longer lengths at that batch size")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = device_auto(a.device)
    dtype = _dtype(a.precision) if device == "cuda" else torch.float32
    rows = []
    for path in a.configs:
        mcfg = json.loads(Path(path).read_text())["model"]
        backends = a.backends if mcfg["arch"] == "mamba" else ["-"]
        for backend in backends:
            if backend == "cuda" and not (HAS_CUDA_SCAN and device == "cuda"):
                print("skipping cuda backend: mamba-ssm not available", flush=True)
                continue
            over = {"scan": backend} if mcfg["arch"] == "mamba" else {}
            cfg = ModelConfig.from_dict({**mcfg, **over, "vocab_size": a.vocab, "max_seq_len": 1024})
            torch.manual_seed(0)
            model = build_model(cfg).to(device=device, dtype=dtype).eval()
            label = {"config": Path(path).stem, "arch": cfg.arch, "backend": backend,
                     "params_total": count_params(model), "device":
                     torch.cuda.get_device_name() if device == "cuda" else "cpu", "precision": str(dtype)}
            done = list(rows)  # rows from earlier models; saved together with each new row

            def save(row, done=done):
                done.append(row)
                write_rows(a.out, done)

            rows += bench_model(model, label, a.lengths, a.batches, device, dtype, a.warmup, a.repeats,
                                a.decode_tokens, a.max_prefill_ms, on_row=save)
            del model
    write_rows(a.out, rows)


if __name__ == "__main__":
    main()
