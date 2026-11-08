"""Train one language model from a JSON run config. Resumable: rerun the same command after a disconnect.

    python -m afml.train --config configs/S_mamba.json --data /path/to/data --out runs/S_mamba_s0
    python -m afml.train --config configs/S_mamba.json ... --set train.lr=1.2e-3 --set train.seed=1

--stop-at <unix time> pauses cleanly (saves last.pt, exits with code 75) so a run never hits a hard
session limit such as Kaggle's 12 hours; rerunning the same command resumes it.
"""
import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig
from .data import TokenData
from .models import build_model, count_params, param_groups, train_flops_per_token
from .models.mamba import HAS_CUDA_SCAN

PAUSED_EXIT_CODE = 75  # "try again later": the run stopped at --stop-at and resumes on the next call
DIVERGED_EXIT_CODE = 3  # the loss became NaN/inf; diverged.json says when (in a sweep: this LR is too high)

# warmup_steps, when set, overrides warmup_frac. The LR sweep sets it so its short runs warm up for as many
# steps as the full run (2% of a 228-step sweep run is only 4 steps, which unfairly punishes high LRs).
DEFAULT_TRAIN = {
    "seed": 0, "seq_len": 1024, "micro_batch": 16, "grad_accum": 16, "total_tokens": 340_000_000,
    "lr": 6e-4, "min_lr_ratio": 0.1, "warmup_frac": 0.02, "warmup_steps": None, "weight_decay": 0.1,
    "betas": [0.9, 0.95], "grad_clip": 1.0, "precision": "auto", "compile": False,
    "eval_every": 200, "eval_tokens": 2_000_000, "save_every_minutes": 20, "log_every": 20,
    "wandb_project": None,
}


def load_config(path, overrides=()):
    cfg = json.loads(Path(path).read_text())
    cfg["train"] = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    for item in overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        cfg[section][name] = value
    return cfg


class Diverged(RuntimeError):
    """The training loss became NaN or inf."""


def planned_steps(tc):
    """Optimizer steps of a full run (tokens per step stay fixed, even when micro_batch is halved)."""
    return max(1, tc["total_tokens"] // (tc["micro_batch"] * tc["grad_accum"] * tc["seq_len"]))


def warmup_steps(total, cfg):
    if cfg.get("warmup_steps"):
        return max(1, min(int(cfg["warmup_steps"]), total))
    return max(1, int(cfg["warmup_frac"] * total))


def lr_at(step, total, cfg):
    warm = warmup_steps(total, cfg)
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total - warm))
    floor = cfg["lr"] * cfg["min_lr_ratio"]
    return floor + 0.5 * (cfg["lr"] - floor) * (1 + math.cos(math.pi * progress))


def pick_precision(name, device):
    if device != "cuda" or name == "fp32":
        return torch.float32
    # bf16 only where it is native (Ampere, compute capability 8.0+). Recent torch reports bf16 as
    # "supported" on T4/P100 through slow emulation, so "auto" checks the hardware directly.
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8
    if name == "bf16" or (name == "auto" and native_bf16):
        return torch.bfloat16
    return torch.float16


def scan_backend(mcfg, device):
    """The Mamba scan that actually runs ("auto" resolved), so speed numbers can name it."""
    if mcfg.arch != "mamba":
        return None
    if mcfg.scan != "auto":
        return mcfg.scan
    return "cuda" if (HAS_CUDA_SCAN and device == "cuda") else "chunked"


@torch.no_grad()
def evaluate(model, x, y, device, dtype, batch=8):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            _, loss = model(xb, yb)
        total += loss.item() * yb.numel()
        n += yb.numel()
    model.train()
    return total / n


def _fits(fwd, model, data, tc, device, dtype):
    x, y = data.random_batch("train", tc["micro_batch"], tc["seq_len"], torch.Generator().manual_seed(0))
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            _, loss = fwd(x.to(device), y.to(device))
        loss.backward()
        # keep ~15% headroom for optimizer state and allocator fragmentation
        ok = torch.cuda.max_memory_allocated() < 0.85 * torch.cuda.get_device_properties(0).total_memory
    except torch.cuda.OutOfMemoryError:
        ok = False
    model.zero_grad(set_to_none=True)
    loss = None
    torch.cuda.empty_cache()
    return ok


def _rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def train(cfg, data_dir, out_dir, device="auto", max_steps=None, stop_at=None):
    """Train (or resume) one run. Returns the summary dict, or {"paused": True, ...} if stop_at
    (a unix time) was reached first; then last.pt is saved and no summary.json is written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    tc = cfg["train"]
    data = TokenData(data_dir)
    mcfg = ModelConfig.from_dict({**cfg["model"], "vocab_size": data.meta["vocab_size"],
                                  "max_seq_len": tc["seq_len"]})
    (out / "config.json").write_text(json.dumps({"model": asdict(mcfg), "train": tc,
                                                 "data_dir": str(data_dir)}, indent=2))
    torch.manual_seed(tc["seed"])
    model = build_model(mcfg).to(device)
    n_total, n_nonemb = count_params(model), count_params(model, exclude_embedding=True)
    (out / "diverged.json").unlink(missing_ok=True)  # a new attempt
    tokens_per_step = tc["micro_batch"] * tc["grad_accum"] * tc["seq_len"]
    total_steps = planned_steps(tc)
    if max_steps:
        total_steps = min(total_steps, max_steps)
    dtype = pick_precision(tc["precision"], device)
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"],
                            betas=tuple(tc["betas"]), fused=device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    gen = torch.Generator().manual_seed(tc["seed"])
    step, best, elapsed, eval_seconds, history = 0, float("inf"), 0.0, 0.0, []

    ckpt = out / "last.pt"
    if ckpt.exists():  # resume
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        gen.set_state(state["gen"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        step, best, elapsed, history = state["step"], state["best"], state["elapsed"], state["history"]
        eval_seconds = state.get("eval_seconds", 0.0)
        print(f"resumed at step {step}/{total_steps}", flush=True)

    fwd = torch.compile(model) if tc["compile"] else model
    if device == "cuda":  # shrink the micro-batch until a step fits; tokens per step stay the same
        while tc["micro_batch"] > 1 and not _fits(fwd, model, data, tc, device, dtype):
            tc["micro_batch"] //= 2
            tc["grad_accum"] *= 2
            print(f"out of memory: micro_batch -> {tc['micro_batch']}, grad_accum -> {tc['grad_accum']}", flush=True)
    val_x, val_y = data.eval_windows("val", tc["seq_len"], tc["eval_tokens"])
    flops_tok = train_flops_per_token(mcfg, n_nonemb, tc["seq_len"])
    print(f"{mcfg.arch}: {n_total:,} params ({n_nonemb:,} non-embedding); {total_steps} steps x "
          f"{tokens_per_step:,} tokens; {dtype}; device {device}", flush=True)

    wb = None
    if tc.get("wandb_project"):
        import wandb
        wb = wandb.init(project=tc["wandb_project"], name=out.name, config=cfg, resume="allow", id=out.name)

    def save(path, full):
        state = {"model": model.state_dict(), "model_config": asdict(mcfg), "step": step}
        if full:
            state.update(opt=opt.state_dict(), scaler=scaler.state_dict(), gen=gen.get_state(),
                         rng=_rng_state(), best=best, elapsed=elapsed, eval_seconds=eval_seconds,
                         history=history)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def log_row(row):
        history.append(row)
        with (out / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in history for k in r}))
            w.writeheader()
            w.writerows(history)
        if wb:
            wb.log(row, step=row["step"])

    def timed_eval():
        nonlocal eval_seconds
        t0 = time.time()
        v = evaluate(model, val_x, val_y, device, dtype)
        eval_seconds += time.time() - t0
        return v

    model.train()
    if step == 0:
        v = timed_eval()
        log_row({"step": 0, "tokens": 0, "flops": 0.0, "val_loss": v, "val_ppl": math.exp(v)})
    last_save, run_loss, run_n, t_mark = time.time(), 0.0, 0, time.time()  # after eval: not training time
    while step < total_steps:
        lr = lr_at(step, total_steps, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(tc["grad_accum"]):
            x, y = data.random_batch("train", tc["micro_batch"], tc["seq_len"], gen)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                _, loss = fwd(x, y)
            if not torch.isfinite(loss):
                (out / "diverged.json").write_text(json.dumps(
                    {"step": step, "lr": lr, "loss": str(loss.item()), "total_steps": total_steps}))
                raise Diverged(f"non-finite loss at step {step} (lr {lr:.3g})")
            scaler.scale(loss / tc["grad_accum"]).backward()
            run_loss += loss.item()
            run_n += 1
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        now = time.time()
        elapsed += now - t_mark
        t_mark = now

        if step % tc["log_every"] == 0 or step == total_steps:
            row = {"step": step, "tokens": step * tokens_per_step, "flops": step * tokens_per_step * flops_tok,
                   "lr": lr, "train_loss": run_loss / run_n, "grad_norm": gnorm,
                   "tokens_per_sec": step * tokens_per_step / max(elapsed, 1e-9), "elapsed_s": elapsed}
            if step % tc["eval_every"] == 0 or step == total_steps:
                v = timed_eval()
                row.update(val_loss=v, val_ppl=math.exp(v))
                if v < best:
                    best = v
                    save(out / "best.pt", full=False)
                t_mark = time.time()  # evaluation time is not training time
            log_row(row)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
            run_loss, run_n = 0.0, 0
        if time.time() - last_save > 60 * tc["save_every_minutes"] or step == total_steps:
            save(ckpt, full=True)
            last_save = time.time()
        if stop_at is not None and time.time() >= stop_at and step < total_steps:
            save(ckpt, full=True)
            print(f"time limit reached: paused at step {step}/{total_steps}; rerun the same command to resume",
                  flush=True)
            if wb:
                wb.finish()
            return {"paused": True, "step": step, "total_steps": total_steps}

    summary = {
        "arch": mcfg.arch, "selective": mcfg.selective, "params_total": n_total, "params_nonembedding": n_nonemb,
        "model_config": asdict(mcfg), "train_config": tc, "steps": step, "tokens": step * tokens_per_step,
        "train_flops": step * tokens_per_step * flops_tok, "best_val_loss": best, "best_val_ppl": math.exp(best),
        "train_seconds": elapsed, "eval_seconds": eval_seconds,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
        "precision": str(dtype), "torch": torch.__version__,
        "scan": mcfg.scan if mcfg.arch == "mamba" else None,
        "scan_backend": scan_backend(mcfg, device),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if wb:
        wb.finish()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke tests)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=1e-3")
    ap.add_argument("--stop-at", type=float, default=None,
                    help="unix time: save last.pt and exit with code 75 once reached (resume by rerunning)")
    a = ap.parse_args()
    try:
        s = train(load_config(a.config, a.set), a.data, a.out, a.device, a.max_steps, a.stop_at)
    except Diverged as e:
        print(f"diverged: {e}. Details in diverged.json.", flush=True)
        sys.exit(DIVERGED_EXIT_CODE)
    if s.get("paused"):
        sys.exit(PAUSED_EXIT_CODE)
    print(json.dumps({k: s[k] for k in ("arch", "params_total", "best_val_loss", "best_val_ppl", "train_seconds")}))


if __name__ == "__main__":
    main()
