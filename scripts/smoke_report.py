"""Turn short GPU smoke runs into measured speed/memory and rough time estimates for the full study.

    python scripts/smoke_report.py --runs RUNS/smoke_* --gpus 2 --session-hours 11 --out RESULTS/smoke

Measured per smoke run: steady-state training tokens/sec (from the logged rows, so start-up cost is
excluded), the micro-batch the trainer settled on, and peak GPU memory (torch.cuda.max_memory_allocated).

Estimated per full run: hours = training tokens / tokens-per-sec + evaluation time. Runs that were not
measured directly (Mamba-LTI, scale M) are scaled from a measured run of the same family by training
FLOPs per token. That scaling is rough: the pure-PyTorch chunked scan is not FLOP-bound. These are
planning numbers for the GPU budget, never results for the report.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afml.config import ModelConfig  # noqa: E402
from afml.models import build_model, count_params, train_flops_per_token  # noqa: E402
from afml.train import DEFAULT_TRAIN  # noqa: E402

TARGETS = ["S_transformer", "S_mamba", "S_mamba_lti", "M_transformer", "M_mamba", "M_mamba_lti"]
SWEEP_LRS = {"transformer": 5, "mamba": 4}  # number of LRs in scripts/sweep_lr.py DEFAULT_LRS
SWEEP_TOKENS = 30_000_000


def label(summary):
    if summary["arch"] == "transformer":
        return "transformer"
    return "mamba" if summary["selective"] else "mamba_lti"


def measure(run_dir):
    """Measured numbers from one finished smoke run."""
    run = Path(run_dir)
    s = json.loads((run / "summary.json").read_text())
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r.get("train_loss")]
    tps = s["tokens"] / max(s["train_seconds"], 1e-9)  # fallback: includes the slow first steps
    if len(train_rows) >= 2:  # steady state: skip everything before the first logged row
        a, b = train_rows[0], train_rows[-1]
        dt = float(b["elapsed_s"]) - float(a["elapsed_s"])
        if dt > 0:
            tps = (float(b["tokens"]) - float(a["tokens"])) / dt
    tc = s["train_config"]
    n_evals = sum(1 for r in rows if r.get("val_loss"))
    return {
        "run": run.name, "model": label(s), "scan_backend": s.get("scan_backend"),
        "device": s["device"], "precision": s["precision"],
        "tokens_per_sec": tps, "micro_batch": tc["micro_batch"], "grad_accum": tc["grad_accum"],
        "peak_memory_gb": s.get("peak_memory_gb"),
        "flops_per_token": s["train_flops"] / max(1, s["tokens"]),
        "eval_s_per_token": s.get("eval_seconds", 0.0) / max(1, n_evals * tc["eval_tokens"]),
        "vocab_size": s["model_config"]["vocab_size"],
    }


def target_cost(name, vocab_size):
    """What a full run of configs/<name>.json costs in tokens, evaluations and FLOPs per token."""
    cfg = json.loads((ROOT / "configs" / f"{name}.json").read_text())
    tc = {**DEFAULT_TRAIN, **cfg["train"]}
    mcfg = ModelConfig.from_dict({**cfg["model"], "vocab_size": vocab_size, "max_seq_len": tc["seq_len"]})
    n_nonemb = count_params(build_model(mcfg), exclude_embedding=True)
    tokens_per_step = tc["micro_batch"] * tc["grad_accum"] * tc["seq_len"]
    return {"family": "transformer" if mcfg.arch == "transformer" else "mamba",
            "flops_per_token": train_flops_per_token(mcfg, n_nonemb, tc["seq_len"]),
            "total_tokens": tc["total_tokens"], "tokens_per_step": tokens_per_step,
            "eval_tokens": tc["eval_tokens"], "eval_every": tc["eval_every"]}


def run_hours(tps, eval_s_per_eval, tokens, tokens_per_step, eval_every):
    """Training plus evaluation hours (evaluations at step 0, every eval_every, and at the end)."""
    n_evals = max(1, tokens // tokens_per_step) // eval_every + 2
    return tokens / tps / 3600, n_evals * eval_s_per_eval / 3600


def estimate(measured, targets):
    """One estimate per (target config, measured run of the same family)."""
    out = []
    for name, t in targets.items():
        for m in measured:
            if ("transformer" if m["model"] == "transformer" else "mamba") != t["family"]:
                continue
            ratio = t["flops_per_token"] / m["flops_per_token"]
            tps = m["tokens_per_sec"] / ratio
            eval_s = t["eval_tokens"] * m["eval_s_per_token"] * ratio
            train_h, eval_h = run_hours(tps, eval_s, t["total_tokens"], t["tokens_per_step"], t["eval_every"])
            sweep_train_h, sweep_eval_h = run_hours(tps, eval_s, SWEEP_TOKENS, t["tokens_per_step"], t["eval_every"])
            exact = name.split("_", 1)[1] == m["model"] and math.isclose(ratio, 1.0, rel_tol=1e-3)
            out.append({"target": name, "scan_backend": m["scan_backend"], "from_run": m["run"],
                        "how": "measured" if exact else "scaled by FLOPs/token",
                        "tokens_per_sec": tps, "train_hours": train_h, "eval_hours": eval_h,
                        "hours": train_h + eval_h, "sweep_run_hours": sweep_train_h + sweep_eval_h})
    return out


def stage_plan(est, gpus, session_hours):
    """GPU-hours, wall-clock hours and Kaggle sessions per stage, for each Mamba scan backend measured."""
    backends = sorted({e["scan_backend"] for e in est if e["scan_backend"]}) or [None]
    plans = []
    for backend in backends:
        def pick(target, key):
            for e in est:
                if e["target"] == target and e["scan_backend"] in (None, backend):
                    return e[key]
            return None

        stages = {
            "sweep (S, Transformer x {transformer} + Mamba x {mamba} LRs)".format(**SWEEP_LRS):
                [pick(f"S_{a}", "sweep_run_hours") for a, n in SWEEP_LRS.items() for _ in range(n)],
            "S (3 models x 3 seeds)": [pick(f"S_{a}", "hours") for a in ("transformer", "mamba", "mamba_lti")
                                       for _ in range(3)],
            "M (transformer + mamba, 1 seed)": [pick(f"M_{a}", "hours") for a in ("transformer", "mamba")],
        }
        for stage, runs in stages.items():
            if any(h is None for h in runs):
                continue
            total = sum(runs)
            wall = max(total / max(1, gpus), max(runs))  # one run cannot be split across GPUs
            plans.append({"stage": stage, "mamba_backend": backend, "runs": len(runs), "gpu_hours": total,
                          "longest_run_hours": max(runs), "wall_hours": wall,
                          "kaggle_sessions": math.ceil(wall / session_hours)})
    return plans


def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("-" if v is None else (fmt[c].format(v) if c in fmt else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="finished smoke run folders")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs running jobs in parallel per session")
    ap.add_argument("--session-hours", type=float, default=11.0, help="usable training hours per session")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    measured = [measure(r) for r in a.runs if (Path(r) / "summary.json").exists()]
    if not measured:
        raise SystemExit("no finished smoke runs found")
    targets = {n: target_cost(n, measured[0]["vocab_size"]) for n in TARGETS
               if (ROOT / "configs" / f"{n}.json").exists()}
    est = estimate(measured, targets)
    plan = stage_plan(est, a.gpus, a.session_hours)

    text = "\n\n".join([
        "## Measured in the smoke runs (steady state)",
        md_table(measured, ["run", "scan_backend", "device", "precision", "tokens_per_sec", "micro_batch",
                            "grad_accum", "peak_memory_gb"],
                 {"tokens_per_sec": "{:,.0f}", "peak_memory_gb": "{:.2f}"}),
        "## Estimated hours per full run (planning only, not results)",
        md_table(est, ["target", "scan_backend", "how", "tokens_per_sec", "train_hours", "eval_hours", "hours"],
                 {"tokens_per_sec": "{:,.0f}", "train_hours": "{:.1f}", "eval_hours": "{:.2f}",
                  "hours": "{:.1f}"}),
        f"## Estimated stage budget ({a.gpus} GPU(s) in parallel, {a.session_hours:g} usable hours per session)",
        md_table(plan, ["stage", "mamba_backend", "runs", "gpu_hours", "longest_run_hours", "wall_hours",
                        "kaggle_sessions"],
                 {"gpu_hours": "{:.1f}", "longest_run_hours": "{:.1f}", "wall_hours": "{:.1f}"}),
    ])
    print(text)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_report.md").write_text(text + "\n")
    (out / "smoke_report.json").write_text(json.dumps({"measured": measured, "estimates": est, "stages": plan,
                                                       "gpus": a.gpus, "session_hours": a.session_hours},
                                                      indent=2))


if __name__ == "__main__":
    main()
