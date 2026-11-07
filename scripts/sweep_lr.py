"""Learning-rate sweep at a reduced token budget; picks the best LR per architecture.

    python scripts/sweep_lr.py --scale S --data DATA --runs RUNS_DIR --lrs 3e-4 6e-4 1.2e-3 2.4e-3

Writes results/<scale>/lr_sweep.csv and results/<scale>/best_lr.json. Resumable: finished runs are skipped.
Runs live in RUNS_DIR/sweep/<scale>_<arch>_lr<lr>. The notebook trains them in parallel (one per GPU) with
sweep_jobs() and then calls this script, which only collects the finished runs.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LRS = [3e-4, 6e-4, 1.2e-3, 2.4e-3]
DEFAULT_TOKENS = 100_000_000
PAUSED_EXIT_CODE = 75  # same as afml.train: stopped at --stop-at, rerun to resume


def sweep_jobs(scale, archs, lrs, tokens, data, runs, stop_at=None):
    """[(arch, lr, run_dir, train command)] for every sweep run; the same names main() looks for."""
    jobs = []
    for arch in archs:
        for lr in lrs:
            out = Path(runs) / "sweep" / f"{scale}_{arch}_lr{lr:g}"
            cmd = [sys.executable, "-m", "afml.train", "--config", str(ROOT / "configs" / f"{scale}_{arch}.json"),
                   "--data", str(data), "--out", str(out), "--set", f"train.lr={lr}",
                   "--set", f"train.total_tokens={tokens}"]
            if stop_at is not None:
                cmd += ["--stop-at", str(stop_at)]
            jobs.append((arch, lr, out, cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="S")
    ap.add_argument("--archs", nargs="+", default=["transformer", "mamba"])
    ap.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="token budget per sweep run")
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--stop-at", type=float, default=None, help="unix time; passed on to afml.train")
    a = ap.parse_args()
    rows = []
    for arch, lr, out, cmd in sweep_jobs(a.scale, a.archs, a.lrs, a.tokens, a.data, a.runs, a.stop_at):
        if not (out / "summary.json").exists():
            code = subprocess.run(cmd, cwd=ROOT).returncode
            if code == PAUSED_EXIT_CODE:
                print("time limit reached during the sweep; rerun to continue", flush=True)
                sys.exit(PAUSED_EXIT_CODE)
            if code != 0:
                raise subprocess.CalledProcessError(code, cmd)
        s = json.loads((out / "summary.json").read_text())
        rows.append({"scale": a.scale, "arch": arch, "lr": lr, "tokens": s["tokens"],
                     "best_val_loss": s["best_val_loss"], "train_seconds": s["train_seconds"]})
        print(rows[-1], flush=True)
    res = Path(a.results) / a.scale
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "lr_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = {arch: min((r for r in rows if r["arch"] == arch), key=lambda r: r["best_val_loss"])["lr"]
            for arch in a.archs}
    for arch in a.archs:  # an edge winner means the sweep range was too narrow
        if best[arch] in (min(a.lrs), max(a.lrs)):
            print(f"WARNING: best LR for {arch} is at the edge of the sweep; extend --lrs")
    if "mamba" in best:
        best.setdefault("mamba_lti", best["mamba"])
    (res / "best_lr.json").write_text(json.dumps(best, indent=2))
    print("best learning rates:", best)


if __name__ == "__main__":
    main()
