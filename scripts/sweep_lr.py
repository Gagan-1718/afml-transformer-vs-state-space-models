"""Learning-rate sweep at a reduced token budget; picks the best LR per architecture.

    python scripts/sweep_lr.py --scale S --data DATA --runs RUNS_DIR --lrs 3e-4 6e-4 1.2e-3 2.4e-3

Writes results/<scale>/lr_sweep.csv and results/<scale>/best_lr.json. Resumable: finished runs are skipped.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="S")
    ap.add_argument("--archs", nargs="+", default=["transformer", "mamba"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 6e-4, 1.2e-3, 2.4e-3])
    ap.add_argument("--tokens", type=int, default=100_000_000, help="token budget per sweep run")
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    rows = []
    for arch in a.archs:
        for lr in a.lrs:
            out = Path(a.runs) / "sweep" / f"{a.scale}_{arch}_lr{lr:g}"
            if not (out / "summary.json").exists():
                subprocess.run([sys.executable, "-m", "afml.train", "--config", str(ROOT / "configs" / f"{a.scale}_{arch}.json"),
                                "--data", a.data, "--out", str(out), "--set", f"train.lr={lr}",
                                "--set", f"train.total_tokens={a.tokens}"], check=True, cwd=ROOT)
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
