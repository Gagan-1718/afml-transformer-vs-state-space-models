"""Learning-rate sweep at a reduced token budget; picks the best LR per architecture.

    python scripts/sweep_lr.py --scale S --data DATA --runs RUNS_DIR [--lrs 3e-4 6e-4 1.2e-3]

Writes results/<scale>/lr_sweep.csv and results/<scale>/best_lr.json. Resumable: finished runs are skipped.
Runs live in RUNS_DIR/sweep/<scale>_<arch>_lr<lr>_w<warmup steps>. The notebook trains them in parallel (one per GPU) with
sweep_jobs() and then calls this script, which only collects the finished runs.

Every sweep run warms up for as many steps as the full run (51 at S), not 2% of its own short length:
with 2% (4 steps) high learning rates blew up and the sweep favoured the lowest LR. A run whose loss
becomes NaN/inf writes diverged.json; it is recorded as "diverged" (that LR is too high) and skipped
when picking the best LR, instead of stopping the sweep.

Tie rule: learning rates whose validation loss is within TIE_TOLERANCE of the best count as tied, and the
lowest tied LR is used. The full run is about 11x longer than a sweep run, longer runs favour lower LRs,
and a lower LR is less likely to blow up in fp16. (Added after sweep v2, where Mamba at 1.2e-3 and 2.4e-3
differed by 0.0008.) The notebook recomputes best_lr.json from lr_sweep.csv with best_from_csv().
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afml.train import DIVERGED_EXIT_CODE, PAUSED_EXIT_CODE, load_config, planned_steps, warmup_steps  # noqa: E402

# The Transformer also gets 1.5e-4: in the first sweep (4 warmup steps) its best LR was the lowest tried.
DEFAULT_LRS = {"transformer": [1.5e-4, 3e-4, 6e-4, 1.2e-3, 2.4e-3], "mamba": [3e-4, 6e-4, 1.2e-3, 2.4e-3]}
DEFAULT_TOKENS = 30_000_000  # ~228 steps at S; ~0.3-0.5 h (Transformer) or ~0.75 h (Mamba) per run on a T4
TIE_TOLERANCE = 0.01  # validation-loss differences smaller than this are treated as noise


def lrs_for(lrs, arch):
    """lrs is one list for every architecture, or a dict {arch: list}."""
    return list(lrs[arch] if isinstance(lrs, dict) else lrs)


def main_run_warmup(scale, arch):
    """Warmup steps of the full run that this sweep is choosing the LR for."""
    tc = load_config(ROOT / "configs" / f"{scale}_{arch}.json")["train"]
    return warmup_steps(planned_steps(tc), tc)


def sweep_jobs(scale, archs, lrs, tokens, data, runs, stop_at=None):
    """[(arch, lr, run_dir, train command)] for every sweep run; the same names main() looks for.
    The warmup is part of the name, so runs from the first sweep (4 warmup steps) are never reused."""
    jobs = []
    for arch in archs:
        warm = main_run_warmup(scale, arch)
        for lr in lrs_for(lrs, arch):
            out = Path(runs) / "sweep" / f"{scale}_{arch}_lr{lr:g}_w{warm}"
            cmd = [sys.executable, "-m", "afml.train", "--config", str(ROOT / "configs" / f"{scale}_{arch}.json"),
                   "--data", str(data), "--out", str(out), "--set", f"train.lr={lr}",
                   "--set", f"train.total_tokens={tokens}", "--set", f"train.warmup_steps={warm}"]
            if stop_at is not None:
                cmd += ["--stop-at", str(stop_at)]
            jobs.append((arch, lr, out, cmd))
    return jobs


def finished(out):
    return (Path(out) / "summary.json").exists() or (Path(out) / "diverged.json").exists()


def choose_lrs(rows):
    """({arch: chosen LR}, [notes]) from sweep rows (dicts as in lr_sweep.csv; values may be strings)."""
    best, notes = {}, []
    for arch in dict.fromkeys(r["arch"] for r in rows):
        mine = [r for r in rows if r["arch"] == arch]
        ok = [r for r in mine if r.get("status", "ok") == "ok"]
        if not ok:
            raise SystemExit(f"every {arch} sweep run diverged: rerun with lower --lrs")
        low = min(float(r["best_val_loss"]) for r in ok)
        tied = sorted(float(r["lr"]) for r in ok if float(r["best_val_loss"]) <= low + TIE_TOLERANCE)
        best[arch] = tied[0]
        grid = [float(r["lr"]) for r in mine]
        diverged = [float(r["lr"]) for r in mine if r.get("status", "ok") != "ok"]
        if diverged:
            notes.append(f"{arch}: diverged (LR too high) at {diverged}")
        if len(tied) > 1:
            notes.append(f"{arch}: LRs {tied} are within {TIE_TOLERANCE} of the best validation loss "
                         f"(tied); using the lowest, {best[arch]:g}")
        if best[arch] in (min(grid), max(grid)):  # an edge winner means the sweep range was too narrow
            notes.append(f"WARNING: chosen LR for {arch} is at the edge of the sweep; extend --lrs")
    if "mamba" in best:
        best.setdefault("mamba_lti", best["mamba"])
    return best, notes


def best_from_csv(path):
    """Chosen LRs from a finished sweep's lr_sweep.csv (prints the notes). Used by the notebook."""
    with open(path, newline="") as f:
        best, notes = choose_lrs(list(csv.DictReader(f)))
    for n in notes:
        print(n)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="S")
    ap.add_argument("--archs", nargs="+", default=["transformer", "mamba"])
    ap.add_argument("--lrs", nargs="+", type=float, default=None,
                    help="one LR list for every arch (default: DEFAULT_LRS, per arch)")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="token budget per sweep run")
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--stop-at", type=float, default=None, help="unix time; passed on to afml.train")
    a = ap.parse_args()
    lrs = a.lrs if a.lrs is not None else DEFAULT_LRS
    rows = []
    for arch, lr, out, cmd in sweep_jobs(a.scale, a.archs, lrs, a.tokens, a.data, a.runs, a.stop_at):
        if not finished(out):
            code = subprocess.run(cmd, cwd=ROOT).returncode
            if code == PAUSED_EXIT_CODE:
                print("time limit reached during the sweep; rerun to continue", flush=True)
                sys.exit(PAUSED_EXIT_CODE)
            if code not in (0, DIVERGED_EXIT_CODE):
                raise subprocess.CalledProcessError(code, cmd)
        warm = int(next(c for c in cmd if c.startswith("train.warmup_steps=")).split("=")[1])
        row = {"scale": a.scale, "arch": arch, "lr": lr, "warmup_steps": warm, "status": "ok",
               "diverged_at_step": "", "tokens": "", "best_val_loss": "", "train_seconds": ""}
        if (out / "summary.json").exists():
            s = json.loads((out / "summary.json").read_text())
            row.update(tokens=s["tokens"], best_val_loss=s["best_val_loss"], train_seconds=s["train_seconds"])
        else:
            row.update(status="diverged", diverged_at_step=json.loads((out / "diverged.json").read_text())["step"])
        rows.append(row)
        print(row, flush=True)
    res = Path(a.results) / a.scale
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "lr_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best, notes = choose_lrs(rows)
    for n in notes:
        print(n)
    (res / "best_lr.json").write_text(json.dumps(best, indent=2))
    print("best learning rates:", best)


if __name__ == "__main__":
    main()
