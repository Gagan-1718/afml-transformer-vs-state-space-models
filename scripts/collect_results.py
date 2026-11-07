"""Copy the small evidence files of finished runs (no weights) into results/<scale>/runs/.

    python scripts/collect_results.py --runs RUNS_DIR --results results
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    for run in sorted(Path(a.runs).glob("[SML]_*")):
        if not (run / "summary.json").exists():
            continue
        dest = Path(a.results) / run.name.split("_")[0] / "runs" / run.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "metrics.csv", "config.json"):
            if (run / name).exists():
                shutil.copy2(run / name, dest / name)
        print(dest)


if __name__ == "__main__":
    main()
