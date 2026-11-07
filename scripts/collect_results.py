"""Copy the small evidence files of finished runs (no weights) into results/<scale>/runs/, and the
data build's metadata (token counts, duplicates, tokenizer hash, PG-19 books) into results/data/.

    python scripts/collect_results.py --runs RUNS_DIR --results results --data DATA
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default=None, help="data build folder; its meta files go to results/data/")
    a = ap.parse_args()
    if a.data:
        dest = Path(a.results) / "data"
        for name in ("meta.json", "pg19_meta.json"):
            if (Path(a.data) / name).exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(a.data) / name, dest / name)
                print(dest / name)
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
