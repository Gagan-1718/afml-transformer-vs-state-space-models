"""Make every report figure and the main results table from the CSVs in a results folder.

    python scripts/make_figures.py --results results --out report/figures

Expected inputs (any missing one is skipped):
  results/{S,M,L}/runs/<run>/metrics.csv + summary.json   training curves
  results/{S,M,L}/perplexity.csv                           test loss, recall-hit split
  results/{S,M,L}/long_context.csv                         per-position PG-19 loss
  results/synthetic/{mqar,selective_copy}.csv              synthetic tasks
  results/bench/bench.csv                                  efficiency
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colour follows the model, never its rank (validated categorical slots 1-3).
COLORS = {"transformer": "#2a78d6", "mamba": "#eb6834", "mamba_lti": "#1baf7a"}
NAMES = {"transformer": "Transformer", "mamba": "Mamba (selective)", "mamba_lti": "Mamba-LTI (ablation)"}
BACKEND_STYLE = {"-": "-", "chunked": "--", "cuda": "-", "parallel": ":", "sequential": ":"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def arch_of(summary):
    return summary["arch"] if summary["arch"] == "transformer" else ("mamba" if summary["selective"] else "mamba_lti")


def mean_std(xs):
    m = sum(xs) / len(xs)
    s = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, s


def fig_training_curves(res, out):
    scales = [s for s in ("S", "M", "L") if (res / s / "runs").exists()]
    if not scales:
        return
    fig, axes = plt.subplots(len(scales), 2, figsize=(10, 3.4 * len(scales)), squeeze=False)
    for row, scale in enumerate(scales):
        curves = defaultdict(list)
        for run in sorted((res / scale / "runs").iterdir()):
            if not (run / "summary.json").exists():
                continue
            s = json.loads((run / "summary.json").read_text())
            pts = [(num(r["tokens"]), num(r["flops"]), num(r["val_loss"])) for r in read_csv(run / "metrics.csv")
                   if r.get("val_loss") and num(r["tokens"]) > 0]
            curves[arch_of(s)].append(pts)
        for col, (xi, xlabel) in enumerate(((0, "Training tokens"), (1, "Training FLOPs (approx.)"))):
            ax = axes[row][col]
            for arch, runs in curves.items():
                n = min(len(r) for r in runs)
                xs = [runs[0][i][xi] for i in range(n)]
                ys = [[r[i][2] for r in runs] for i in range(n)]
                m = [mean_std(v)[0] for v in ys]
                ax.plot(xs, m, color=COLORS[arch], label=f"{NAMES[arch]}: {m[-1]:.3f} (n={len(runs)})")
                if len(runs) > 1:
                    ax.fill_between(xs, [min(v) for v in ys], [max(v) for v in ys], color=COLORS[arch], alpha=0.15,
                                    lw=0)
            ax.set(xscale="log", xlabel=xlabel, ylabel="Validation loss (nats/token)", title=f"Scale {scale}")
            ax.legend(fontsize=8)
    fig.suptitle("Validation loss during training (legend: final loss; band = min-max over seeds)",
                 x=0.01, y=1.01, ha="left")
    fig.savefig(out / "fig1_training_curves.png")
    plt.close(fig)


def table_main(res, out):
    lines = ["| Scale | Model | Params (total / non-emb) | Train tokens | Train FLOPs | Seeds | Test loss | "
             "Test ppl | Recall-hit loss | Other-token loss |", "|" + " --- |" * 10]
    found = False
    for scale in ("S", "M", "L"):
        p = res / scale / "perplexity.csv"
        if not p.exists():
            continue
        found = True
        groups = defaultdict(list)
        for r in read_csv(p):
            groups[r["arch"]].append(r)
        for arch, rows in groups.items():
            loss_m, loss_s = mean_std([num(r["all_loss"]) for r in rows])
            hit_m, _ = mean_std([num(r["hit_loss"]) for r in rows])
            oth_m, _ = mean_std([num(r["other_loss"]) for r in rows])
            r0 = rows[0]
            pm = f"{num(r0['params_total']) / 1e6:.1f}M / {num(r0['params_nonembedding']) / 1e6:.1f}M"
            lines.append(f"| {scale} | {NAMES[arch]} | {pm} | {num(r0['train_tokens']) / 1e6:.0f}M | "
                         f"{num(r0['train_flops']):.2e} | {len(rows)} | {loss_m:.4f} ± {loss_s:.4f} | "
                         f"{math.exp(loss_m):.2f} | {hit_m:.3f} | {oth_m:.3f} |")
    if found:
        (out / "table_main.md").write_text("\n".join(lines) + "\n")


def fig_recall_split(res, out):
    data = []
    for scale in ("S", "M", "L"):
        p = res / scale / "perplexity.csv"
        if p.exists():
            g = defaultdict(list)
            for r in read_csv(p):
                g[r["arch"]].append((num(r["hit_loss"]), num(r["other_loss"])))
            for arch, v in g.items():
                data.append((scale, arch, mean_std([a for a, _ in v])[0], mean_std([b for _, b in v])[0]))
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    for ax, idx, title in ((axes[0], 2, "Recall-hit tokens (bigram seen earlier in context)"),
                           (axes[1], 3, "All other tokens")):
        labels = [f"{d[0]}\n{NAMES[d[1]].split(' ')[0]}{'-LTI' if d[1] == 'mamba_lti' else ''}" for d in data]
        bars = ax.bar(range(len(data)), [d[idx] for d in data], color=[COLORS[d[1]] for d in data],
                      edgecolor="white", linewidth=2)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK)
        ax.set_xticks(range(len(data)), labels, fontsize=7)
        ax.set(ylabel="Test loss (nats/token)", title=title)
        ax.grid(axis="x", visible=False)
    fig.savefig(out / "fig4_recall_split.png")
    plt.close(fig)


def fig_long_context(res, out):
    files = [(s, res / s / "long_context.csv") for s in ("S", "M", "L") if (res / s / "long_context.csv").exists()]
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 3.4), squeeze=False)
    for ax, (scale, p) in zip(axes[0], files):
        g = defaultdict(lambda: defaultdict(list))
        train_len = 1024
        for r in read_csv(p):
            g[r["arch"]][(int(r["pos_start"]), int(r["pos_end"]))].append(num(r["loss"]))
            train_len = int(r["train_seq_len"])
        for arch, bins in g.items():
            keys = sorted(bins)
            xs = [math.sqrt(max(a, 1) * b) for a, b in keys]
            ax.plot(xs, [mean_std(bins[k])[0] for k in keys], marker="o", ms=4, color=COLORS[arch],
                    label=NAMES[arch])
        ax.axvline(train_len, color=MUTED, ls="--", lw=1)
        ax.text(train_len, ax.get_ylim()[1], " training length", color=MUTED, fontsize=8, va="top")
        ax.set(xscale="log", xlabel="Token position in PG-19 book", ylabel="Loss (nats/token)",
               title=f"Scale {scale}: loss beyond the training context")
        ax.legend(fontsize=8)
    fig.savefig(out / "fig5_long_context.png")
    plt.close(fig)


def fig_synthetic(res, out):
    p_mqar, p_copy = res / "synthetic" / "mqar.csv", res / "synthetic" / "selective_copy.csv"
    panels = [p for p in (p_mqar, p_copy) if p.exists()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.4), squeeze=False)
    for ax, p in zip(axes[0], panels):
        best = defaultdict(float)  # best over learning rates, at the largest width tried
        rows = read_csv(p)
        width = max(int(r["d_model"]) for r in rows)
        for r in rows:
            if int(r["d_model"]) == width:
                k = (r["arch"], int(r["seq_len"]))
                best[k] = max(best[k], num(r["accuracy"]))
        for arch in COLORS:
            pts = sorted((L, a) for (ar, L), a in best.items() if ar == arch)
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", ms=5, color=COLORS[arch],
                        label=NAMES[arch])
        task = "MQAR (associative recall)" if p == p_mqar else "Selective copying"
        ax.set(xscale="log", xlabel="Sequence length", ylabel="Accuracy on answer tokens", ylim=(-0.03, 1.03),
               title=f"{task}, width {width}")
        ax.legend(fontsize=8)
    fig.savefig(out / "fig3_synthetic.png")
    plt.close(fig)
    # width sweep for MQAR: the Zoology-style plot
    if p_mqar.exists():
        rows = read_csv(p_mqar)
        lens = sorted({int(r["seq_len"]) for r in rows})
        fig, axes = plt.subplots(1, len(lens), figsize=(3.2 * len(lens), 3), squeeze=False, sharey=True)
        for ax, L in zip(axes[0], lens):
            best = defaultdict(float)
            for r in rows:
                if int(r["seq_len"]) == L:
                    k = (r["arch"], int(r["d_model"]))
                    best[k] = max(best[k], num(r["accuracy"]))
            for arch in COLORS:
                pts = sorted((d, a) for (ar, d), a in best.items() if ar == arch)
                if pts:
                    ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", ms=5, color=COLORS[arch],
                            label=NAMES[arch])
            ax.set(xscale="log", xlabel="Model width", title=f"L = {L}", ylim=(-0.03, 1.03))
        axes[0][0].set_ylabel("MQAR accuracy")
        axes[0][-1].legend(fontsize=7)
        fig.savefig(out / "fig3b_mqar_width.png")
        plt.close(fig)


def fig_bench(res, out):
    p = res / "bench" / "bench.csv"
    if not p.exists():
        return
    rows = [r for r in read_csv(p) if r["status"] == "ok"]
    batch = min(int(r["batch"]) for r in rows)
    rows = [r for r in rows if int(r["batch"]) == batch]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    specs = (("prefill_ms", "Prefill latency (ms)"),
             ("prefill_peak_mb", "Peak memory, prefill incl. cache (MiB)"),
             ("decode_ms_per_token", "Decode latency per new token (ms)"),
             ("decode_peak_mb", "Peak memory while decoding (MiB)"))
    for ax, (key, label) in zip(axes, specs):
        drawn = False
        series = defaultdict(list)
        for r in rows:
            series[(r["arch"], r["backend"], r["config"])].append((int(r["seq_len"]), num(r[key])))
        for (arch, backend, config), pts in sorted(series.items()):
            pts = sorted(p for p in pts if p[1] > 0)  # GPU-only metrics are NaN on CPU
            if not pts:
                continue
            drawn = True
            name = NAMES[arch] if arch == "transformer" else f"Mamba, {backend} scan"
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", ms=4, color=COLORS[arch],
                    ls=BACKEND_STYLE.get(backend, "-"), label=f"{name} ({config})")
        if not drawn:
            ax.text(0.5, 0.5, "not measured on this device", ha="center", transform=ax.transAxes, color=MUTED)
            ax.set(xlabel="Sequence / context length (tokens)", ylabel=label)
            continue
        ax.set(xscale="log", yscale="log", xlabel="Sequence / context length (tokens)", ylabel=label)
    axes[0].legend(fontsize=7)
    dev = rows[0].get("device", "")
    fig.suptitle(f"Inference cost vs length, batch {batch}, {dev}, {rows[0].get('precision', '')} "
                 "(missing points = out of memory, or skipped after a prefill over 20 s)", x=0.01, y=1.02,
                 ha="left")
    fig.savefig(out / "fig6_efficiency.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_training_curves, table_main, fig_recall_split, fig_long_context, fig_synthetic, fig_bench):
        try:
            fn(res, out)
        except Exception as e:  # noqa: BLE001 - one broken input should not stop the others
            print(f"{fn.__name__}: skipped ({e})")
    for f in sorted(out.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
