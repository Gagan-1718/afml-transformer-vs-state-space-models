"""Make every figure and results table from the CSVs in a results folder.

    python scripts/make_figures.py --results results --out results/figures

Inputs (any missing one is skipped) and outputs:
  {S,M,L}/runs/<run>/metrics.csv + summary.json        fig1_training_curves.png        E1
  synthetic_long/selective_copy_lr*{,_curve}.csv        fig2_selective_copy.png         E2
  synthetic_long/mqar_lr*{,_curve}.csv                  fig3_mqar.png                   E3
  synthetic_long/*.csv                                  table_synthetic_long.md         E2, E3
  {S,M,L}/perplexity.csv                                fig4_recall_split.png, table_main.md  E1, E3
  {S,M,L}/long_context.csv                              fig5_long_context.png           E4
  bench/bench.csv                                       fig6_efficiency.png             E5
  analysis/S/delta_stats.json                           fig7_memory_horizon.png         E6
  sft/test.csv, sft/lr_sweep.csv                        fig8_sft.png, table_sft.md      E7
  synthetic/{mqar,selective_copy}.csv (4,000 steps)     figA1_synthetic_4000_steps.png, figA2_mqar_width_4000_steps.png
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

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
    with open(path, encoding="utf-8") as f:
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


def model_order(arch):
    return list(COLORS).index(arch)


def short_name(arch):
    return NAMES[arch].split(" (")[0]


def plain_log_axis(axis, fmt=lambda v: f"{v:,.0f}" if v >= 1 else f"{v:g}", subs=None):
    """Log axis with plain-number labels (no 2x10^1); 1-2-5 ticks when the range is under ~2.5 decades."""
    lo, hi = axis.get_view_interval()
    if subs is None:
        subs = (1.0,) if hi / max(lo, 1e-12) > 300 else (1.0, 2.0, 5.0)
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    axis.set_minor_formatter(NullFormatter())


def fixed_log_x(ax, values, fmt=lambda v: f"{v:g}"):
    ax.set_xticks(values, [fmt(v) for v in values])
    ax.set_xticks([], minor=True)


def length_label(v):
    return f"{v / 1024:g}K" if v >= 1024 else f"{v:g}"


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
            for arch, runs in sorted(curves.items(), key=lambda kv: model_order(kv[0])):
                n = min(len(r) for r in runs)
                xs = [runs[0][i][xi] for i in range(n)]
                ys = [[r[i][2] for r in runs] for i in range(n)]
                m = [mean_std(v)[0] for v in ys]
                ax.plot(xs, m, color=COLORS[arch], label=f"{NAMES[arch]}: {m[-1]:.3f} (n={len(runs)})")
                if len(runs) > 1:
                    ax.fill_between(xs, [min(v) for v in ys], [max(v) for v in ys], color=COLORS[arch], alpha=0.15,
                                    lw=0)
            ax.set(xscale="log", xlabel=xlabel, ylabel="Validation loss (nats/token)", title=f"Scale {scale}")
            if xi == 0:
                plain_log_axis(ax.xaxis, fmt=lambda v: f"{v / 1e6:,.0f}M", subs=(1.0, 2.0, 3.0, 5.0))
            else:
                plain_log_axis(ax.xaxis, fmt=lambda v: "{}e{}".format(f"{v:.0e}".split("e")[0],
                                                                      int(f"{v:.0e}".split("e")[1])),
                               subs=(1.0, 2.0, 3.0, 5.0))
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
        for arch, rows in sorted(groups.items(), key=lambda kv: model_order(kv[0])):
            loss_m, loss_s = mean_std([num(r["all_loss"]) for r in rows])
            hit_m, _ = mean_std([num(r["hit_loss"]) for r in rows])
            oth_m, _ = mean_std([num(r["other_loss"]) for r in rows])
            r0 = rows[0]
            pm = f"{num(r0['params_total']) / 1e6:.1f}M / {num(r0['params_nonembedding']) / 1e6:.1f}M"
            lines.append(f"| {scale} | {NAMES[arch]} | {pm} | {num(r0['train_tokens']) / 1e6:.0f}M | "
                         f"{num(r0['train_flops']):.2e} | {len(rows)} | {loss_m:.4f} ± {loss_s:.4f} | "
                         f"{math.exp(loss_m):.2f} | {hit_m:.3f} | {oth_m:.3f} |")
    if found:
        (out / "table_main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fig_recall_split(res, out):
    data = []
    for scale in ("S", "M", "L"):
        p = res / scale / "perplexity.csv"
        if p.exists():
            g = defaultdict(list)
            for r in read_csv(p):
                g[r["arch"]].append((num(r["hit_loss"]), num(r["other_loss"])))
            for arch, v in sorted(g.items(), key=lambda kv: model_order(kv[0])):
                data.append((scale, arch, mean_std([a for a, _ in v])[0], mean_std([b for _, b in v])[0], len(v)))
    if not data:
        return
    one_scale = len({d[0] for d in data}) == 1
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    for ax, idx, title in ((axes[0], 2, "Recall-hit tokens (bigram seen earlier in the window)"),
                           (axes[1], 3, "All other tokens")):
        labels = [f"{short_name(d[1])}\n({d[4]} seeds)" if one_scale else f"{d[0]}\n{short_name(d[1])}"
                  for d in data]
        bars = ax.bar(range(len(data)), [d[idx] for d in data], color=[COLORS[d[1]] for d in data],
                      edgecolor="white", linewidth=2)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK)
        ax.set_xticks(range(len(data)), labels, fontsize=8)
        ax.set(ylabel="Test loss (nats/token)", title=title)
        ax.grid(axis="x", visible=False)
    if one_scale:
        fig.suptitle(f"Scale {data[0][0]}: test loss split by token type (mean over seeds)", x=0.01, y=1.04,
                     ha="left")
    fig.savefig(out / "fig4_recall_split.png")
    plt.close(fig)


def fig_long_context(res, out):
    files = [(s, res / s / "long_context.csv") for s in ("S", "M", "L") if (res / s / "long_context.csv").exists()]
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 3.4), squeeze=False)
    for ax, (scale, p) in zip(axes[0], files):
        g = defaultdict(lambda: defaultdict(list))
        train_len, books, runs = 1024, "", defaultdict(set)
        for r in read_csv(p):
            g[r["arch"]][(int(r["pos_start"]), int(r["pos_end"]))].append(num(r["loss"]))
            train_len, books = int(r["train_seq_len"]), r.get("books", "")
            runs[r["arch"]].add(r["run"])
        for arch, bins in sorted(g.items(), key=lambda kv: model_order(kv[0])):
            keys = sorted(bins)
            xs = [math.sqrt(max(a, 1) * b) for a, b in keys]
            ax.plot(xs, [mean_std(bins[k])[0] for k in keys], marker="o", ms=4, color=COLORS[arch],
                    label=f"{NAMES[arch]}, seed {', '.join(r.rsplit('_s', 1)[-1] for r in sorted(runs[arch]))}")
        ax.axvline(train_len, color=MUTED, ls="--", lw=1)
        ax.text(train_len, ax.get_ylim()[1], f" trained on {train_len:,}-token windows", color=MUTED, fontsize=8,
                va="top")
        ax.set(xscale="log", xlabel="Token position in the book (bin centre)", ylabel="Loss (nats/token)",
               title=f"Scale {scale}: loss by position in {books} PG-19 test books")
        plain_log_axis(ax.xaxis, subs=(1.0,))
        ax.legend(fontsize=7.5, loc="upper left")
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
        fixed_log_x(ax, sorted({int(r["seq_len"]) for r in rows}))
        ax.legend(fontsize=8)
    fig.suptitle("Short runs: 4,000 training steps per point, best of the learning rates tried (under-trained; "
                 "see fig2 and fig3 for 40,000 steps)", x=0.01, y=1.03, ha="left", fontsize=9)
    fig.savefig(out / "figA1_synthetic_4000_steps.png")
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
            fixed_log_x(ax, sorted({int(r["d_model"]) for r in rows}))
        axes[0][0].set_ylabel("MQAR accuracy")
        axes[0][-1].legend(fontsize=7)
        fig.suptitle("MQAR after 4,000 training steps, best of the learning rates tried (under-trained)", x=0.01,
                     y=1.04, ha="left", fontsize=9)
        fig.savefig(out / "figA2_mqar_width_4000_steps.png")
        plt.close(fig)


def first_step_at(curve, level=0.99):
    return next((s for s, a in curve if a >= level), None)


def read_long(res, task):
    """Summary rows and accuracy curves of the 40,000-step synthetic runs, keyed by (arch, d_model, seq_len, lr)."""
    rows, curves = [], defaultdict(list)
    for p in sorted((res / "synthetic_long").glob(f"{task}_lr*.csv")):
        if p.stem.endswith("_curve"):
            for r in read_csv(p):
                curves[(r["arch"], int(r["d_model"]), int(r["seq_len"]), num(r["lr"]))].append(
                    (int(r["step"]), num(r["accuracy"])))
        else:
            rows += read_csv(p)
    for v in curves.values():
        v.sort()
    return rows, curves


def fig_selective_copy_long(res, out):
    rows, curves = read_long(res, "selective_copy")
    if not curves:
        return
    lens, lrs = sorted({k[2] for k in curves}), sorted({k[3] for k in curves}, reverse=True)
    styles = dict(zip(lrs, ("-", "--", ":", "-.")))
    fig, axes = plt.subplots(1, len(lens), figsize=(5.2 * len(lens), 3.4), squeeze=False, sharey=True)
    for ax, L in zip(axes[0], lens):
        for (arch, d, seq, lr), pts in sorted(curves.items(), key=lambda kv: (list(COLORS).index(kv[0][0]), -kv[0][3])):
            if seq == L:
                ax.plot([s for s, _ in pts], [a for _, a in pts], color=COLORS[arch], ls=styles[lr], lw=1.8)
        width = next(k[1] for k in curves if k[2] == L)
        ax.set(xlabel="Training step", title=f"Sequence length {L} (width {width})", ylim=(-0.03, 1.03))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    axes[0][0].set_ylabel("Accuracy on copied tokens")
    handles = [Line2D([], [], color=COLORS[a], label=NAMES[a]) for a in COLORS]
    handles += [Line2D([], [], color=MUTED, ls=styles[lr], label=f"learning rate {lr:g}") for lr in lrs]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Selective copying: held-out accuracy during 40,000 training steps (one seed; runs stop at 99.5%)",
                 x=0.01, y=1.03, ha="left")
    fig.savefig(out / "fig2_selective_copy.png")
    plt.close(fig)


def fig_mqar_long(res, out):
    rows, curves = read_long(res, "mqar")
    if not curves:
        return
    widths = sorted({k[1] for k in curves})
    kv = int(rows[0]["num_kv_or_tokens"]) if rows else 16
    guess = sum(1 / k for k in range(1, kv + 1)) / kv  # pick at random among the values not yet asked for
    fig, axes = plt.subplots(1, len(widths), figsize=(4.4 * len(widths), 3.4), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], widths):
        for arch in COLORS:
            cands = {k: v for k, v in curves.items() if k[0] == arch and k[1] == d}
            if not cands:
                continue
            k = max(cands, key=lambda k: (max(a for _, a in cands[k]), -k[3]))
            pts = cands[k]
            best = max(a for _, a in pts)
            ax.plot([s for s, _ in pts], [a for _, a in pts], color=COLORS[arch],
                    label=f"{short_name(arch)}, lr {k[3]:g}: {best:.1%}")
        ax.axhline(guess, color=MUTED, ls=":", lw=1)
        ax.set(xlabel="Training step", title=f"Width {d}", ylim=(-0.03, 1.03))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), title="best learning rate: best accuracy",
                  title_fontsize=7.5)
    axes[0][0].set_ylabel("Accuracy on queried values")
    axes[0][0].text(axes[0][0].get_xlim()[1], guess + 0.03, f"{guess:.2f}: guessing among values not yet asked",
                    color=MUTED, fontsize=7.5, ha="right", va="bottom")
    lrs = sorted({k[3] for k in curves})
    fig.suptitle(f"MQAR, length 128 with {kv} key-value pairs: best of {len(lrs)} learning rates "
                 f"({', '.join(f'{lr:g}' for lr in lrs)}) per model, one seed, up to 40,000 steps",
                 x=0.01, y=1.03, ha="left")
    fig.savefig(out / "fig3_mqar.png")
    plt.close(fig)


def table_synthetic_long(res, out):
    lines = []
    for task, title in (("selective_copy", "Selective copying"), ("mqar", "MQAR")):
        rows, curves = read_long(res, task)
        if not rows:
            continue
        lines += [f"## {title} (40,000-step runs, one seed)", "",
                  "| Length | Width | Model | Learning rate | Best accuracy | First step at >= 99% | Steps run |",
                  "|" + " --- |" * 7]
        order = list(COLORS)
        for r in sorted(rows, key=lambda r: (int(r["seq_len"]), int(r["d_model"]), order.index(r["arch"]),
                                             num(r["lr"]))):
            key = (r["arch"], int(r["d_model"]), int(r["seq_len"]), num(r["lr"]))
            hit = first_step_at(curves.get(key, []))
            lines.append(f"| {r['seq_len']} | {r['d_model']} | {NAMES[r['arch']]} | {num(r['lr']):g} | "
                         f"{num(r['accuracy']):.1%} | {f'{hit:,}' if hit else 'not reached'} | "
                         f"{int(r['steps']):,} |")
        lines.append("")
    if lines:
        (out / "table_synthetic_long.md").write_text("\n".join(lines), encoding="utf-8")


def fig_bench(res, out):
    p = res / "bench" / "bench.csv"
    if not p.exists():
        return
    rows = [r for r in read_csv(p) if r["status"] == "ok"]
    batches = sorted({int(r["batch"]) for r in rows})
    specs = (("prefill_ms", "Prefill latency (ms)"),
             ("prefill_peak_mb", "Peak memory, prefill incl. cache (MiB)"),
             ("decode_ms_per_token", "Decode latency per new token (ms)"),
             ("decode_peak_mb", "Peak memory while decoding (MiB)"))
    fig, axes = plt.subplots(len(batches), 4, figsize=(17, 3.4 * len(batches)), squeeze=False)
    for row, batch in enumerate(batches):
        for ax, (key, label) in zip(axes[row], specs):
            drawn = False
            series = defaultdict(list)
            for r in rows:
                if int(r["batch"]) == batch:
                    series[(r["arch"], r["backend"])].append((int(r["seq_len"]), num(r[key])))
            for (arch, backend), pts in sorted(series.items(), key=lambda kv: (kv[0][0] != "transformer", kv[0])):
                pts = sorted(p for p in pts if p[1] > 0)  # GPU-only metrics are NaN on CPU
                if not pts:
                    continue
                drawn = True
                name = NAMES[arch] if arch == "transformer" else f"Mamba, {backend} scan"
                ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", ms=4, color=COLORS[arch],
                        ls=BACKEND_STYLE.get(backend, "-"), label=name)
            ax.set(xlabel="Sequence / context length (tokens)", ylabel=label, title=f"Batch {batch}")
            if not drawn:
                ax.text(0.5, 0.5, "not measured on this device", ha="center", transform=ax.transAxes, color=MUTED)
                continue
            ax.set_xscale("log")
            fixed_log_x(ax, sorted({int(r["seq_len"]) for r in rows}), fmt=length_label)
            if key != "decode_ms_per_token":  # decode latency spans a small range: keep it linear
                ax.set_yscale("log")
                plain_log_axis(ax.yaxis)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=max(len(labels), 1), fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    dev, prec = rows[0].get("device", ""), rows[0].get("precision", "").replace("torch.", "")
    fig.suptitle(f"Inference cost vs length on one {dev}, {prec}. Missing points: skipped after a prefill over "
                 "20 s (chunked scan only)", x=0.01, y=1.02, ha="left")
    fig.savefig(out / "fig6_efficiency.png")
    plt.close(fig)


def fig_memory_horizon(res, out):
    files = [res / s / "delta_stats.json" for s in ("analysis/S", "analysis/M", "analysis/L")]
    files = [f for f in files if f.exists()]
    if not files:
        return
    d = json.loads(files[0].read_text())
    h = d["median_horizon_tokens_per_layer"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(len(h)), h, color=COLORS["mamba"], edgecolor="white", linewidth=1.5, width=0.8)
    ax.set_yscale("log")
    ax.axhline(1024, color=MUTED, ls="--", lw=1)
    ax.text(len(h) - 0.5, 1024 * 1.12, "training context (1,024 tokens)", color=MUTED, fontsize=8, ha="right")
    for i in sorted(range(len(h)), key=lambda i: -h[i])[:2]:
        ax.text(i, h[i] * 1.08, f"{h[i]:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(h)))
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", visible=False)
    ax.set(xlabel="Mamba layer (0 = nearest the input)", ylabel="Median 1/(Δ |A|), tokens", ylim=(1, 3000),
           title=f"Mamba (scale {files[0].parent.name}, seed 0): median memory horizon of the state channels per "
                 f"layer, on {d['tokens']} test tokens")
    plain_log_axis(ax.yaxis, subs=(1.0,))
    fig.savefig(out / "fig7_memory_horizon.png")
    plt.close(fig)


def sft_rows(res):
    p = res / "sft" / "test.csv"
    return read_csv(p) if p.exists() else []


def fig_sft(res, out):
    rows = sft_rows(res)
    if not rows:
        return
    archs = [a for a in COLORS if any(r["arch"] == a for r in rows)]
    stages = ("pretrained", "instruction-tuned")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, key, label in ((axes[0], "test_response_loss", "Test loss on response tokens (nats/token)"),
                           (axes[1], "rouge_l", "ROUGE-L against the reference answer")):
        ticks, names = [], []
        for i, arch in enumerate(archs):
            for j, stage in enumerate(stages):
                r = next((r for r in rows if r["arch"] == arch and r["stage"] == stage), None)
                if r is None:
                    continue
                x, v = i * 2.6 + j, num(r[key])
                ax.bar(x, v, color=COLORS[arch], alpha=0.35 if stage == "pretrained" else 1.0, width=0.9,
                       edgecolor="white", linewidth=2)
                ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
                ticks.append(x)
                names.append("before" if stage == "pretrained" else "after")
            ax.text(i * 2.6 + 0.5, -0.13, short_name(arch), ha="center", va="top", fontsize=9, color=INK,
                    transform=blended_transform_factory(ax.transData, ax.transAxes))
        ax.set_xticks(ticks, names, fontsize=8)
        ax.set(ylabel=label)
        ax.grid(axis="x", visible=False)
    n = rows[0].get("generated", "")
    fig.suptitle(f"Instruction fine-tuning on Alpaca-cleaned: before vs after (test split; ROUGE-L on {n} prompts, "
                 "greedy decoding)", x=0.01, y=1.02, ha="left")
    fig.savefig(out / "fig8_sft.png")
    plt.close(fig)


def table_sft(res, out):
    rows = sft_rows(res)
    if not rows:
        return
    lines = ["## Test split", "",
             "| Model | Stage | Checkpoint | Response loss | Response ppl | ROUGE-L | Stops with EOS | "
             "Mean new tokens |", "|" + " --- |" * 8]
    order = list(COLORS)
    for r in sorted(rows, key=lambda r: (order.index(r["arch"]), r["stage"] != "pretrained")):
        lines.append(f"| {NAMES[r['arch']]} | {r['stage']} | {r['run']} | {num(r['test_response_loss']):.3f} | "
                     f"{num(r['test_response_ppl']):.2f} | {num(r['rouge_l']):.3f} | {num(r['eos_rate']):.0%} | "
                     f"{num(r['mean_new_tokens']):.1f} |")
    p = res / "sft" / "lr_sweep.csv"
    if p.exists():
        best = json.loads((res / "sft" / "best.json").read_text()) if (res / "sft" / "best.json").exists() else {}
        lines += ["", "## Learning-rate choice (validation split)", "",
                  "| Model | Learning rate | Val response loss before | Best val response loss | Chosen |",
                  "|" + " --- |" * 5]
        for r in sorted(read_csv(p), key=lambda r: (order.index(r["arch"]), num(r["lr"]))):
            chosen = "yes" if best.get(r["arch"]) == r["run"] else ""
            lines.append(f"| {NAMES[r['arch']]} | {num(r['lr']):g} | {num(r['init_val_loss']):.4f} | "
                         f"{num(r['best_val_loss']):.4f} | {chosen} |")
    (out / "table_sft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


FIGURES = (fig_training_curves, table_main, fig_selective_copy_long, fig_mqar_long, table_synthetic_long,
           fig_recall_split, fig_long_context, fig_bench, fig_memory_horizon, fig_sft, table_sft, fig_synthetic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        try:
            fn(res, out)
        except Exception as e:  # noqa: BLE001 - one broken input should not stop the others
            print(f"{fn.__name__}: skipped ({e})")
    for f in sorted(out.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
