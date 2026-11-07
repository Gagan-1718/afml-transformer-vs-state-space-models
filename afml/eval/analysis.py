"""What does selectivity do? Records Mamba's step size delta_t on real text, and attention maps.

delta_t controls how much the state is overwritten at token t: exp(delta * A) near 0 means "forget
and write this token", near 1 means "ignore this token, keep the state". The effective memory
horizon of a channel is about 1 / (delta * |A|) tokens.

    python -m afml.eval.analysis --mamba runs/M_mamba_s0 --transformer runs/M_transformer_s0 \
        --data DATA --out results/analysis
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from ..data import TokenData
from .common import device_auto, load_model


@torch.no_grad()
def record_deltas(model, ids):
    blocks = [layer.block for layer in model.layers]
    for b in blocks:
        b.record_delta = True
    model(ids)
    deltas = torch.stack([b.last_delta[0] for b in blocks])  # (layers, l, d_inner)
    A = torch.stack([-torch.exp(b.A_log.float()).cpu() for b in blocks])  # (layers, d_inner, n)
    for b in blocks:
        b.record_delta = False
    return deltas, A


@torch.no_grad()
def record_attention(model, ids):
    for layer in model.layers:
        layer.attn.record_attention = True
    model(ids)
    maps = torch.stack([layer.attn.last_attention[0] for layer in model.layers])  # (layers, heads, l, l)
    for layer in model.layers:
        layer.attn.record_attention = False
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mamba", required=True)
    ap.add_argument("--transformer", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--offset", type=int, default=0, help="start position in the test split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = device_auto(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = TokenData(a.data)
    tok = Tokenizer.from_file(str(Path(a.data) / "tokenizer.json"))
    ids_np = np.asarray(data.tokens("test")[a.offset:a.offset + a.length], dtype=np.int64)
    ids = torch.from_numpy(ids_np).unsqueeze(0).to(device)
    words = [tok.decode([int(i)]) for i in ids_np]

    mamba = load_model(a.mamba, device=device, scan="chunked")
    deltas, A = record_deltas(mamba, ids)
    horizon = 1.0 / (deltas.mean(1).unsqueeze(-1) * A.abs()).clamp_min(1e-9)  # (layers, d_inner, n)
    np.savez_compressed(out / "deltas.npz", deltas=deltas.numpy(), A=A.numpy(), ids=ids_np)
    per_token = deltas.mean(-1)  # (layers, l)
    stats = {
        "layers": int(deltas.shape[0]), "tokens": int(deltas.shape[1]),
        "median_horizon_tokens_per_layer": [float(h.median()) for h in horizon],
        "delta_mean_per_layer": [float(d.mean()) for d in deltas],
        "eos_delta_ratio": None,
    }
    eos = data.meta.get("eos_id", 1)
    is_eos = torch.from_numpy(ids_np == eos)
    if is_eos.any():
        stats["eos_delta_ratio"] = float(per_token[:, is_eos].mean() / per_token[:, ~is_eos].mean())
    (out / "delta_stats.json").write_text(json.dumps(stats, indent=2))

    show = min(96, len(words))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(per_token[:, :show].numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(show), [w.replace("\n", "\\n") for w in words[:show]], rotation=90, fontsize=6)
    ax.set_ylabel("Mamba layer")
    ax.set_title("Mean step size delta per token (high = overwrite state with this token)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out / "delta_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, h in enumerate(horizon):
        vals = np.sort(h.flatten().numpy())
        ax.plot(vals, np.linspace(0, 1, len(vals)), label=f"layer {i}", lw=1)
    ax.set(xscale="log", xlabel="Effective memory horizon 1/(delta |A|), tokens", ylabel="Fraction of state channels",
           title="How far back each state channel remembers")
    ax.axvline(1024, color="grey", ls="--", lw=0.8)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "memory_horizon.png", dpi=180)
    plt.close(fig)

    if a.transformer:
        tf = load_model(a.transformer, device=device)
        maps = record_attention(tf, ids)
        np.savez_compressed(out / "attention.npz", attention=maps.numpy())
        n = maps.shape[0]
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        for i, ax in enumerate(np.atleast_1d(axes)):
            ax.imshow(maps[i].mean(0)[:show, :show].numpy() ** 0.5, cmap="magma")
            ax.set_title(f"layer {i}", fontsize=8)
            ax.axis("off")
        fig.suptitle("Transformer attention (head mean, sqrt scale)")
        fig.tight_layout()
        fig.savefig(out / "attention_maps.png", dpi=160)
        plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
