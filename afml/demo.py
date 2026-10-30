"""Train, evaluate and sample the small Transformer baseline."""
import csv
import json
import math
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .models import ModelConfig, build_model, count_params


@torch.no_grad()
def evaluate(model, blocks, device, batch_size=8):
    was_training = model.training
    model.eval()
    loss_sum, tokens = 0.0, 0
    for rows in blocks.split(batch_size):
        rows = rows.to(device)
        _, loss = model(rows[:, :-1], rows[:, 1:])
        n = rows[:, 1:].numel()
        loss_sum += loss.item() * n
        tokens += n
    model.train(was_training)
    return loss_sum / tokens


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=60, temperature=0.8, top_k=40, seed=42):
    if not prompt or temperature <= 0 or max_new_tokens < 1 or top_k < 1:
        raise ValueError("Use a non-empty prompt and positive sampling settings")
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    eos = tokenizer.token_to_id("[EOS]")
    for _ in range(max_new_tokens):
        # Full forward pass over a bounded context; this is not a KV-cache speed benchmark.
        logits, _ = model(ids[:, -model.cfg.max_seq_len:])
        scores = logits[:, -1].float() / temperature
        scores[:, tokenizer.token_to_id("[UNK]")] = -float("inf")
        cutoff = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
        scores = scores.masked_fill(scores < cutoff, -float("inf"))
        token = torch.multinomial(scores.softmax(-1), 1, generator=rng)
        if token.item() == eos:
            break
        ids = torch.cat((ids, token), dim=1)
    return tokenizer.decode(ids[0].tolist())


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for key, label in [("train_loss", "Fixed training subset"), ("validation_loss", "Held-out validation")]:
        ax.plot([r["step"] for r in history], [r[key] for r in history], marker="o", label=label)
    ax.set(xlabel="Optimizer steps", ylabel="Mean next-token loss (nats)",
           title="Transformer pilot: loss during training")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_demo(cfg, data_dir, run_dir, device="auto"):
    out, data_dir = Path(run_dir), Path(data_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Run directory is not empty. Choose a new directory to preserve earlier results.")
    out.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or enable a GPU runtime.")
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    data = torch.load(data_dir / "blocks.pt", map_location="cpu", weights_only=True)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    model_cfg = ModelConfig(arch="transformer", vocab_size=tokenizer.get_vocab_size(),
                            max_seq_len=manifest["seq_len"], **cfg["model"])
    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = torch.Generator().manual_seed(cfg["seed"])
    eval_rng = torch.Generator().manual_seed(cfg["seed"] + 1)
    train_eval = data["train"][torch.randperm(len(data["train"]), generator=eval_rng)[:cfg["train_eval_blocks"]]]
    validation = data["validation"]
    batch_size = cfg["batch_size"]
    history, best = [], float("inf")
    started = time.perf_counter()
    best_step = 0

    def record(step):
        nonlocal best, best_step
        train_loss = evaluate(model, train_eval, device, batch_size)
        val_loss = evaluate(model, validation, device, batch_size)
        if not math.isfinite(train_loss + val_loss):
            raise RuntimeError("Non-finite evaluation loss")
        row = {"step": step, "train_loss": train_loss, "validation_loss": val_loss,
               "validation_perplexity": math.exp(val_loss),
               "training_tokens_seen": step * batch_size * model_cfg.max_seq_len}
        history.append(row)
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if val_loss < best:
            best, best_step = val_loss, step
            torch.save({"model_config": asdict(model_cfg), "state_dict": model.state_dict(),
                        "step": step, "validation_loss": val_loss}, out / "best.pt")
        print(f"step={step:4d} train={train_loss:.4f} validation={val_loss:.4f} ppl={math.exp(val_loss):.2f}", flush=True)

    record(0)
    for step in range(1, cfg["steps"] + 1):
        model.train()
        rows = data["train"][torch.randint(len(data["train"]), (batch_size,), generator=rng)].to(device)
        _, loss = model(rows[:, :-1], rows[:, 1:])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            record(step)
    if device == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer.save(str(out / "tokenizer.json"))
    samples = [{"prompt": p, "text": generate(model, tokenizer, p, device, seed=cfg["seed"] + i)}
               for i, p in enumerate(cfg["prompts"])]
    summary = {
        "experiment": "Review 1 Transformer pilot; not a Transformer versus Mamba comparison",
        "seed": cfg["seed"], "parameters": count_params(model), "model_config": asdict(model_cfg),
        "training_config": cfg, "initial": history[0], "final": history[-1],
        "best_validation_loss": best, "best_validation_perplexity": math.exp(best), "best_step": best_step,
        "train_eval_blocks": len(train_eval), "validation_blocks": len(validation),
        "elapsed_seconds_including_evaluation_and_checkpointing": duration,
        "device": device, "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "python": platform.python_version(), "torch": torch.__version__, "precision": "float32",
        "samples": samples,
        "sampling": {"temperature": 0.8, "top_k": 40, "max_new_tokens": 60, "seed_base": cfg["seed"]},
        "data_manifest": manifest,
        "limitations": ["Tiny sample and short run", "One seed", "Validation used for checkpoint selection; no final test result",
                        "Perplexity depends on this tokenizer", "Uncached generation is a demonstration, not an efficiency benchmark",
                        "Mamba training and architecture comparison remain future work"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_history(history, out / "loss_curve.png")
    (out / "samples.txt").write_text("\n\n".join(f"PROMPT: {s['prompt']}\nOUTPUT: {s['text']}" for s in samples), encoding="utf-8")
    return summary
