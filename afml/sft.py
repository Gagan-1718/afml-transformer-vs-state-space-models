"""Instruction fine-tuning (supervised fine-tuning, SFT) of a pretrained model on Alpaca-cleaned.

The default course track asks for a small language model trained from scratch *and* instruction-tuned.
This starts from a pretrained run's best.pt (e.g. S_transformer_s0, S_mamba_s0) and fine-tunes every
weight on instruction -> response pairs. Only the response tokens (and the closing [EOS]) are scored;
the prompt is context. Both architectures get the same data, template, batch, epochs and learning-rate
grid; the learning rate is chosen on validation response loss, and the test split is scored once.

    python -m afml.sft prepare  --data DATA --out SFT_DATA
    python -m afml.sft train    --init RUNS/S_mamba_s0 --sft-data SFT_DATA --out RUNS/sft/S_mamba_s0_lr0.0001 --lr 1e-4
    python -m afml.sft pick     --runs RUNS/sft --out results/sft
    python -m afml.sft evaluate --pairs RUNS/S_mamba_s0=RUNS/sft/S_mamba_s0_lr0.0001 --sft-data SFT_DATA \
        --data DATA --out results/sft
"""
import argparse
import csv
import hashlib
from array import array
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .eval.common import device_auto, load_model
from .models import count_params, param_groups
from .train import DIVERGED_EXIT_CODE, lr_at, pick_precision, scan_backend

DATASET = "yahma/alpaca-cleaned"
REVISION = "12567cabf869d7c92e573c7c783905fc160e9639"  # pinned (CC BY 4.0, 2023-04-10)
IGNORE = -100  # target id that F.cross_entropy skips
PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_WITH_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

DEFAULT_SFT = {
    "seed": 0, "lr": 1e-4, "epochs": 2, "micro_batch": 8, "grad_accum": 8, "min_lr_ratio": 0.1,
    "warmup_frac": 0.03, "warmup_steps": None, "weight_decay": 0.0, "betas": [0.9, 0.95], "grad_clip": 1.0,
    "precision": "auto", "eval_every": 100, "log_every": 10,
}

# Hand-written prompts (not from Alpaca), only to show sample answers; nothing is chosen with them.
FIXED_PROMPTS = [
    {"instruction": "What is the capital of France?", "input": ""},
    {"instruction": "List three ways to save water at home.", "input": ""},
    {"instruction": "Explain what photosynthesis is in one or two sentences.", "input": ""},
    {"instruction": "Give one tip for studying for an exam.", "input": ""},
    {"instruction": "Write a short sentence describing the ocean.", "input": ""},
    {"instruction": "What is 12 plus 7?", "input": ""},
    {"instruction": "Summarize the text below in one sentence.",
     "input": "The sun is a star at the center of our solar system. It gives light and heat to Earth, "
              "and plants use its light to grow."},
    {"instruction": "Classify the sentiment of this review as positive or negative.",
     "input": "The food was cold and the service was very slow."},
]


def prompt_text(ex):
    instruction, inp = ex["instruction"].strip(), (ex.get("input") or "").strip()
    if inp:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
    return PROMPT.format(instruction=instruction)


def split_of(text, val_permille=20, test_permille=20):
    """Same hash rule as the pretraining data: the split depends only on the example's text."""
    bucket = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if bucket < test_permille:
        return "test"
    return "val" if bucket < test_permille + val_permille else "train"


# ---------------------------------------------------------------- data
def load_examples(source=None):
    if source:
        return json.loads(Path(source).read_text(encoding="utf-8"))
    from datasets import load_dataset
    return list(load_dataset(DATASET, revision=REVISION, split="train"))


def prepare(data_dir, out_dir, source=None, max_len=1024):
    """Dedup, split by hash, tokenize with the pretraining tokenizer, and write
    {train,val,test}.npz (ids, offsets, prompt_lens), test_examples.jsonl and meta.json."""
    from tokenizers import Tokenizer
    data_dir, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    data_meta = json.loads((data_dir / "meta.json").read_text())
    eos = data_meta["eos_id"]
    seen, counts = set(), {"raw": 0, "duplicates": 0, "empty": 0, "too_long": 0}
    parts = {s: {"ids": array("H"), "lens": [], "prompt_lens": [], "raw": []} for s in ("train", "val", "test")}
    for ex in load_examples(source):
        counts["raw"] += 1
        prompt, response = prompt_text(ex), (ex.get("output") or "").strip()
        if not ex["instruction"].strip() or not response:
            counts["empty"] += 1
            continue
        key = prompt + "\x00" + response
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        p = tok.encode(prompt).ids
        ids = p + tok.encode(response).ids + [eos]
        if len(ids) > max_len + 1:  # inputs and targets are each at most max_len tokens
            counts["too_long"] += 1
            continue
        part = parts[split_of(key)]
        part["ids"].extend(ids)
        part["lens"].append(len(ids))
        part["prompt_lens"].append(len(p))
        part["raw"].append({"instruction": ex["instruction"], "input": ex.get("input") or "", "output": response})
    stats = {}
    for split, part in parts.items():
        lens = np.asarray(part["lens"], dtype=np.int64)
        pl = np.asarray(part["prompt_lens"], dtype=np.int64)
        np.savez(out / f"{split}.npz", ids=np.array(part["ids"], dtype=np.uint16),
                 offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), prompt_lens=pl)
        stats[split] = {"examples": int(len(lens)), "tokens": int(lens.sum()),
                        "prompt_tokens": int(pl.sum()), "response_tokens": int((lens - pl).sum()),
                        "mean_tokens": round(float(lens.mean()), 1) if len(lens) else 0,
                        "max_tokens": int(lens.max()) if len(lens) else 0}
    with open(out / "test_examples.jsonl", "w", encoding="utf-8") as f:
        for r in parts["test"]["raw"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"dataset": DATASET if not source else str(source), "revision": REVISION if not source else None,
            "license": "CC BY 4.0", "template": PROMPT, "template_with_input": PROMPT_WITH_INPUT,
            "split_rule": "int(sha256(prompt + NUL + response)[:8], 16) % 1000: <20 test, <40 val, else train",
            "loss": "response tokens and the closing [EOS] only", "max_len": max_len, "eos_id": eos,
            "tokenizer_sha256": data_meta.get("tokenizer_sha256"), **counts, "splits": stats}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("raw", "duplicates", "empty", "too_long", "splits")}, indent=2))
    return meta


class SFTData:
    def __init__(self, sft_dir):
        self.dir = Path(sft_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.eos = self.meta["eos_id"]
        self._splits = {}

    def _load(self, split):
        if split not in self._splits:
            z = np.load(self.dir / f"{split}.npz")
            self._splits[split] = (z["ids"], z["offsets"], z["prompt_lens"])
        return self._splits[split]

    def lengths(self, split):
        return np.diff(self._load(split)[1])

    def examples(self, split, indices):
        ids, off, pl = self._load(split)
        return [(ids[off[i]:off[i + 1]].astype(np.int64), int(pl[i])) for i in indices]

    def ordered_batches(self, split, batch):
        """Deterministic batches sorted by length (little padding), for evaluation."""
        order = np.argsort(self.lengths(split), kind="stable")
        return [order[i:i + batch] for i in range(0, len(order), batch)]


def collate(examples, pad_id):
    """Right-pad a list of (ids, prompt_len). Targets are IGNORE on prompt tokens and padding, so only
    the response and its [EOS] are scored. Both models are causal, so right padding changes nothing."""
    width = max(len(s) for s, _ in examples) - 1
    x = torch.full((len(examples), width), pad_id, dtype=torch.long)
    y = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    for i, (s, p) in enumerate(examples):
        s = torch.as_tensor(s)
        n = len(s) - 1
        x[i, :n] = s[:-1]
        y[i, :n] = s[1:]
        y[i, :p - 1] = IGNORE  # y[j] = s[j + 1] is a prompt token when j + 1 < p
    return x, y


def epoch_batches(lengths, batch, rng):
    """Shuffle, sort by length inside pools of 50 batches (less padding), then shuffle the batches."""
    order = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(order), batch * 50):
        pool = order[start:start + batch * 50]
        pool = pool[np.argsort(lengths[pool], kind="stable")]
        batches += [pool[i:i + batch] for i in range(0, len(pool), batch)]
    return [batches[i] for i in rng.permutation(len(batches))]


@torch.no_grad()
def response_loss(model, data, split, device, dtype, batch=16):
    """Mean loss per response token over a whole split. Returns (loss, response tokens)."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for idx in data.ordered_batches(split, batch):
        x, y = collate(data.examples(split, idx), data.eos)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            _, loss = model(x, y, reduction="none")
        total += loss.sum().item()  # ignored targets contribute 0
        n += int((y != IGNORE).sum())
    model.train(was_training)
    return total / max(n, 1), n


# ---------------------------------------------------------------- training
def model_label(cfg):
    return cfg.arch if cfg.arch == "transformer" else ("mamba" if cfg.selective else "mamba_lti")


def train_sft(init_run, sft_dir, out_dir, device="auto", **overrides):
    """Fine-tune init_run/best.pt; writes best.pt (lowest validation response loss), metrics.csv,
    config.json and summary.json to out_dir. Starts fresh every time (a run takes minutes)."""
    tc = {**DEFAULT_SFT, **overrides}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "diverged.json", "metrics.csv", "best.pt"):
        (out / name).unlink(missing_ok=True)
    device = device_auto(device)
    data = SFTData(sft_dir)
    torch.manual_seed(tc["seed"])
    rng = np.random.default_rng(tc["seed"])
    model = load_model(init_run, device=device).train()
    mcfg = model.cfg
    (out / "config.json").write_text(json.dumps({"init": str(init_run), "sft_data": str(sft_dir), "sft": tc,
                                                 "model": asdict(mcfg)}, indent=2))
    dtype = pick_precision(tc["precision"], device)
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"], betas=tuple(tc["betas"]),
                            fused=device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    lengths = data.lengths("train")
    steps_per_epoch = math.ceil(math.ceil(len(lengths) / tc["micro_batch"]) / tc["grad_accum"])
    total = tc["epochs"] * steps_per_epoch
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"{model_label(mcfg)}: {count_params(model):,} params from {init_run}; {len(lengths):,} examples x "
          f"{tc['epochs']} epochs = {total} steps of {tc['micro_batch'] * tc['grad_accum']} examples; "
          f"lr {tc['lr']:g}; {dtype}; device {device}", flush=True)

    history, eval_seconds = [], 0.0

    def log_row(row):
        history.append(row)
        with (out / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in history for k in r}))
            w.writeheader()
            w.writerows(history)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    def timed_val():
        nonlocal eval_seconds
        t0 = time.time()
        v, _ = response_loss(model, data, "val", device, dtype)
        eval_seconds += time.time() - t0
        return v

    def save_best():
        tmp = out / "best.tmp"
        torch.save({"model": model.state_dict(), "model_config": asdict(mcfg), "step": step,
                    "init": str(init_run)}, tmp)
        tmp.replace(out / "best.pt")

    step, elapsed, seen_tokens, seen_resp = 0, 0.0, 0, 0
    init_val = best = timed_val()
    best_step = 0
    save_best()
    log_row({"step": 0, "epoch": 0.0, "val_loss": init_val, "val_ppl": math.exp(init_val)})
    run_sum, run_tok, t_mark = 0.0, 0, time.time()
    for epoch in range(tc["epochs"]):
        batches = epoch_batches(lengths, tc["micro_batch"], rng)
        for start in range(0, len(batches), tc["grad_accum"]):
            micro = [data.examples("train", b) for b in batches[start:start + tc["grad_accum"]]]
            group = [collate(exs, data.eos) for exs in micro]
            n_tok = sum(int((y != IGNORE).sum()) for _, y in group)  # response tokens in this step
            seen_tokens += sum(len(s) - 1 for exs in micro for s, _ in exs)  # real (unpadded) input tokens
            lr = lr_at(step, total, tc)
            for g in opt.param_groups:
                g["lr"] = lr
            for x, y in group:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                    _, loss = model(x, y, reduction="none")
                loss = loss.sum()
                if not torch.isfinite(loss):
                    (out / "diverged.json").write_text(json.dumps({"step": step, "lr": lr, "loss": str(loss.item())}))
                    raise FloatingPointError(f"non-finite loss at step {step} (lr {lr:.3g})")
                scaler.scale(loss / n_tok).backward()
                run_sum += loss.item()
            run_tok += n_tok
            seen_resp += n_tok
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            now = time.time()
            elapsed += now - t_mark
            t_mark = now
            if step % tc["log_every"] == 0 or step == total:
                row = {"step": step, "epoch": round(step / steps_per_epoch, 3), "lr": lr,
                       "train_loss": run_sum / max(run_tok, 1), "grad_norm": gnorm,
                       "tokens_per_sec": seen_tokens / max(elapsed, 1e-9), "elapsed_s": elapsed}
                if step % tc["eval_every"] == 0 or step == total:
                    v = timed_val()
                    row.update(val_loss=v, val_ppl=math.exp(v))
                    if v < best:
                        best, best_step = v, step
                        save_best()
                    t_mark = time.time()  # evaluation time is not training time
                log_row(row)
                run_sum, run_tok = 0.0, 0

    summary = {
        "arch": model_label(mcfg), "init": Path(init_run).name, "params_total": count_params(model),
        "lr": tc["lr"], "epochs": tc["epochs"], "steps": step, "best_step": best_step,
        "init_val_loss": init_val, "best_val_loss": best, "best_val_ppl": math.exp(best),
        "train_examples": int(len(lengths)), "response_tokens_trained": seen_resp, "tokens_processed": seen_tokens,
        "tokens_per_sec": seen_tokens / max(elapsed, 1e-9), "train_seconds": elapsed, "eval_seconds": eval_seconds,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
        "precision": str(dtype), "scan_backend": scan_backend(mcfg, device), "sft_config": tc,
        "dataset": data.meta["dataset"], "revision": data.meta["revision"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- choosing the learning rate (validation only)
def pick(runs_dir, out_dir, tie=0.01):
    """Per pretrained model, the fine-tuning run with the lowest validation response loss. Runs within
    `tie` of the best count as tied and the lowest learning rate wins (the study's pretraining rule)."""
    rows = []
    for cfg in sorted(Path(runs_dir).glob("*/config.json")):
        run = cfg.parent
        c = json.loads(cfg.read_text())
        row = {"run": run.name, "init": Path(c["init"]).name, "lr": c["sft"]["lr"]}
        if (run / "summary.json").exists():
            s = json.loads((run / "summary.json").read_text())
            row.update(arch=s["arch"], status="done", init_val_loss=s["init_val_loss"],
                       best_val_loss=s["best_val_loss"], best_step=s["best_step"], steps=s["steps"],
                       train_seconds=s["train_seconds"], tokens_per_sec=s["tokens_per_sec"],
                       scan_backend=s["scan_backend"])
        else:
            row.update(status="diverged" if (run / "diverged.json").exists() else "unfinished")
        rows.append(row)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["run", "init", "arch", "lr", "status", "init_val_loss", "best_val_loss", "best_step", "steps",
              "train_seconds", "tokens_per_sec", "scan_backend"]
    with open(out / "lr_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    best = {}
    for arch in sorted({r["arch"] for r in rows if r["status"] == "done"}):
        done = sorted((r for r in rows if r["status"] == "done" and r["arch"] == arch), key=lambda r: r["lr"])
        low = min(r["best_val_loss"] for r in done)
        chosen = next(r for r in done if r["best_val_loss"] <= low + tie)
        best[arch] = chosen["run"]
        if chosen["lr"] in (done[0]["lr"], done[-1]["lr"]) and len(done) > 2:
            print(f"WARNING: {arch}: the chosen fine-tuning LR {chosen['lr']:g} is at the edge of the grid")
        print(f"{arch}: {chosen['run']} (val response loss {chosen['best_val_loss']:.4f}, "
              f"before fine-tuning {chosen['init_val_loss']:.4f})")
    (out / "best.json").write_text(json.dumps(best, indent=2))
    for r in rows:  # keep the small evidence files (no weights) with the results
        dest = out / "runs" / r["run"]
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "metrics.csv", "config.json", "diverged.json"):
            if (Path(runs_dir) / r["run"] / name).exists():
                (dest / name).write_bytes((Path(runs_dir) / r["run"] / name).read_bytes())
    return best


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def greedy(model, prompt_ids, max_new, eos, device):
    """Greedy decoding with the KV cache / recurrent state. Returns (new ids, stopped at [EOS])."""
    model.eval()
    caches = model.init_cache(1, len(prompt_ids) + max_new, device=device)
    logits = model.prefill(torch.tensor([prompt_ids], device=device), caches)
    out = []
    for i in range(max_new):
        nxt = int(logits[0].argmax())
        if nxt == eos:
            return out, True
        out.append(nxt)
        if i + 1 < max_new:
            logits = model.decode_step(torch.tensor([nxt], device=device), caches)
    return out, False


def rouge_l(pred, ref):
    """ROUGE-L F1 on lower-cased whitespace words (longest common subsequence)."""
    a, b = pred.lower().split(), ref.lower().split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def evaluate(pairs, sft_dir, data_dir, out_dir, n_generate=100, max_new=128, device="auto"):
    """pairs: [(pretrained run dir, fine-tuned run dir)]. Scores the test split once for both stages of each
    model: response loss on every test example, plus greedy answers to the first n_generate test prompts
    (ROUGE-L against the reference, share of answers that end with [EOS]) and to FIXED_PROMPTS."""
    from tokenizers import Tokenizer
    device = device_auto(device)
    data = SFTData(sft_dir)
    tok = Tokenizer.from_file(str(Path(data_dir) / "tokenizer.json"))
    test_raw = [json.loads(l) for l in (Path(sft_dir) / "test_examples.jsonl").read_text(encoding="utf-8").splitlines()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_meta.json").write_text(json.dumps(data.meta, indent=2))  # split sizes, for the report
    rows, gens = [], []
    for base, tuned in pairs:
        for stage, run in (("pretrained", base), ("instruction-tuned", tuned)):
            model = load_model(run, device=device)
            arch = model_label(model.cfg)
            dtype = pick_precision("auto", device)
            loss, n_tok = response_loss(model, data, "test", device, dtype)
            scores, eos_hits, new_tokens = [], 0, 0
            prompts = [("test", i, ex) for i, ex in enumerate(test_raw[:n_generate])] + \
                      [("fixed", i, ex) for i, ex in enumerate(FIXED_PROMPTS)]
            for source, i, ex in prompts:
                ids, stopped = greedy(model, tok.encode(prompt_text(ex)).ids, max_new, data.eos, device)
                text = tok.decode(ids).strip()
                if source == "test":
                    scores.append(rouge_l(text, ex["output"]))
                    eos_hits += stopped
                    new_tokens += len(ids)
                gens.append({"arch": arch, "stage": stage, "run": Path(run).name, "source": source, "id": i,
                             "instruction": ex["instruction"], "input": ex.get("input", ""),
                             "reference": ex.get("output"), "output": text, "ended_with_eos": stopped})
            n = max(len(scores), 1)
            row = {"arch": arch, "stage": stage, "run": Path(run).name, "test_examples": len(data.lengths("test")),
                   "test_response_tokens": n_tok, "test_response_loss": loss, "test_response_ppl": math.exp(loss),
                   "generated": len(scores), "max_new_tokens": max_new, "rouge_l": sum(scores) / n,
                   "eos_rate": eos_hits / n, "mean_new_tokens": new_tokens / n, "decoding": "greedy"}
            print(row, flush=True)
            rows.append(row)
            del model
    with open(out / "test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    lines = ["# Sample answers (greedy decoding, at most %d new tokens)" % max_new, "",
             "Hand-written prompts, not from the training data. Nothing was chosen with them.", ""]
    for i, ex in enumerate(FIXED_PROMPTS):
        lines += [f"## {i + 1}. {ex['instruction']}", ""]
        if ex["input"]:
            lines += [f"Input: {ex['input']}", ""]
        for g in gens:
            if g["source"] == "fixed" and g["id"] == i:
                lines += [f"- **{g['arch']}, {g['stage']}**{'' if g['ended_with_eos'] else ' (no [EOS])'}: "
                          f"{g['output'][:600].replace(chr(10), ' ') or '(empty)'}"]
        lines.append("")
    (out / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True, help="pretraining data folder (tokenizer.json, meta.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--source", default=None, help="local JSON list instead of the Hugging Face download")
    p.add_argument("--max-len", type=int, default=1024)
    t = sub.add_parser("train")
    t.add_argument("--init", required=True, help="pretrained run folder with best.pt")
    t.add_argument("--sft-data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="auto")
    t.add_argument("--set", action="append", default=[], help="override, e.g. epochs=3")
    for key in ("lr", "epochs", "seed"):
        t.add_argument(f"--{key}", type=float if key == "lr" else int, default=None)
    k = sub.add_parser("pick")
    k.add_argument("--runs", required=True)
    k.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--pairs", nargs="+", required=True, help="PRETRAINED_RUN=FINETUNED_RUN ...")
    e.add_argument("--sft-data", required=True)
    e.add_argument("--data", required=True, help="pretraining data folder (for tokenizer.json)")
    e.add_argument("--out", required=True)
    e.add_argument("--n-generate", type=int, default=100)
    e.add_argument("--max-new", type=int, default=128)
    e.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.data, a.out, a.source, a.max_len)
    elif a.cmd == "train":
        overrides = {k: getattr(a, k) for k in ("lr", "epochs", "seed") if getattr(a, k) is not None}
        for item in a.set:
            key, value = item.split("=", 1)
            overrides[key] = json.loads(value)
        try:
            s = train_sft(a.init, a.sft_data, a.out, a.device, **overrides)
        except FloatingPointError as err:
            print(f"diverged: {err}. Details in diverged.json.", flush=True)
            sys.exit(DIVERGED_EXIT_CODE)
        print(json.dumps({k: s[k] for k in ("arch", "lr", "init_val_loss", "best_val_loss", "train_seconds")}))
    elif a.cmd == "pick":
        pick(a.runs, a.out)
    else:
        evaluate([tuple(p.split("=", 1)) for p in a.pairs], a.sft_data, a.data, a.out, a.n_generate, a.max_new,
                 a.device)


if __name__ == "__main__":
    main()
