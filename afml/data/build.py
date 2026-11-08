"""Build tokenized FineWeb-Edu (and PG-19) shards for the study.

Pipeline (run once, save the output folder to Drive):
  1. Stream FineWeb-Edu `sample-10BT` at a pinned revision, shuffled with a fixed seed.
  2. Drop empty and exact-duplicate documents (SHA-256 of the stripped text).
  3. Assign each document to train / val / test from its hash, so the split never depends on order.
  4. Fit a byte-level BPE tokenizer on the first `tokenizer_docs` *training* documents only.
  5. Tokenize everything with that tokenizer, append [EOS] per document, and write uint16 files.

Output folder:
  tokenizer.json, train.bin, val.bin, test.bin, pg19_test.bin, pg19_test_offsets.npy, meta.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

SPECIALS = ["[UNK]", "[EOS]"]


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def split_of(h: str, val_permille: int, test_permille: int) -> str:
    bucket = int(h[:8], 16) % 1000
    if bucket < test_permille:
        return "test"
    if bucket < test_permille + val_permille:
        return "val"
    return "train"


def fit_tokenizer(texts, vocab_size):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


class ShardWriter:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.tokens = 0
        self.docs = 0

    def write(self, ids):
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(self.f)
        self.tokens += arr.size
        self.docs += 1

    def close(self):
        self.f.close()


def iter_fineweb(dataset, subset, revision, seed, shuffle_buffer):
    from datasets import load_dataset
    ds = load_dataset(dataset, name=subset, split="train", revision=revision, streaming=True)
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for row in ds:
        text = (row.get("text") or "").strip()
        if text:
            yield text


def build_fineweb(cfg, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        raise SystemExit(f"{out} already holds a finished build; choose a new folder")
    t0 = time.time()
    vp, tp = cfg["val_permille"], cfg["test_permille"]

    # Pass 1: tokenizer from training documents only.
    tok_texts, seen = [], set()
    for text in iter_fineweb(cfg["dataset"], cfg["subset"], cfg["revision"], cfg["seed"], cfg["shuffle_buffer"]):
        h = doc_hash(text)
        if h in seen or split_of(h, vp, tp) != "train":
            continue
        seen.add(h)
        tok_texts.append(text)
        if len(tok_texts) >= cfg["tokenizer_docs"]:
            break
    print(f"fitting {cfg['vocab_size']}-token BPE on {len(tok_texts):,} training documents", flush=True)
    tok = fit_tokenizer(tok_texts, cfg["vocab_size"])
    assert tok.get_vocab_size() <= 65535, "uint16 storage needs vocab < 65536"
    tok.save(str(out / "tokenizer.json"))
    eos = tok.token_to_id("[EOS]")
    del tok_texts

    # Pass 2: tokenize and write every split (same stream, same order, so the tokenizer docs are included).
    writers = {s: ShardWriter(out / f"{s}.bin") for s in ("train", "val", "test")}
    targets = {"train": cfg["train_tokens"], "val": cfg["val_tokens"], "test": cfg["test_tokens"]}
    seen, dupes, batch = set(), 0, []

    def flush():
        enc = tok.encode_batch([t for _, t in batch])
        for (split, _), e in zip(batch, enc):
            if writers[split].tokens < targets[split]:
                writers[split].write(e.ids + [eos])
        batch.clear()

    last = time.time()
    for text in iter_fineweb(cfg["dataset"], cfg["subset"], cfg["revision"], cfg["seed"], cfg["shuffle_buffer"]):
        h = doc_hash(text)
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        batch.append((split_of(h, vp, tp), text))
        if len(batch) >= 2048:
            flush()
            if time.time() - last > 60:
                last = time.time()
                print({s: f"{w.tokens / 1e6:.1f}M" for s, w in writers.items()}, flush=True)
            if all(writers[s].tokens >= targets[s] for s in writers):
                break
    if batch:
        flush()
    for w in writers.values():
        w.close()

    meta = {
        "dataset": cfg["dataset"], "subset": cfg["subset"], "revision": cfg["revision"],
        "shuffle_seed": cfg["seed"], "shuffle_buffer": cfg["shuffle_buffer"],
        "split_rule": f"int(sha256(text)[:8], 16) % 1000: <{tp} test, <{tp + vp} val, else train",
        "dedup": "exact SHA-256 of stripped text", "duplicates_dropped": dupes,
        "tokenizer": "byte-level BPE fitted on training documents only",
        "tokenizer_docs": cfg["tokenizer_docs"], "vocab_size": tok.get_vocab_size(), "eos_id": eos,
        "tokenizer_sha256": hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest(),
        "dtype": "uint16", "documents_separated_by": "[EOS]",
        "splits": {s: {"tokens": w.tokens, "documents": w.docs} for s, w in writers.items()},
        "build_seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["splits"], indent=2))
    return meta


def build_pg19(out: Path, n_books: int, min_tokens: int, revision=None, dataset="emozilla/pg19"):
    """PG-19 test books, each stored whole, for the long-context evaluation.
    Default source is a Parquet copy of PG-19: the original deepmind/pg19 repo needs a loading
    script, which recent `datasets` versions no longer run."""
    from datasets import load_dataset
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    eos = tok.token_to_id("[EOS]")
    ds = load_dataset(dataset, split="test", streaming=True, revision=revision)
    offsets, books = [0], []
    with open(out / "pg19_test.bin", "wb") as f:
        for row in ds:
            ids = tok.encode(row["text"]).ids + [eos]
            if len(ids) < min_tokens:
                continue
            np.asarray(ids, dtype=np.uint16).tofile(f)
            offsets.append(offsets[-1] + len(ids))
            books.append(row.get("short_book_title", ""))
            print(f"pg19 book {len(books)}: {len(ids):,} tokens", flush=True)
            if len(books) >= n_books:
                break
    np.save(out / "pg19_test_offsets.npy", np.asarray(offsets, dtype=np.int64))
    (out / "pg19_meta.json").write_text(json.dumps({"dataset": dataset, "split": "test", "books": books,
                                                    "min_tokens": min_tokens}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pg19-only", action="store_true", help="add PG-19 to an existing build")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    if not a.pg19_only:
        build_fineweb(cfg, out)
    build_pg19(out, cfg["pg19_books"], cfg["pg19_min_tokens"], cfg.get("pg19_revision"),
               cfg.get("pg19_dataset", "emozilla/pg19"))


if __name__ == "__main__":
    main()
