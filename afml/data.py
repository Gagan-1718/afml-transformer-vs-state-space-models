"""Small, reproducible FineWeb-Edu preparation for the Review 1 experiment."""
import hashlib
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def document_hash(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def split_documents(records, val_fraction=0.15, seed=42):
    """Deduplicate before splitting whole documents; never split token chunks randomly."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    unique = {}
    for row in records:
        text = row.get("text", "").strip()
        if text:
            key = document_hash(text)
            unique.setdefault(key, {"text": text, "sha256": key, "id": row.get("id", key)})
    docs = list(unique.values())
    if len(docs) < 4:
        raise ValueError("Need at least four unique non-empty documents")
    random.Random(seed).shuffle(docs)
    n_val = max(1, round(len(docs) * val_fraction))
    return docs[n_val:], docs[:n_val]


def fit_tokenizer(train_docs, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[EOS]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    tokenizer.train_from_iterator((row["text"] for row in train_docs), trainer=trainer)
    return tokenizer


def make_blocks(docs, tokenizer, seq_len):
    """Each row contains seq_len + 1 IDs. Windows stay inside a document."""
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    eos = tokenizer.token_to_id("[EOS]")
    rows, token_count, dropped = [], 0, 0
    for doc in docs:
        ids = tokenizer.encode(doc["text"]).ids + [eos]
        token_count += len(ids)
        starts = list(range(0, len(ids) - seq_len, seq_len))
        rows.extend(ids[start:start + seq_len + 1] for start in starts)
        dropped += max(0, len(ids) - 1 - len(starts) * seq_len)
    if not rows:
        raise ValueError("No full sequences: increase sample size or reduce sequence length")
    return torch.tensor(rows, dtype=torch.long), {
        "token_count": token_count, "blocks": len(rows),
        "scored_target_tokens": len(rows) * seq_len, "unscored_tail_targets": dropped,
    }


def prepare_data(cfg, output_dir):
    from huggingface_hub import HfApi

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Data directory is not empty. Choose a new directory to preserve its tokenizer and split.")
    out.mkdir(parents=True, exist_ok=True)
    revision = cfg.get("revision") or HfApi().dataset_info(cfg["dataset"]).sha
    loader = cfg.get("loader", "rows")
    if loader == "stream":
        from datasets import load_dataset
        stream = load_dataset(cfg["dataset"], name=cfg["subset"], split="train",
                              revision=revision, streaming=True)
    elif loader == "rows":
        stream = dataset_rows(cfg["dataset"], cfg["subset"], revision)
    else:
        raise ValueError("loader must be 'rows' or 'stream'")
    print(f"Loading {cfg['documents']} FineWeb-Edu documents using {loader}; revision {revision}", flush=True)
    records = []
    for row in stream:
        if row.get("text", "").strip():
            records.append(row)
        if len(records) >= cfg["documents"]:
            break
    if len(records) < cfg["documents"]:
        raise ValueError("Dataset ended before the requested sample size")
    train_docs, val_docs = split_documents(records, cfg["val_fraction"], cfg["seed"])
    tokenizer = fit_tokenizer(train_docs, cfg["vocab_size"])
    tokenizer.save(str(out / "tokenizer.json"))
    train, train_stats = make_blocks(train_docs, tokenizer, cfg["seq_len"])
    val, val_stats = make_blocks(val_docs, tokenizer, cfg["seq_len"])
    torch.save({"train": train, "validation": val}, out / "blocks.pt")
    manifest = {
        "dataset": cfg["dataset"], "subset": cfg["subset"], "revision": revision,
        "source_split": "train", "loader": loader,
        "sampling": "first N non-empty documents; rows API excludes truncated text",
        "requested_documents": cfg["documents"], "seed": cfg["seed"],
        "validation_fraction": cfg["val_fraction"], "seq_len": cfg["seq_len"],
        "tokenizer": "byte-level BPE fitted on training documents only",
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest(),
        "train": {**train_stats, "documents": len(train_docs)},
        "validation": {**val_stats, "documents": len(val_docs)},
        "train_documents": [{k: r[k] for k in ("id", "sha256")} for r in train_docs],
        "validation_documents": [{k: r[k] for k in ("id", "sha256")} for r in val_docs],
        "limitations": "Tiny ordered sample; exact-text deduplication only; validation is not a final test set.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def dataset_rows(dataset, subset, revision):
    """Read bounded pages for a tiny pilot, rejecting stale or truncated text."""
    offset = 0
    while True:
        query = urllib.parse.urlencode({"dataset": dataset, "config": subset, "split": "train",
                                        "offset": offset, "length": 100})
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=120) as response:
            observed = response.headers.get("x-revision")
            if observed != revision:
                raise RuntimeError(f"Dataset API revision {observed!r} does not match {revision}. Use loader='stream' for the pinned revision.")
            payload = json.load(response)
        rows = payload.get("rows", [])
        if not rows:
            return
        for item in rows:
            if "text" not in item.get("truncated_cells", []):
                yield item["row"]
        offset += len(rows)
        if offset >= payload["num_rows_total"]:
            return
