"""The dataset builder, with the Hugging Face download replaced by generated documents."""
import json
import sys
import types

import numpy as np

from afml.data import TokenData
from afml.data import build


def fake_docs(n=3000):
    words = "the a model state attention token memory learns reads writes long short".split()
    rng = np.random.default_rng(0)
    for i in range(n):
        yield f"doc {i} " + " ".join(rng.choice(words, size=rng.integers(20, 80)))
        if i % 50 == 0:
            yield "doc 0 duplicate"  # exact duplicates must be dropped


def test_build_writes_disjoint_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "iter_fineweb", lambda *a, **k: fake_docs())
    cfg = {"dataset": "x", "subset": "y", "revision": "r", "seed": 0, "shuffle_buffer": 0,
           "val_permille": 50, "test_permille": 50, "vocab_size": 300, "tokenizer_docs": 500,
           "train_tokens": 40_000, "val_tokens": 3_000, "test_tokens": 3_000}
    meta = build.build_fineweb(cfg, tmp_path)
    assert meta["duplicates_dropped"] > 0
    for split in ("train", "val", "test"):
        assert meta["splits"][split]["tokens"] > 0
        assert (tmp_path / f"{split}.bin").stat().st_size == 2 * meta["splits"][split]["tokens"]
    data = TokenData(tmp_path)
    assert int(data.tokens("train").max()) < meta["vocab_size"]
    assert (np.asarray(data.tokens("val")) == meta["eos_id"]).sum() == meta["splits"]["val"]["documents"]

    # PG-19: books shorter than min_tokens are skipped; offsets index whole books
    books = [{"text": "short book"}, {"text": " ".join(["long"] * 400), "short_book_title": "Long"},
             {"text": " ".join(["longer"] * 500), "short_book_title": "Longer"}]
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: iter(books)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    build.build_pg19(tmp_path, n_books=5, min_tokens=100)
    got = TokenData(tmp_path).pg19_books()
    assert len(got) == 2 and all(len(b) >= 100 for b in got)
    assert json.loads((tmp_path / "pg19_meta.json").read_text())["books"] == ["Long", "Longer"]
