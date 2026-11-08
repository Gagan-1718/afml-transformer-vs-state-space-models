"""Memory-mapped access to the uint16 token files written by afml.data.build."""
import json
from pathlib import Path

import numpy as np
import torch


class TokenData:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self._maps = {}

    def tokens(self, split) -> np.memmap:
        if split not in self._maps:
            self._maps[split] = np.memmap(self.dir / f"{split}.bin", dtype=np.uint16, mode="r")
        return self._maps[split]

    def random_batch(self, split, batch, seq_len, generator: torch.Generator):
        """Random windows (can cross [EOS] document boundaries, as in GPT-style pretraining)."""
        data = self.tokens(split)
        starts = torch.randint(0, len(data) - seq_len - 1, (batch,), generator=generator).tolist()
        rows = np.stack([data[s:s + seq_len + 1] for s in starts]).astype(np.int64)
        rows = torch.from_numpy(rows)
        return rows[:, :-1], rows[:, 1:]

    def eval_windows(self, split, seq_len, max_tokens):
        """Deterministic, non-overlapping windows from the start of a split."""
        data = self.tokens(split)
        n = min(len(data) - 1, max_tokens) // seq_len
        rows = np.asarray(data[:n * seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(rows[:-1].reshape(n, seq_len))
        y = torch.from_numpy(rows[1:].reshape(n, seq_len))
        return x, y

    def pg19_books(self):
        data = np.memmap(self.dir / "pg19_test.bin", dtype=np.uint16, mode="r")
        offsets = np.load(self.dir / "pg19_test_offsets.npy")
        return [data[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def write_fake_dataset(out, vocab_size=512, n_tokens=(60_000, 8_000, 8_000), seed=0, pg19=(3, 4096)):
    """A tiny synthetic dataset with the real file layout, for tests and CPU smoke runs.
    Tokens follow a random bigram chain so models can actually learn something."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nxt = rng.integers(2, vocab_size, size=(vocab_size, 4))

    def stream(n):
        ids = np.empty(n, dtype=np.uint16)
        t = 2
        for i in range(n):
            t = nxt[t, rng.integers(0, 4)] if rng.random() > 0.02 else 1
            ids[i] = t
        return ids

    for split, n in zip(("train", "val", "test"), n_tokens):
        stream(n).tofile(out / f"{split}.bin")
    books = [stream(pg19[1]) for _ in range(pg19[0])]
    np.concatenate(books).tofile(out / "pg19_test.bin")
    np.save(out / "pg19_test_offsets.npy", np.cumsum([0] + [len(b) for b in books]).astype(np.int64))
    vocab = {"[UNK]": 0, "[EOS]": 1, **{f"t{i}": i for i in range(2, vocab_size)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.decoder = decoders.WordPiece(prefix="##")
    tok.save(str(out / "tokenizer.json"))
    meta = {"vocab_size": vocab_size, "eos_id": 1, "fake": True,
            "splits": {s: {"tokens": n} for s, n in zip(("train", "val", "test"), n_tokens)}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
