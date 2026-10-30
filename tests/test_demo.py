import math
import io
import json

import pytest
import torch

from afml.data import dataset_rows, fit_tokenizer, make_blocks, split_documents
from afml.demo import evaluate, generate
from afml.models import ModelConfig, build_model


def docs():
    return [{"id": str(i), "text": (f"Document {i}. A student learns how computers work. " * 12)} for i in range(10)]


def test_split_deduplicates_and_is_reproducible():
    records = docs() + docs()[:2] + [{"text": " "}]
    train, val = split_documents(records, seed=42)
    assert len(train) + len(val) == 10
    assert {r["sha256"] for r in train}.isdisjoint(r["sha256"] for r in val)
    assert (train, val) == split_documents(records, seed=42)


def test_blocks_keep_document_boundaries_and_shift_targets():
    train, _ = split_documents(docs())
    tokenizer = fit_tokenizer(train, 300)
    combined, _ = make_blocks(train, tokenizer, 16)
    separate = torch.cat([make_blocks([doc], tokenizer, 16)[0] for doc in train])
    torch.testing.assert_close(combined, separate)
    x, y = combined[:, :-1], combined[:, 1:]
    assert x.shape[1] == y.shape[1] == 16
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_evaluation_weights_partial_batch_by_token_count():
    class Toy(torch.nn.Module):
        def forward(self, x, y):
            return None, y.float().mean()
    model = Toy().train()
    rows = torch.arange(15).view(3, 5)
    assert evaluate(model, rows, "cpu", batch_size=2) == pytest.approx(rows[:, 1:].float().mean().item())
    assert model.training


def test_generation_and_checkpoint_roundtrip(tmp_path):
    train, _ = split_documents(docs())
    tok = fit_tokenizer(train, 300)
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_layers=1, n_heads=4, max_seq_len=16)
    model = build_model(cfg).eval()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(cfg).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    a = generate(model, tok, "A student", "cpu", max_new_tokens=3)
    b = generate(restored, tok, "A student", "cpu", max_new_tokens=3)
    assert a == b and a.startswith("A student")
    with pytest.raises(ValueError):
        generate(model, tok, "", "cpu")


def test_rows_api_checks_revision(monkeypatch):
    response = io.BytesIO(b'{}')
    response.headers = {"x-revision": "different"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: response)
    with pytest.raises(RuntimeError, match="revision"):
        list(dataset_rows("dataset", "subset", "expected"))


def test_rows_api_skips_truncated_text(monkeypatch):
    payload = {"rows": [{"row": {"text": "complete"}, "truncated_cells": []},
                        {"row": {"text": "partial"}, "truncated_cells": ["text"]}],
               "num_rows_total": 2}
    response = io.BytesIO(json.dumps(payload).encode())
    response.headers = {"x-revision": "expected"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: response)
    assert list(dataset_rows("dataset", "subset", "expected")) == [{"text": "complete"}]
