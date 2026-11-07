"""End-to-end smoke tests on a tiny synthetic dataset with the real file layout."""
import json
import math

import pytest
import torch

from afml.data import TokenData, write_fake_dataset
from afml.data.build import doc_hash, split_of
from afml.eval.perplexity import evaluate_split, recall_hit_mask
from afml.eval.synthetic import IGNORE, make_mqar, make_selective_copy
from afml.train import load_config, lr_at, train


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return write_fake_dataset(tmp_path_factory.mktemp("data"))


def test_split_is_a_pure_function_of_text():
    h = doc_hash("  some document \n")
    assert h == doc_hash("some document")
    assert split_of(h, 10, 10) == split_of(h, 10, 10)
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(20_000):
        counts[split_of(doc_hash(f"doc {i}"), 10, 10)] += 1
    assert 150 < counts["val"] < 250 and 150 < counts["test"] < 250


def test_loader_windows_are_shifted(data_dir):
    data = TokenData(data_dir)
    x, y = data.random_batch("train", 4, 32, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 32)
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
    ex, ey = data.eval_windows("val", 64, 1000)
    assert ex.shape == (15, 64)
    torch.testing.assert_close(ex.reshape(-1)[1:], ey.reshape(-1)[:-1])


def test_lr_schedule():
    cfg = {"lr": 1e-3, "min_lr_ratio": 0.1, "warmup_frac": 0.1}
    assert lr_at(0, 100, cfg) == pytest.approx(1e-4)
    assert lr_at(9, 100, cfg) == pytest.approx(1e-3)
    assert lr_at(99, 100, cfg) == pytest.approx(1e-4, rel=0.05)
    assert all(lr_at(s, 100, cfg) >= lr_at(s + 1, 100, cfg) for s in range(10, 99))


@pytest.mark.parametrize("arch,extra", [("transformer", {"n_heads": 2}), ("mamba", {"scan": "chunked"})])
def test_train_learns_and_resumes(data_dir, tmp_path, arch, extra):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "model": {"arch": arch, "d_model": 64, "n_layers": 2, **extra},
        "train": {"seq_len": 32, "micro_batch": 8, "grad_accum": 2, "total_tokens": 32 * 16 * 100,
                  "lr": 1e-2, "eval_every": 25, "log_every": 25, "eval_tokens": 4000, "save_every_minutes": 0}}))
    out = tmp_path / "run"
    s = train(load_config(cfg_path), data_dir, out, device="cpu", max_steps=50)
    assert s["steps"] == 50 and s["best_val_loss"] < math.log(512) - 1.0
    s2 = train(load_config(cfg_path), data_dir, out, device="cpu")  # resumes from last.pt
    assert s2["steps"] == 100 and s2["best_val_loss"] < 3.0  # near the bigram chain's entropy (~1.4)
    rows = (out / "metrics.csv").read_text().splitlines()
    assert "step" in rows[0].split(",")
    assert len(rows) == 6  # header + steps 0, 25, 50, 75, 100 (no duplicates after resume)


def test_recall_hit_mask():
    x = torch.tensor([[5, 6, 5, 6, 7]])
    y = torch.tensor([[6, 5, 6, 7, 1]])
    assert recall_hit_mask(x, y, 10).tolist() == [[False, False, True, False, False]]


def test_evaluate_split_consistency():
    from afml.models import ModelConfig, build_model
    model = build_model(ModelConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2)).eval()
    x = torch.randint(0, 50, (3, 40))
    y = torch.randint(0, 50, (3, 40))
    r = evaluate_split(model, x, y, "cpu")
    mixed = (r["hit_loss"] * r["hit_tokens"] + r["other_loss"] * r["other_tokens"]) / r["all_tokens"]
    assert r["all_loss"] == pytest.approx(mixed, rel=1e-5)


def test_synthetic_generators():
    g = torch.Generator().manual_seed(0)
    x, y = make_mqar(8, 64, 8, 256, g)
    mask = y != IGNORE
    assert mask.sum(1).tolist() == [8] * 8
    for i in range(8):  # every answer is the value paired with that key in the prefix
        prefix = {int(x[i, j]): int(x[i, j + 1]) for j in range(0, 16, 2)}
        for p in mask[i].nonzero().flatten().tolist():
            assert prefix[int(x[i, p])] == int(y[i, p])
    x, y = make_selective_copy(4, 48, 6, 16, g)
    assert x.shape == (4, 54)
    for i in range(4):
        data = x[i, :48][x[i, :48] > 1]
        assert x[i, 48] == 1  # separator
        assert torch.equal(y[i, -6:], data) and (y[i, :48] == IGNORE).all()
