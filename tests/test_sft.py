"""Instruction fine-tuning: data preparation, loss masking, training, LR choice and evaluation."""
import json
import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from afml import sft
from afml.data import write_fake_dataset
from afml.models import ModelConfig, build_model


def fake_examples(n=600, seed=0):
    """Instruction = a few tokens; response = the same tokens shifted by 100 (learnable)."""
    rng = np.random.default_rng(seed)
    exs = []
    for i in range(n):
        words = rng.integers(2, 60, size=rng.integers(2, 6))
        exs.append({"instruction": " ".join(f"t{w}" for w in words), "input": "t70 t71" if i % 3 == 0 else "",
                    "output": " ".join(f"t{w + 100}" for w in words)})
    exs.append(dict(exs[0]))  # exact duplicate
    exs.append({"instruction": "t5", "input": "", "output": "   "})  # empty response
    exs.append({"instruction": "t3", "input": "", "output": " ".join(["t200"] * 100)})  # longer than max_len
    return exs


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return write_fake_dataset(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="module")
def sft_dir(data_dir, tmp_path_factory):
    d = tmp_path_factory.mktemp("sft")
    (d / "src.json").write_text(json.dumps(fake_examples()))
    sft.prepare(data_dir, d / "out", source=d / "src.json", max_len=64)
    return d / "out"


def test_collate_scores_only_the_response():
    x, y = sft.collate([(np.array([10, 11, 12, 20, 21, 1]), 3), (np.array([10, 20, 1]), 1)], pad_id=1)
    assert x.tolist() == [[10, 11, 12, 20, 21], [10, 20, 1, 1, 1]]
    assert y.tolist() == [[-100, -100, 20, 21, 1], [20, 1, -100, -100, -100]]


def test_prepare_dedups_splits_and_ends_with_eos(sft_dir):
    meta = json.loads((sft_dir / "meta.json").read_text())
    assert meta["duplicates"] >= 1 and meta["empty"] == 1 and meta["too_long"] == 1
    kept = sum(s["examples"] for s in meta["splits"].values())
    assert kept == meta["raw"] - meta["duplicates"] - meta["empty"] - meta["too_long"]
    data = sft.SFTData(sft_dir)
    for split in ("train", "val", "test"):
        exs = data.examples(split, range(len(data.lengths(split))))
        assert exs and all(s[-1] == data.eos and 0 < p < len(s) - 1 for s, p in exs)
    assert len((sft_dir / "test_examples.jsonl").read_text().splitlines()) == meta["splits"]["test"]["examples"]
    assert sft.split_of("same text") == sft.split_of("same text")


def test_epoch_batches_use_every_example_once():
    batches = sft.epoch_batches(np.arange(1, 101), 8, np.random.default_rng(0))
    assert sorted(np.concatenate(batches).tolist()) == list(range(100))


def test_rouge_l():
    assert sft.rouge_l("a b c", "a c") == pytest.approx(0.8)
    assert sft.rouge_l("x y", "X y") == 1.0 and sft.rouge_l("", "a") == 0.0


@pytest.mark.parametrize("arch", ["transformer", "mamba"])
def test_finetune_learns_then_pick_and_evaluate(arch, data_dir, sft_dir, tmp_path):
    cfg = ModelConfig(arch=arch, vocab_size=512, d_model=32, n_layers=2, n_heads=2, scan="chunked")
    init = tmp_path / f"S_{arch}_s0"
    init.mkdir()
    torch.manual_seed(0)
    torch.save({"model": build_model(cfg).state_dict(), "model_config": asdict(cfg)}, init / "best.pt")
    runs, results = tmp_path / "sft", tmp_path / "results"
    small = {"micro_batch": 8, "grad_accum": 2, "eval_every": 10, "log_every": 5}
    s = sft.train_sft(init, sft_dir, runs / "lr_high", device="cpu", lr=1e-2, epochs=2, **small)
    n_train = len(sft.SFTData(sft_dir).lengths("train"))
    assert s["steps"] == 2 * math.ceil(math.ceil(n_train / 8) / 2)
    assert s["best_val_loss"] < s["init_val_loss"] - 1.0 and s["scan_backend"] == (
        "chunked" if arch == "mamba" else None)
    sft.train_sft(init, sft_dir, runs / "lr_low", device="cpu", lr=1e-5, epochs=1, **small)
    assert sft.pick(runs, results) == {arch: "lr_high"}  # chosen on validation loss
    assert (results / "runs" / "lr_low" / "summary.json").exists() and not list(results.rglob("*.pt"))
    rows = sft.evaluate([(init, runs / "lr_high")], sft_dir, data_dir, results, n_generate=3, max_new=8,
                        device="cpu")
    assert [r["stage"] for r in rows] == ["pretrained", "instruction-tuned"]
    assert rows[1]["test_response_loss"] < rows[0]["test_response_loss"]
    assert all(0 <= r["rouge_l"] <= 1 and 0 <= r["eos_rate"] <= 1 for r in rows)
    gens = (results / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(gens) == 2 * (3 + len(sft.FIXED_PROMPTS))
    assert "What is the capital of France?" in (results / "samples.md").read_text(encoding="utf-8")
    assert json.loads((results / "data_meta.json").read_text())["splits"]["test"]["examples"] == rows[0]["test_examples"]
