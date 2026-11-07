"""Session-limit pause/resume, the sweep job list and the smoke-run report used by the notebook."""
import importlib.util
import json
import time
from pathlib import Path

import pytest
import torch

from afml.data import write_fake_dataset
from afml.train import load_config, pick_precision, train

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return write_fake_dataset(tmp_path_factory.mktemp("data"))


def _cfg(tmp_path, arch, extra):
    p = tmp_path / f"{arch}.json"
    p.write_text(json.dumps({
        "model": {"arch": arch, "d_model": 32, "n_layers": 2, **extra},
        "train": {"seq_len": 32, "micro_batch": 4, "grad_accum": 2, "total_tokens": 32 * 8 * 12,
                  "lr": 1e-2, "eval_every": 4, "log_every": 2, "eval_tokens": 2000, "save_every_minutes": 60}}))
    return p


def test_cpu_precision_is_fp32():
    assert pick_precision("auto", "cpu") == torch.float32


def test_pause_at_time_limit_then_resume(data_dir, tmp_path):
    cfg = _cfg(tmp_path, "mamba", {"scan": "auto"})
    out = tmp_path / "run"
    paused = train(load_config(cfg), data_dir, out, device="cpu", stop_at=time.time())  # already past
    assert paused["paused"] and paused["step"] == 1
    assert (out / "last.pt").exists() and not (out / "summary.json").exists()
    s = train(load_config(cfg), data_dir, out, device="cpu")
    assert s["steps"] == 12 and json.loads((out / "summary.json").read_text())["steps"] == 12
    assert s["scan_backend"] == "chunked" and s["eval_seconds"] > 0 and s["peak_memory_gb"] is None


def test_sweep_job_names_match_collection():
    sweep = _script("sweep_lr")
    jobs = sweep.sweep_jobs("S", ["transformer", "mamba"], [3e-4, 1.2e-3], 1000, "DATA", "RUNS", stop_at=5.0)
    assert [j[2].name for j in jobs] == ["S_transformer_lr0.0003", "S_transformer_lr0.0012",
                                         "S_mamba_lr0.0003", "S_mamba_lr0.0012"]
    assert all(j[2].parent.name == "sweep" and "--stop-at" in j[3] for j in jobs)


def test_smoke_report(data_dir, tmp_path):
    rep = _script("smoke_report")
    runs = []
    for arch, extra in (("transformer", {"n_heads": 2}), ("mamba", {"scan": "chunked"})):
        out = tmp_path / f"smoke_{arch}"
        train(load_config(_cfg(tmp_path, arch, extra)), data_dir, out, device="cpu", max_steps=8)
        runs.append(out)
    measured = [rep.measure(r) for r in runs]
    assert all(m["tokens_per_sec"] > 0 and m["eval_s_per_token"] > 0 for m in measured)
    assert [m["scan_backend"] for m in measured] == [None, "chunked"]
    t = {"transformer": rep.target_cost("S_transformer", 512), "mamba": rep.target_cost("S_mamba", 512)}
    assert t["transformer"]["total_tokens"] == 940_000_000
    targets = {"S_transformer": t["transformer"], "S_mamba": t["mamba"], "S_mamba_lti": t["mamba"],
               "M_transformer": t["transformer"], "M_mamba": t["mamba"]}
    est = rep.estimate(measured, targets)
    assert {e["target"] for e in est} == set(targets) and all(e["hours"] > 0 for e in est)
    plan = rep.stage_plan(est, gpus=2, session_hours=11)
    assert [p["runs"] for p in plan] == [8, 9, 2]
    assert all(p["wall_hours"] >= p["gpu_hours"] / 2 for p in plan)


def test_collect_results_copies_runs_and_data_meta(data_dir, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name in ("S_mamba_s0", "smoke_S_mamba_chunked"):
        (runs / name).mkdir(parents=True)
        (runs / name / "summary.json").write_text("{}")
        (runs / name / "best.pt").write_bytes(b"weights")
    res = tmp_path / "results"
    monkeypatch.setattr("sys.argv", ["collect_results.py", "--runs", str(runs), "--results", str(res),
                                     "--data", str(data_dir)])
    _script("collect_results").main()
    assert (res / "S" / "runs" / "S_mamba_s0" / "summary.json").exists()
    assert not (res / "S" / "runs" / "S_mamba_s0" / "best.pt").exists()  # no weights in results
    assert not list(res.rglob("smoke_*"))  # smoke runs are never collected
    assert json.loads((res / "data" / "meta.json").read_text()) == json.loads((data_dir / "meta.json").read_text())
