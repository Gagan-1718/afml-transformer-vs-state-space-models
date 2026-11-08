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
    assert [j[2].name for j in jobs] == ["S_transformer_lr0.0003_w51", "S_transformer_lr0.0012_w51",
                                         "S_mamba_lr0.0003_w51", "S_mamba_lr0.0012_w51"]
    assert all(j[2].parent.name == "sweep" and "--stop-at" in j[3] for j in jobs)
    # sweep runs warm up for the main run's 51 steps (2% of 2,593), not 2% of their own length
    assert all("train.warmup_steps=51" in j[3] for j in jobs)
    per_arch = sweep.sweep_jobs("S", ["transformer", "mamba"], sweep.DEFAULT_LRS, 1000, "DATA", "RUNS")
    assert [j[1] for j in per_arch if j[0] == "transformer"] == sweep.DEFAULT_LRS["transformer"]
    assert [j[1] for j in per_arch if j[0] == "mamba"] == sweep.DEFAULT_LRS["mamba"]


def test_diverged_run_writes_marker_and_exit_code(data_dir, tmp_path, monkeypatch):
    from afml import train as train_mod
    out = tmp_path / "run"
    cfg = _cfg(tmp_path, "transformer", {"n_heads": 2})
    monkeypatch.setattr("sys.argv", ["train", "--config", str(cfg), "--data", str(data_dir), "--out", str(out),
                                     "--device", "cpu", "--set", "train.lr=1e30"])  # inf weights after step 0
    with pytest.raises(SystemExit) as e:
        train_mod.main()
    assert e.value.code == train_mod.DIVERGED_EXIT_CODE
    d = json.loads((out / "diverged.json").read_text())
    assert 1 <= d["step"] < 12 and not (out / "summary.json").exists()


def test_sweep_records_diverged_runs_and_picks_best_of_the_rest(tmp_path, monkeypatch, capsys):
    sweep = _script("sweep_lr")
    runs, res = tmp_path / "runs", tmp_path / "results"
    lrs = [3e-4, 6e-4, 1.2e-3]
    losses = {"transformer": [5.9, 5.8, 5.85], "mamba": [5.6, 5.4, None]}  # None: diverged
    for arch, lr, out, _ in sweep.sweep_jobs("S", ["transformer", "mamba"], lrs, 1000, "D", runs):
        out.mkdir(parents=True)
        loss = losses[arch][lrs.index(lr)]
        if loss is None:
            (out / "diverged.json").write_text(json.dumps({"step": 19}))
        else:
            (out / "summary.json").write_text(json.dumps({"tokens": 1000, "best_val_loss": loss,
                                                          "train_seconds": 1.0}))
    monkeypatch.setattr("sys.argv", ["sweep_lr.py", "--data", "D", "--runs", str(runs), "--results", str(res),
                                     "--lrs", *map(str, lrs)])
    sweep.main()
    assert json.loads((res / "S" / "best_lr.json").read_text()) == {"transformer": 6e-4, "mamba": 6e-4,
                                                                     "mamba_lti": 6e-4}
    rows = (res / "S" / "lr_sweep.csv").read_text().splitlines()
    assert rows[0].startswith("scale,arch,lr,warmup_steps,status") and sum(",diverged," in r for r in rows) == 1
    out = capsys.readouterr().out
    assert "WARNING" not in out and "mamba: diverged (LR too high) at [0.0012]" in out


def test_tied_learning_rates_pick_the_lowest(tmp_path, capsys):
    """The real sweep v2 numbers (Kaggle, 2026-11-08): Mamba 1.2e-3 and 2.4e-3 differ by 0.0008."""
    sweep = _script("sweep_lr")
    losses = {"transformer": {1.5e-4: 6.021586, 3e-4: 5.643808, 6e-4: 5.413483, 1.2e-3: 5.619077, 2.4e-3: 5.744278},
              "mamba": {3e-4: 5.602395, 6e-4: 5.201107, 1.2e-3: 4.924240, 2.4e-3: 4.923466}}
    f = tmp_path / "lr_sweep.csv"
    lines = ["scale,arch,lr,warmup_steps,status,diverged_at_step,tokens,best_val_loss,train_seconds"]
    lines += [f"S,{arch},{lr},51,ok,,29884416,{loss},1.0" for arch, d in losses.items() for lr, loss in d.items()]
    f.write_text("\n".join(lines) + "\n")
    assert sweep.best_from_csv(f) == {"transformer": 6e-4, "mamba": 1.2e-3, "mamba_lti": 1.2e-3}
    out = capsys.readouterr().out
    assert "WARNING" not in out and "using the lowest, 0.0012" in out
    best, _ = sweep.choose_lrs([{"arch": "mamba", "lr": lr, "status": "ok", "best_val_loss": v}
                                for lr, v in ((1e-3, 5.0), (2e-3, 4.95))])
    assert best["mamba"] == 2e-3  # a real (> 0.01) improvement still wins


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
    assert t["transformer"]["total_tokens"] == 340_000_000
    targets = {"S_transformer": t["transformer"], "S_mamba": t["mamba"], "S_mamba_lti": t["mamba"],
               "M_transformer": t["transformer"], "M_mamba": t["mamba"]}
    est = rep.estimate(measured, targets)
    assert {e["target"] for e in est} == set(targets) and all(e["hours"] > 0 for e in est)
    plan = rep.stage_plan(est, gpus=2, session_hours=11)
    assert [p["runs"] for p in plan] == [9, 9, 2]
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
