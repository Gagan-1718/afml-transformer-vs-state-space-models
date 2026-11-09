"""Efficiency benchmark: CSV saved after every measurement, slow lengths skipped, kernel backend first."""
import csv
import json

from afml.eval import bench


def _configs(tmp_path):
    paths = []
    for arch, extra in (("transformer", {"n_heads": 2}), ("mamba", {})):
        p = tmp_path / f"S_{arch}.json"
        p.write_text(json.dumps({"model": {"arch": arch, "d_model": 32, "n_layers": 2, **extra}}))
        paths.append(str(p))
    return paths


def _run(tmp_path, monkeypatch, *extra):
    out = tmp_path / "bench" / "bench.csv"
    monkeypatch.setattr("sys.argv", ["bench", "--configs", *_configs(tmp_path), "--vocab", "64", "--lengths", "16",
                                     "32", "--batches", "1", "2", "--warmup", "1", "--repeats", "1",
                                     "--decode-tokens", "2", "--device", "cpu", "--out", str(out), *extra])
    bench.main()
    return out, list(csv.DictReader(open(out)))


def test_bench_measures_every_length_and_saves_csv(tmp_path, monkeypatch):
    out, rows = _run(tmp_path, monkeypatch)
    assert list(rows[0]) == bench.FIELDS and not out.with_suffix(".tmp").exists()
    # transformer + mamba (chunked only: no mamba-ssm kernel on CPU) x 2 lengths x 2 batches
    assert [(r["arch"], r["backend"]) for r in rows] == [("transformer", "-")] * 4 + [("mamba", "chunked")] * 4
    assert all(r["status"] == "ok" and float(r["prefill_ms"]) > 0 for r in rows)


def test_bench_skips_longer_lengths_once_too_slow(tmp_path, monkeypatch):
    _, rows = _run(tmp_path, monkeypatch, "--max-prefill-ms", "0")  # every prefill counts as too slow
    status = [(r["arch"], r["seq_len"], r["batch"], r["status"]) for r in rows]
    assert status == [(a, L, b, "ok" if L == "16" else "skipped_slow")
                      for a in ("transformer", "mamba") for L in ("16", "32") for b in ("1", "2")]


def test_bench_rows_saved_as_they_are_measured(tmp_path, monkeypatch):
    seen = []
    real = bench.write_rows
    monkeypatch.setattr(bench, "write_rows", lambda path, rows: (seen.append(len(rows)), real(path, rows)))
    _run(tmp_path, monkeypatch)
    assert seen[:8] == list(range(1, 9))  # one save per measurement, earlier models' rows kept


def test_kernel_backend_is_timed_first():
    assert bench.DEFAULT_BACKENDS == ["cuda", "chunked"]
