# Reproducing the study

Everything runs on free hardware: tests on a laptop CPU, experiments on free Kaggle GPUs (Colab works as a
backup). Who ran which part, and with which code commit, is listed in
[results/README.md](../results/README.md#where-each-file-came-from).

## 1. Check the code on a laptop

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The tests check that every Mamba scan (`sequential`, `parallel`, `chunked`) gives the same outputs and
gradients as the step-by-step reference, and they run each script end to end on a tiny fake dataset. One
test needs a GPU with `mamba-ssm` installed and is skipped on a CPU.

## 2. Run the experiments on Kaggle

1. Kaggle → Settings → verify your phone number (needed for GPU and Internet).
2. New Notebook → File → Import Notebook → `notebooks/run_study.ipynb`.
3. Session options: *Accelerator* = **GPU T4 x2**, *Internet* = **On**. The `data` stage needs no GPU, so
   run it with *Accelerator* = **None**.
4. Edit the first code cell: set `STAGES`, for example `STAGES = {'data'}`.
5. **Save Version → Save & Run All**. The run continues with the browser closed. Its output, including
   `results.zip`, is saved with the version.
6. For the next stage, start a new notebook the same way and use **Add Input** to attach the `data`
   version's output and the outputs of the stages it needs. Finished runs, paused runs and results in the
   inputs are picked up automatically.

The notebook clones this repository, installs PyTorch 2.10 (CUDA 12.6) and a prebuilt `mamba-ssm` wheel,
runs the tests, then runs the chosen stages. If `mamba-ssm` cannot be installed, Mamba falls back to the
pure-PyTorch `chunked` scan, which gives the same results but is far slower (380 against 10,815 training
tokens per second in our smoke test).

### Stages

| Stage | What it does | Needs as input | Time we measured |
| --- | --- | --- | --- |
| `data` | streams FineWeb-Edu (pinned revision), removes duplicates, splits by hash, fits a 16K BPE on training documents, writes 2B training tokens; downloads 50 PG-19 test books | – | 1.1 h (CPU) |
| `smoke` | 20 training steps per model: speed, memory, time estimates | `data` | 0.3 h |
| `sweep` | learning-rate sweep at 30M tokens per point | `data` | about 2.5–3 h |
| `S` | the 8 main runs, 340M tokens each, two at a time (one per GPU) | `data`, `sweep` | Transformer 4.0–4.4 h, Mamba 8.2–8.5 h, Mamba-LTI 7.4–8.1 h per run |
| `eval` | test loss and recall split, PG-19 per-position loss, efficiency benchmark, Δ analysis | `data`, the finished `S` runs | – |
| `synthetic` | MQAR and selective-copying grids, 4,000 steps per point | – | 8.2 h |
| `synthetic_long` | selective copying, 40,000 steps, accuracy curves | – | 1.1 h |
| `mqar_long` | MQAR at length 128, learning rates 1e-3 and 3e-3, 40,000 steps | – | 5.4 h |
| `mqar_long_extra` | the same MQAR grid at learning rates 3e-4 and 1e-2 | – | 5.7 h |
| `sft` | instruction fine-tuning of the seed-0 Transformer and Mamba on Alpaca-cleaned | `data`, the seed-0 `S` runs | 1.9 h for three learning rates |
| `figures` | all figures and tables, and `results.zip` | the outputs above | minutes |

A free Kaggle session stops at 12 hours. Training saves `last.pt` and pauses at `SESSION_HOURS` (11 by
default), and the next session resumes from it.

### Splitting the work across several accounts

Each Kaggle account gets about 30 free GPU hours per week. We split the 8 main runs over the team's 4
accounts by setting `ONLY_RUNS` in the first code cell, two runs per account:

| Account | `ONLY_RUNS` |
| --- | --- |
| Gagandeep | `{'S_transformer_s0', 'S_mamba_s0'}` |
| Krishna | `{'S_transformer_s1', 'S_mamba_lti_s0'}` |
| Dikshith | `{'S_mamba_s1', 'S_mamba_lti_s1'}` |
| Mallikarjuna | `{'S_transformer_s2', 'S_mamba_s2'}` |

The `eval` notebook then adds all four outputs as inputs. A notebook must be public before other accounts
can add its output as an input.

## 3. Single commands

The notebook calls these. They also run on a CPU, slowly.

```bash
python -m afml.data.build --config configs/data.json --out DATA
python scripts/make_configs.py
python scripts/sweep_lr.py --help
python -m afml.train --config configs/S_mamba.json --data DATA --out runs/S_mamba_s0 \
    --set train.seed=0 --set train.lr=0.0012    # learning rate from results/S/best_lr.json
python -m afml.eval.perplexity --runs runs/S_* --data DATA --out results/S/perplexity.csv
python -m afml.eval.long_context --runs runs/S_*_s0 --data DATA --max-len 16384 --out results/S/long_context.csv
python -m afml.eval.bench --configs configs/S_transformer.json configs/S_mamba.json \
    --lengths 512 1024 2048 4096 8192 16384 32768 --batches 1 8 --out results/bench/bench.csv
python -m afml.eval.analysis --mamba runs/S_mamba_s0 --transformer runs/S_transformer_s0 --data DATA \
    --out results/analysis/S
python -m afml.eval.synthetic --task mqar --archs transformer mamba mamba_lti --lrs 1e-3 \
    --d-models 64 128 256 --seq-lens 128 --steps 40000 --eval-every 1000 \
    --out results/synthetic_long/mqar_lr0.001.csv --curve results/synthetic_long/mqar_lr0.001_curve.csv
python -m afml.sft --help
python scripts/make_figures.py --results results --out results/figures
```

## 4. Settings that matter for fairness

- Same tokenizer, data, context (1,024) and token count (339,869,696) for every model.
- Mamba's inner width is chosen so its non-embedding parameter count is within 0.5% of the Transformer's
  (the actual gaps are 0.22% for Mamba and 0.20% for Mamba-LTI).
- AdamW (β 0.9, 0.95), weight decay 0.1 on weight matrices only (not on norms, biases, embeddings, or
  Mamba's `A_log`, `D` and Δ bias), gradient clipping at 1.0, 51 warmup steps then cosine decay to 10%.
- Learning rate per model from the sweep, chosen on the validation split. When two learning rates are
  within 0.01 in loss, the lower one is used.
- The test split is scored once, after all choices are made.
