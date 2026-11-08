# Transformer vs selective state-space (Mamba) language models

Team Abcd — AFML course project (UE24AM342AA1)

**Question.** At the same parameter count, training data and token budget, can a Mamba-style
selective state-space model match a Transformer's language-modelling quality, and where does it
win or lose: memory recall, long context, inference speed and memory?

## What is in the study

| ID | Experiment | Script |
| --- | --- | --- |
| E1 | Transformer vs Mamba vs Mamba-LTI at ~47M (3 seeds) and ~98M parameters, ~20 tokens/parameter | `afml/train.py` |
| E2 | Selectivity ablation: Mamba vs time-invariant Mamba (LM loss + selective copying) | `configs/*_mamba_lti.json`, `afml/eval/synthetic.py` |
| E3 | Associative recall: MQAR grid, and test loss split into recall-hit vs other tokens | `afml/eval/synthetic.py`, `afml/eval/perplexity.py` |
| E4 | Length extrapolation: per-position loss on PG-19 books up to 16K tokens (trained at 1K) | `afml/eval/long_context.py` |
| E5 | Inference efficiency: prefill/decode latency, peak memory vs length, cached decoding | `afml/eval/bench.py` |
| E6 | Mechanism: Mamba's step size Δ per token, memory horizon, attention maps | `afml/eval/analysis.py` |

Model sizes (vocabulary 16,384; Mamba uses 2 layers per Transformer layer, `d_inner` matched by
`scripts/make_configs.py`):

| Scale | Width | Transformer | Mamba | Non-embedding params | Total params | Mismatch | Training tokens | Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S | 512 | 12 layers | 24 layers, d_inner 968 | 38.5M | ~47M | 0.22% | 940M | main study, 3 seeds |
| M | 768 | 12 layers | 24 layers, d_inner 1440 | 85.0M | ~98M | 0.11% | 1.95B | scale check, 1 seed |

The time-invariant ablation (Mamba-LTI, S only) uses d_inner 1032 (mismatch 0.20%).

## Code map

```text
afml/
  config.py              model settings
  models/transformer.py  pre-norm Transformer: RoPE, SwiGLU, RMSNorm, KV cache
  models/mamba.py        Mamba-1 block; scans: sequential (reference), parallel, chunked (default), cuda (optional)
  models/lm.py           shared LM wrapper, cached generation, weight-decay groups, FLOP estimate
  data/build.py          FineWeb-Edu + PG-19 -> 16K BPE -> uint16 token files (hash-based train/val/test split)
  data/loader.py         memory-mapped batches
  train.py               trainer: warmup + cosine, AdamW, mixed precision, grad accumulation, resume, auto batch size
  eval/                  perplexity, long_context, synthetic, bench, analysis
configs/                 data.json and parameter-matched run configs
scripts/                 make_configs, sweep_lr, smoke_report, collect_results, make_figures
notebooks/run_study.ipynb   runs everything on a free Kaggle or Colab GPU
tests/                   correctness tests (run these first)
report/report.md         course report draft; result slots marked TODO(results) are filled from results/
docs/                    TEAMMATE_GUIDE.md (one Kaggle run per teammate), COORDINATOR_CHECKLIST.md
results/pilot/           the earlier 0.56M-parameter Review 1 pilot, kept for history
```

### Why the chunked scan matters

The original parallel scan keeps O(log L) tensors of size batch × length × d_inner × d_state for
backpropagation. On an earlier ~17M-parameter Mamba (width 384, 12 layers) it used more than 7 GB for
**two** sequences of 1,024 tokens and was killed. The chunked scan runs the same recurrence chunk by
chunk under activation checkpointing; that same ~17M model then trained a 2 × 1,024 batch in 2.4 GB
on CPU. These numbers come from that earlier model, not from the current S and M sizes. Tests check
that all scans agree with the step-by-step reference in outputs **and gradients**.

## Run it

**On your laptop (CPU): check everything works**

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

**On a free GPU: the real experiments.** Open `notebooks/run_study.ipynb` on Kaggle (recommended:
about 30 free GPU-hours per week after phone verification) or Colab. Set `STAGES` in the first code
cell, then run. Every stage is resumable.

1. `data` — build the token files once: 2B training tokens (2–3 h).
2. `smoke` — 20-step GPU check: speed, memory and time estimates for full runs.
3. `sweep` — learning-rate sweep at scale S.
4. `S` — 3 models × 3 seeds.
5. `synthetic` — MQAR and selective-copying grids.
6. `eval` — test perplexity, PG-19, efficiency, Δ analysis (run again after `M`).
7. `M` — 2 models × 1 seed.
8. `figures` — all figures, `table_main.md`, and `results.zip`.

On Kaggle: run each stage with *Save Version → Save & Run All* (Internet on, GPU **T4 x2**). Next
session, *Add Input* → the data version's output **and** the latest version's output, so data,
finished runs and paused runs are reused.

- **Both T4 GPUs are used:** independent runs train two at a time, one per GPU.
- **No lost work at the 12-hour limit:** training pauses cleanly at `SESSION_HOURS` (default 11) by
  saving `last.pt`, and resumes in the next session.
- **The `data` stage needs no GPU:** run it in a CPU session to save GPU quota.
- **Small outputs:** finished runs keep only `best.pt`; finished sweep and smoke runs keep no weights.
- **Splitting work across team accounts:** set `ONLY_RUNS`, then add each other's outputs as inputs.
  Step-by-step: [docs/TEAMMATE_GUIDE.md](docs/TEAMMATE_GUIDE.md) and
  [docs/COORDINATOR_CHECKLIST.md](docs/COORDINATOR_CHECKLIST.md).
- **The `smoke` stage measures real speed first:** `scripts/smoke_report.py` turns 20-step runs into
  tokens/sec, micro-batch and peak memory, plus planning estimates of hours per run and per stage.

Single commands, for reference:

```bash
python -m afml.data.build --config configs/data.json --out DATA
python scripts/make_configs.py
python -m afml.train --config configs/S_mamba.json --data DATA --out runs/S_mamba_s0 --set train.seed=0
python -m afml.eval.perplexity --runs runs/S_* --data DATA --out results/S/perplexity.csv
python scripts/make_figures.py --results results --out results/figures
```

## Honesty rules for results

- Validation picks checkpoints and learning rates; the **test** split is scored once at the end.
- Perplexities compare only models sharing this tokenizer and test set.
- Speed claims name the scan backend: pure-PyTorch Mamba is slower than the fused kernel, which is
  an implementation effect, not an architecture effect.
- Results at 47–98M parameters (~20 tokens/parameter) may not hold at billions of parameters.

## References

- Gu & Dao (2023), [Mamba](https://arxiv.org/abs/2312.00752)
- Gu, Goel & Ré (2021), [S4](https://arxiv.org/abs/2111.00396)
- Dao & Gu (2024), [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060)
- Arora et al. (2023), [Zoology / MQAR](https://arxiv.org/abs/2312.04927)
- Hoffmann et al. (2022), [Chinchilla](https://arxiv.org/abs/2203.15556)
- Data: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [PG-19](https://huggingface.co/datasets/deepmind/pg19) (Parquet copy: `emozilla/pg19`)
