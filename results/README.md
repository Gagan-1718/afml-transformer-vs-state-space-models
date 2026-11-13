# Results

Every number in the paper, slides and main README comes from a file in this folder. The files were
downloaded from the Kaggle runs (`results.zip` of each notebook version) and copied here unchanged.
Model weights and the raw analysis arrays (`analysis/S/*.npz`, 35 MB) are not included.

To remake the figures and tables from these files:

```bash
python scripts/make_figures.py --results results --out results/figures
```

## Which file backs which experiment

| Experiment | Files | Figure / table |
| --- | --- | --- |
| Data | `data/meta.json` (FineWeb-Edu revision, split rule, token counts), `data/pg19_meta.json` (the 50 PG-19 books) | – |
| Speed check before training | `smoke/smoke_report.{md,json}` (tokens/s, memory, planning estimates) | – |
| Learning-rate sweep | `S/lr_sweep.csv`, `S/best_lr.json`; `S/sweep_v1_failed_log.txt` is the first attempt (see notes) | – |
| E1 quality | `S/runs/<run>/{config.json, metrics.csv, summary.json}` for all 8 runs; `S/perplexity.csv` (test loss) | `fig1_training_curves.png`, `table_main.md` |
| E1 by training time | `S/runs/<run>/metrics.csv` (`elapsed_s` = training seconds, evaluation excluded) and `summary.json` | `fig9_time_budget.png`, `table_time_budget.md` |
| E2 selectivity | `S/perplexity.csv` (Mamba-LTI rows); `synthetic_long/selective_copy_lr*.csv` and `*_curve.csv` | `fig2_selective_copy.png`, `table_synthetic_long.md` |
| E3 recall | `S/perplexity.csv` (`hit_loss`, `other_loss`); `synthetic_long/mqar_lr*.csv` and `*_curve.csv` | `fig3_mqar.png`, `fig4_recall_split.png`, `table_synthetic_long.md` |
| E4 long context | `S/long_context.csv` (seed-0 models, 50 PG-19 books, log-spaced position bins) | `fig5_long_context.png` |
| E5 efficiency | `bench/bench.csv` (one T4, fp16, batch 1 and 8, 512 to 32,768 tokens) | `fig6_efficiency.png` |
| E6 mechanism | `analysis/S/delta_stats.json`; `analysis/S/*.png` are the raw plots made on Kaggle | `fig7_memory_horizon.png` |
| E7 fine-tuning | `sft/test.csv`, `sft/lr_sweep.csv`, `sft/best.json`, `sft/runs/*`, `sft/data_meta.json`; answers in `sft/samples.md` and `sft/generations.jsonl` | `fig8_sft.png`, `table_sft.md` |
| Short synthetic runs | `synthetic/{mqar,selective_copy}.{csv,log}` (4,000 steps per point) | `figA1_synthetic_4000_steps.png`, `figA2_mqar_width_4000_steps.png` |

Run names: `S_<model>_s<seed>`, where `S` is the 47M scale and `<model>` is `transformer`, `mamba` or
`mamba_lti`. In the synthetic files, `lr` is the learning rate and `accuracy` is the best held-out accuracy
during the run; runs stop early at 99.5%.

## Notes for reading the files

- **Perplexity in `table_main.md`** is exp(mean test loss over seeds).
- **Recall-hit tokens** are positions whose (current token, next token) pair already appeared earlier in
  the same 1,024-token test window. They are 841,199 of the 4,999,168 scored test tokens (16.8%).
- **Layers in `analysis/S/`** are numbered from 0. `eos_delta_ratio` is `null` because the 256 analysed
  test tokens contain no end-of-text token.
- **MQAR plateau.** Most MQAR runs that fail level off near 0.21. Each key is asked once, and its value
  appears right after the question. A model that guesses uniformly among the values not yet asked would
  score (1 + 1/2 + … + 1/16) / 16 = 0.211. This is our reading of the plateau, not something we tested
  directly.
- **The 4,000-step synthetic runs (`synthetic/`) were too short.** The 40,000-step runs in
  `synthetic_long/` replace them in the paper. We keep the short runs because they show how much the
  conclusion depends on training length.
- **`bench/bench.csv` is the third benchmark run and the only one used.** The earlier two are kept as a
  record. `bench_v1_full_logits.csv` measured prefill as a full forward pass that scored every position, so
  its prefill memory was dominated by the output layer (see `bench/README.md`).
  `bench_v2_cache_views.csv` kept views into the scan buffers in Mamba's cache, so Mamba's decoding memory
  grew with length. Both problems were fixed before the third run.
- **`S/sweep_v1_failed_log.txt`.** The first sweep used a 4-step warmup and Mamba diverged (NaN) at the two
  highest learning rates. The second sweep used the main runs' 51 warmup steps and is the one in
  `S/lr_sweep.csv`. For Mamba, 1.2e-3 (4.9242) and 2.4e-3 (4.9235) tied within 0.01, and the rule picks the
  lower one.
- **Fine-tuning learning rates.** The first fine-tuning run tried 3e-5, 1e-4 and 3e-4; both models were best
  at 3e-4, the edge of the grid. A second run added 1e-3, which was worse for both, so 3e-4 stayed chosen.
  The test split was scored once, for the chosen checkpoints. The pretrained Mamba's ROUGE-L is 0 because
  with greedy decoding it repeats the `###` of the prompt template (see `sft/samples.md`).

## Where each file came from

| Folder | Kaggle stage | Run by | Code commit |
| --- | --- | --- | --- |
| `data/`, `smoke/`, `S/lr_sweep.csv`, `S/best_lr.json` | `data`, `smoke`, `sweep` | Gagandeep | – |
| `S/runs/S_transformer_s0`, `S/runs/S_mamba_s0` | `S` | Gagandeep | – |
| `S/runs/S_transformer_s1`, `S/runs/S_mamba_lti_s0` | `S` | Krishna | – |
| `S/runs/S_mamba_s1`, `S/runs/S_mamba_lti_s1` | `S` | Dikshith | – |
| `S/runs/S_transformer_s2`, `S/runs/S_mamba_s2` | `S` | Mallikarjuna | – |
| `synthetic/` | `synthetic` | Gagandeep | – |
| `bench/bench.csv` | `eval` (benchmark part) | Dikshith | `094a2f5` |
| `S/perplexity.csv`, `S/long_context.csv`, `analysis/` | `eval` | Krishna | – |
| `sft/` | `sft` | Mallikarjuna | `c5bfd10`, then `2c295fb` for learning rate 1e-3 |
| `synthetic_long/selective_copy_*` | `synthetic_long` | Dikshith | `eea26f4` |
| `synthetic_long/mqar_lr0.001*`, `mqar_lr0.003*` | `mqar_long` | Mallikarjuna | `a45c0b5` |
| `synthetic_long/mqar_lr0.0003*`, `mqar_lr0.01*` | `mqar_long_extra` | Dikshith | `234c96a` |
