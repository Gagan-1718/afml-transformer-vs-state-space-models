# Efficiency benchmark files

- `bench.csv`: the benchmark used in the report (to be added from the re-run). Prefill is measured as
  generation runs it: the prompt fills the KV cache (Transformer) or the recurrent state (Mamba) and only
  the last position is scored. Prefill peak memory includes the weights and the cache; decode peak memory
  is measured after the prefill (weights, filled cache, one decoding step).
- `bench_v1_full_logits.csv`: the first run (Kaggle, 2026-11-09, commit 1d4b463), kept for the record.
  Its prefill was a full forward pass that also scored every position against the 16,384-token
  vocabulary. At long lengths that logits tensor dominated prefill memory and was the same size for both
  models, and it caused the Mamba out-of-memory point at 32,768 tokens x batch 8. Its decode latency and
  analytic cache size are not affected; it was replaced so the memory comparison measures the models.
