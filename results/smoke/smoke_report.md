## Measured in the smoke runs (steady state)

| run | scan_backend | device | precision | tokens_per_sec | micro_batch | grad_accum | peak_memory_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_S_transformer | - | Tesla T4 | torch.float16 | 10,370 | 16 | 8 | 9.48 |
| smoke_S_mamba_cuda | cuda | Tesla T4 | torch.float16 | 10,815 | 8 | 16 | 8.28 |
| smoke_S_mamba_chunked | chunked | Tesla T4 | torch.float16 | 380 | 4 | 4 | 10.16 |

## Estimated hours per full run (planning only, not results)

| target | scan_backend | how | tokens_per_sec | train_hours | eval_hours | hours |
| --- | --- | --- | --- | --- | --- | --- |
| S_transformer | - | measured | 10,370 | 25.2 | 0.08 | 25.3 |
| S_mamba | cuda | measured | 10,815 | 24.1 | 0.19 | 24.3 |
| S_mamba | chunked | measured | 380 | 686.5 | 4.48 | 690.9 |
| S_mamba_lti | cuda | scaled by FLOPs/token | 10,754 | 24.3 | 0.19 | 24.5 |
| S_mamba_lti | chunked | scaled by FLOPs/token | 378 | 690.4 | 4.51 | 694.9 |
| M_transformer | - | scaled by FLOPs/token | 5,160 | 105.0 | 0.33 | 105.3 |
| M_mamba | cuda | scaled by FLOPs/token | 5,251 | 103.2 | 0.77 | 103.9 |
| M_mamba | chunked | scaled by FLOPs/token | 185 | 2932.8 | 18.47 | 2951.3 |
| M_mamba_lti | cuda | scaled by FLOPs/token | 5,241 | 103.3 | 0.77 | 104.1 |
| M_mamba_lti | chunked | scaled by FLOPs/token | 184 | 2938.2 | 18.50 | 2956.7 |

## Estimated stage budget (2 GPU(s) in parallel, 11 usable hours per session)

| stage | mamba_backend | runs | gpu_hours | longest_run_hours | wall_hours | kaggle_sessions |
| --- | --- | --- | --- | --- | --- | --- |
| sweep (S, 2 models x 4 LRs) | chunked | 8 | 307.0 | 74.1 | 153.5 | 14 |
| S (3 models x 3 seeds) | chunked | 9 | 4233.2 | 694.9 | 2116.6 | 193 |
| M (transformer + mamba, 1 seed) | chunked | 2 | 3056.6 | 2951.3 | 2951.3 | 269 |
| sweep (S, 2 models x 4 LRs) | cuda | 8 | 21.2 | 2.7 | 10.6 | 1 |
| S (3 models x 3 seeds) | cuda | 9 | 222.2 | 25.3 | 111.1 | 11 |
| M (transformer + mamba, 1 seed) | cuda | 2 | 209.2 | 105.3 | 105.3 | 10 |
