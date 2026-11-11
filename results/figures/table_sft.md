## Test split

| Model | Stage | Checkpoint | Response loss | Response ppl | ROUGE-L | Stops with EOS | Mean new tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | pretrained | S_transformer_s0 | 2.993 | 19.95 | 0.027 | 0% | 128.0 |
| Transformer | instruction-tuned | S_transformer_s0_lr0.0003 | 2.201 | 9.03 | 0.212 | 54% | 75.9 |
| Mamba (selective) | pretrained | S_mamba_s0 | 2.975 | 19.60 | 0.000 | 0% | 128.0 |
| Mamba (selective) | instruction-tuned | S_mamba_s0_lr0.0003 | 2.163 | 8.70 | 0.222 | 55% | 77.4 |

## Learning-rate choice (validation split)

| Model | Learning rate | Val response loss before | Best val response loss | Chosen |
| --- | --- | --- | --- | --- |
| Transformer | 3e-05 | 2.9848 | 2.3921 |  |
| Transformer | 0.0001 | 2.9848 | 2.2793 |  |
| Transformer | 0.0003 | 2.9848 | 2.2321 | yes |
| Transformer | 0.001 | 2.9848 | 2.3017 |  |
| Mamba (selective) | 3e-05 | 2.9634 | 2.3899 |  |
| Mamba (selective) | 0.0001 | 2.9634 | 2.2639 |  |
| Mamba (selective) | 0.0003 | 2.9634 | 2.1855 | yes |
| Mamba (selective) | 0.001 | 2.9634 | 2.1948 |  |
