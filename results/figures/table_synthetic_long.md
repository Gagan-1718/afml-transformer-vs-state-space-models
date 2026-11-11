## Selective copying (40,000-step runs, one seed)

| Length | Width | Model | Learning rate | Best accuracy | First step at >= 99% | Steps run |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 64 | Transformer | 0.001 | 99.6% | 31,000 | 32,000 |
| 64 | 64 | Transformer | 0.003 | 99.7% | 10,000 | 11,000 |
| 64 | 64 | Mamba (selective) | 0.001 | 99.7% | 15,000 | 16,000 |
| 64 | 64 | Mamba (selective) | 0.003 | 99.7% | 9,000 | 9,000 |
| 64 | 64 | Mamba-LTI (ablation) | 0.001 | 54.2% | not reached | 40,000 |
| 64 | 64 | Mamba-LTI (ablation) | 0.003 | 99.5% | 29,000 | 31,000 |
| 256 | 64 | Transformer | 0.001 | 91.9% | not reached | 40,000 |
| 256 | 64 | Transformer | 0.003 | 93.3% | not reached | 40,000 |
| 256 | 64 | Mamba (selective) | 0.001 | 99.6% | 26,000 | 26,000 |
| 256 | 64 | Mamba (selective) | 0.003 | 99.6% | 13,000 | 13,000 |
| 256 | 64 | Mamba-LTI (ablation) | 0.001 | 12.9% | not reached | 40,000 |
| 256 | 64 | Mamba-LTI (ablation) | 0.003 | 99.7% | 28,000 | 32,000 |

## MQAR (40,000-step runs, one seed)

| Length | Width | Model | Learning rate | Best accuracy | First step at >= 99% | Steps run |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 64 | Transformer | 0.0003 | 17.8% | not reached | 40,000 |
| 128 | 64 | Transformer | 0.001 | 18.1% | not reached | 40,000 |
| 128 | 64 | Transformer | 0.003 | 3.1% | not reached | 40,000 |
| 128 | 64 | Transformer | 0.01 | 18.1% | not reached | 40,000 |
| 128 | 64 | Mamba (selective) | 0.0003 | 9.9% | not reached | 40,000 |
| 128 | 64 | Mamba (selective) | 0.001 | 18.6% | not reached | 40,000 |
| 128 | 64 | Mamba (selective) | 0.003 | 19.6% | not reached | 40,000 |
| 128 | 64 | Mamba (selective) | 0.01 | 16.0% | not reached | 40,000 |
| 128 | 64 | Mamba-LTI (ablation) | 0.0003 | 0.2% | not reached | 40,000 |
| 128 | 64 | Mamba-LTI (ablation) | 0.001 | 15.5% | not reached | 40,000 |
| 128 | 64 | Mamba-LTI (ablation) | 0.003 | 19.2% | not reached | 40,000 |
| 128 | 64 | Mamba-LTI (ablation) | 0.01 | 18.2% | not reached | 40,000 |
| 128 | 128 | Transformer | 0.0003 | 20.6% | not reached | 40,000 |
| 128 | 128 | Transformer | 0.001 | 99.5% | 24,000 | 24,000 |
| 128 | 128 | Transformer | 0.003 | 5.7% | not reached | 40,000 |
| 128 | 128 | Transformer | 0.01 | 5.4% | not reached | 40,000 |
| 128 | 128 | Mamba (selective) | 0.0003 | 19.7% | not reached | 40,000 |
| 128 | 128 | Mamba (selective) | 0.001 | 19.9% | not reached | 40,000 |
| 128 | 128 | Mamba (selective) | 0.003 | 21.3% | not reached | 40,000 |
| 128 | 128 | Mamba (selective) | 0.01 | 99.6% | 29,000 | 33,000 |
| 128 | 128 | Mamba-LTI (ablation) | 0.0003 | 2.9% | not reached | 40,000 |
| 128 | 128 | Mamba-LTI (ablation) | 0.001 | 15.3% | not reached | 40,000 |
| 128 | 128 | Mamba-LTI (ablation) | 0.003 | 16.6% | not reached | 40,000 |
| 128 | 128 | Mamba-LTI (ablation) | 0.01 | 16.9% | not reached | 40,000 |
| 128 | 256 | Transformer | 0.0003 | 21.9% | not reached | 40,000 |
| 128 | 256 | Transformer | 0.001 | 21.9% | not reached | 40,000 |
| 128 | 256 | Transformer | 0.003 | 21.8% | not reached | 40,000 |
| 128 | 256 | Transformer | 0.01 | 99.6% | 17,000 | 18,000 |
| 128 | 256 | Mamba (selective) | 0.0003 | 21.1% | not reached | 40,000 |
| 128 | 256 | Mamba (selective) | 0.001 | 21.1% | not reached | 40,000 |
| 128 | 256 | Mamba (selective) | 0.003 | 99.5% | 18,000 | 22,000 |
| 128 | 256 | Mamba (selective) | 0.01 | 20.2% | not reached | 40,000 |
| 128 | 256 | Mamba-LTI (ablation) | 0.0003 | 12.3% | not reached | 40,000 |
| 128 | 256 | Mamba-LTI (ablation) | 0.001 | 18.3% | not reached | 40,000 |
| 128 | 256 | Mamba-LTI (ablation) | 0.003 | 20.6% | not reached | 40,000 |
| 128 | 256 | Mamba-LTI (ablation) | 0.01 | 20.4% | not reached | 40,000 |
