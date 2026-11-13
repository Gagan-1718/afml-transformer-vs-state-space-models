Time budget T = 4.16 h: the mean training time of the Transformer runs. Target loss 3.5500: the mean final validation loss of the Transformer runs. Hours are training time on one Tesla T4 in fp16, evaluation excluded; Mamba uses the mamba-ssm kernel (`cuda` scan).

| Run | Model | Micro-batch | Training hours | Final val loss | Last evaluation within T: hours / tokens / val loss | Hours to reach the target loss |
| --- | --- | --- | --- | --- | --- | --- |
| S_transformer_s0 | Transformer | 16 | 4.07 | 3.5555 | 4.07 h / 340M / 3.5555 | not reached |
| S_transformer_s1 | Transformer | 16 | 3.98 | 3.5452 | 3.98 h / 340M / 3.5452 | 3.87 |
| S_transformer_s2 | Transformer | 16 | 4.44 | 3.5494 | 4.16 h / 315M / 3.5617 | 4.43 |
| S_mamba_s0 | Mamba (selective) | 8 | 8.43 | 3.5040 | 3.90 h / 157M / 3.7704 | 6.78 |
| S_mamba_s1 | Mamba (selective) | 8 | 8.50 | 3.5154 | 4.12 h / 157M / 3.7924 | 7.25 |
| S_mamba_s2 | Mamba (selective) | 8 | 8.15 | 3.5083 | 3.79 h / 157M / 3.7800 | 6.70 |
| S_mamba_lti_s0 | Mamba-LTI (ablation) | 8 | 7.39 | 3.6831 | 3.53 h / 157M / 3.9317 | not reached |
| S_mamba_lti_s1 | Mamba-LTI (ablation) | 8 | 8.11 | 3.6923 | 4.08 h / 157M / 3.9460 | not reached |

Caveat: a run's loss partway through is not what a run planned for that shorter budget would reach, because its learning rate has not decayed yet. The middle column therefore understates Mamba at a 4-hour budget.
