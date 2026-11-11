| Scale | Model | Params (total / non-emb) | Train tokens | Train FLOPs | Seeds | Test loss | Test ppl | Recall-hit loss | Other-token loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S | Transformer | 46.9M / 38.5M | 340M | 1.09e+17 | 3 | 3.5305 ± 0.0054 | 34.14 | 1.387 | 3.964 |
| S | Mamba (selective) | 46.9M / 38.5M | 340M | 9.89e+16 | 3 | 3.4841 ± 0.0053 | 32.59 | 1.762 | 3.832 |
| S | Mamba-LTI (ablation) | 47.0M / 38.6M | 340M | 9.95e+16 | 2 | 3.6605 ± 0.0060 | 38.88 | 2.416 | 3.912 |
