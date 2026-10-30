# Transformer versus selective state-space language models

Team Abcd — AFML course project

We compare small Transformer and Mamba-style language models using matched data and training budgets. The final study will measure prediction quality and inference efficiency as sequence length increases.

## Review 1 scope

This branch prepares **one small Transformer training demo**. Mamba training and the full architectural comparison are later work. The existing model implementations are reused.

- Bounded FineWeb-Edu sample with separate training and validation documents.
- Shared byte-level BPE tokenizer, trained only on the training split.
- Small Transformer, training loop, validation, checkpoint saving and text generation.
- Loss plot, measured pilot results and course-template presentation.

## Start here

[Open the guided notebook in Colab](https://colab.research.google.com/github/Gagan-1718/afml-transformer-vs-state-space-models/blob/review1/transformer-demo/notebooks/review1_transformer_demo.ipynb)

Read [the project and code guide](docs/PROJECT_GUIDE.md) for a simple explanation of every stage, then [the presentation notes](docs/PRESENTATION_NOTES.md) to rehearse.

## Run locally

Use Python 3.12 and an isolated environment. Install the PyTorch build appropriate for your machine; CPU is sufficient for the small demo. The notebook uses the available Colab runtime.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/review1_demo.py prepare
python scripts/review1_demo.py train --device auto
python scripts/review1_demo.py generate --prompt "The student learned that"
```

Run commands from the repository root. Preparation downloads a bounded sample through the dataset rows API, not the entire dataset. Training uses local prepared data. Existing non-empty data/run directories are protected; pass a fresh `--data-dir` or `--run-dir` when repeating an experiment.

Settings are in [`configs/review1.json`](configs/review1.json): 200 source documents, 15% validation, 1,024-token vocabulary, 128-token context, two layers, width 128, four heads, and 300 optimizer updates.

## Outputs

| Location | Contents |
| --- | --- |
| `data/review1/` | Tokenizer, token windows and source/split manifest; ignored by Git |
| `runs/review1/` | Best checkpoint, tokenizer, metrics CSV, summary, loss graph and sample text; ignored by Git |
| `results/review1/` | Committed evidence from the recorded pilot, without raw dataset text or model weights |
| `slides/` | Review 1 PowerPoint and readable slide notes |

The saved weights and tokenizer must stay together. Checkpoints are excluded from Git; train with the notebook or use the separately delivered demo archive.

## What the results mean

This is a tiny, single-seed pipeline experiment. Validation selects the checkpoint and is not an untouched final test set. Perplexity is comparable only with the same tokenizer and evaluation procedure. Generated text may be repetitive or incoherent. Mamba has not been trained on this dataset yet, so no architecture wins are claimed.

The existing Mamba module is a plain PyTorch selective-state implementation with sequential and parallel scans and a recurrent step. Its parallel scan materializes large tensors. It should not be treated as the optimized official Mamba kernel in later efficiency claims.

## Dataset and references

- [FineWeb-Edu dataset card and license information](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [S4](https://arxiv.org/abs/2111.00396)
- [Mamba-2](https://arxiv.org/abs/2405.21060)

The committed data manifest records the source revision and document hashes. Raw web documents are not redistributed in this repository. PG-19 remains optional; Dolly instruction tuning is outside the current submitted scope.
