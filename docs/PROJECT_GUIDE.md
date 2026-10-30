# Review 1 project guide

## What we are building

We are studying two small language models. Both learn to predict the next token from text. The Transformer uses attention to combine information from earlier tokens. The Mamba-style model updates an internal state as it reads the sequence.

Our final question is whether the Mamba-style model can approach the Transformer's prediction quality while improving speed or memory use at longer sequence lengths. We have not answered that question yet.

For Review 1, we train only a small Transformer. This checks that our data, training, validation and generation pipeline works. The Mamba implementation already exists, but its dataset training and comparison are later work.

## What the demo does

```text
200 FineWeb-Edu documents
           |
Remove exact duplicates and split whole documents
           |
Fit a small tokenizer using training documents only
           |
Create 128-token inputs and shifted targets
           |
Train the Transformer for 300 updates
           |
Evaluate held-out validation documents every 50 updates
           |
Save the best checkpoint, plot losses and generate examples
```

The sample comes from the start of the `sample-10BT` dataset configuration. We do not download or train on ten billion tokens. This ordered sample is convenient for a demo, but it is not a representative random sample of all FineWeb-Edu.

## Five ideas to understand first

**Token:** a piece of text represented by an integer. Our byte-level BPE tokenizer learns frequent combinations from the training text. A token is not necessarily a whole word.

**Embedding:** a learned vector for each token ID. It turns integers into vectors that the model can process.

**Next-token target:** the correct token immediately after an input token. If the token sequence is `[10, 20, 30, 40]`, the input is `[10, 20, 30]` and the target is `[20, 30, 40]`.

**Loss:** a measure of how poorly the model predicts the correct next tokens. Cross-entropy penalizes the model when it assigns low probability to the correct answer.

**Training:** calculate predictions and loss, calculate gradients, then update weights with an optimizer. Repeat with more batches.

## Files to read in order

| File | What it does |
| --- | --- |
| `configs/review1.json` | Small demo settings and dataset selection |
| `afml/data.py` | Download a bounded sample, split documents, fit tokenizer, build batches |
| `afml/config.py` | Existing model architecture settings |
| `afml/models/transformer.py` | Existing attention and feed-forward layers |
| `afml/models/lm.py` | Existing shared language-model wrapper and loss |
| `afml/demo.py` | New training, validation, plotting and generation functions |
| `scripts/review1_demo.py` | Command-line entry point |
| `notebooks/review1_transformer_demo.ipynb` | Guided Colab walkthrough |
| `tests/test_models.py` | Existing model correctness checks |
| `tests/test_demo.py` | New data, evaluation and generation checks |
| `results/review1/` | Recorded pilot evidence, once the local run completes |

The existing model implementations are reused. We have not replaced your architectures or added Mamba training.

## How the Transformer works

1. **Embedding:** look up a vector for each token.
2. **Causal attention:** each position can use its current and earlier input tokens. It cannot see later tokens. Four heads provide four learned attention patterns.
3. **Position information:** rotary positional embeddings help attention distinguish token positions.
4. **Feed-forward network:** SwiGLU transforms each position's representation.
5. **Normalization and residual connections:** RMSNorm controls scale, while residual connections add a layer's output back to its input.
6. **Output head:** turn each final vector into a score for every possible next token. This head shares weights with the embedding table.

The demo uses two layers and width 128. These are deliberately much smaller settings than the default model configuration. A 128-token context means the model receives up to 128 tokens at a time.

## How data leakage is avoided

We remove exact duplicate documents, shuffle with seed 42 and hold out 15% of documents. Only then do we tokenize and create windows. The tokenizer learns its vocabulary from the training documents only.

Each window stays inside one document. The target is shifted by one token. Short tails that cannot make a full window are discarded and counted. The manifest records document hashes, IDs, dataset revision, token counts and tokenizer hash.

This prevents exact documents from appearing in both splits. It does not detect near-duplicate pages or guarantee that the small sample represents the whole dataset.

## How training works

The training script samples batches of eight windows. For each batch it:

1. Computes predictions and cross-entropy loss.
2. Clears old gradients with `zero_grad`.
3. Runs `backward` to compute new gradients.
4. Clips the gradient norm to reduce extreme updates.
5. Calls the AdamW optimizer to update weights.

At step zero and every 50 steps, it measures loss on a fixed training subset and the full prepared validation set. This makes the plotted points more comparable than plotting unrelated training batches.

Three hundred updates expose the model to 307,200 target-token positions: `300 × 8 × 128`. Some windows can be sampled repeatedly, so this is not the count of unique training tokens.

The script checks that losses and gradients stay finite. It saves the checkpoint with the lowest observed validation loss. It deliberately refuses to overwrite an existing run directory.

## How to interpret the measurements

**Training loss:** prediction error on the fixed training subset. Lower values suggest that the model is learning that data.

**Validation loss:** prediction error on documents excluded from training. This is more useful than training loss for checking generalization.

**Perplexity:** `exp(mean next-token loss)`. Lower is better under the same tokenizer and scoring procedure. Do not compare this small tokenizer's perplexity directly against published models with different tokenizers.

**Best checkpoint:** selected using validation loss. Therefore the reported validation result is not an untouched final test result. Later experiments should reserve a separate test set.

The recorded duration includes evaluation and checkpoint writing. It is not an inference-speed benchmark. No efficiency advantage over Mamba is claimed.

## How generation works

We tokenize a prompt, predict the next token, sample it, append it, and repeat. Sampling uses temperature 0.8 and the top 40 candidate tokens. Generation stops at the end-of-document token or after 60 new tokens.

When the text exceeds the 128-token context, generation uses only the most recent 128 tokens. It recomputes the full context on each step and does not use a KV cache. This is sufficient for demonstrating continuation, but unsuitable for claiming optimized decoding speed.

Short training on a tiny corpus may produce repetitive or incoherent text. Show the actual output and explain the limitation. This model has not been instruction-tuned and is not expected to answer questions like a chatbot.

## How to run

Use Python 3.12 if possible. A CPU can run this tiny configuration; a CUDA-enabled PyTorch installation can use a supported NVIDIA GPU. The Colab notebook installs dependencies and runs the steps in order.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/review1_demo.py prepare
python scripts/review1_demo.py train
python scripts/review1_demo.py generate --prompt "The student learned that"
```

Run commands from the repository root. Use `--device cpu` if CUDA is unavailable. Use new `--data-dir` and `--run-dir` paths for a new experiment. Keep the tokenizer with its checkpoint; using another tokenizer changes the meaning of token IDs.

The preparation step needs internet access. By default it requests bounded pages from the Hugging Face dataset rows API and verifies the revision header on every page. Set the data loader to `stream` to use the standard Datasets streaming path instead. It saves a local tokenized dataset, after which training can run offline. Never substitute another dataset silently if FineWeb-Edu cannot load.

## What to show during the review

1. Explain the project question in one sentence.
2. Explain that this review covers the initial Transformer baseline.
3. Show the model settings and parameter count.
4. Show one input/target pair and explain the one-token shift.
5. Show the saved loss graph and validation metrics.
6. Generate a short continuation from the saved checkpoint.
7. Explain that Mamba training, matched budgets and efficiency measurements come next.

Run the demo once before presenting. Use saved results if the network fails during the review. Retraining live is unnecessary.

## Likely questions

**Why start with the Transformer?** It establishes the data and evaluation pipeline that we will reuse for the comparison.

**What is new here?** The course study investigates an architectural comparison at a controlled small scale. We are not claiming to invent either architecture.

**Is Mamba better?** We have not measured that yet.

**Why is generated text weak?** The model, dataset and training budget are deliberately small. This run demonstrates the pipeline.

**How will the later comparison be fair?** Use the same tokenizer, documents, token budget, sequence lengths and evaluation procedure, with closely matched parameter counts and reported settings.

**Is the current Mamba implementation optimized?** No. Its PyTorch scan materializes large intermediate tensors. Later efficiency claims must account for that implementation limitation.

**What remains?** Mamba dataset training, controlled comparisons, reliable latency/memory/throughput measurements, optional PG-19 evaluation, and final reporting.

## Sources and scope

- Project scope: submitted Team 7 one-pager, “Training and Evaluating a Mamba-Style SLM.”
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Streaming: https://huggingface.co/docs/datasets/en/stream
- Mamba: https://arxiv.org/abs/2312.00752
- S4: https://arxiv.org/abs/2111.00396
- Mamba-2: https://arxiv.org/abs/2405.21060

Dolly instruction tuning is outside the submitted one-pager's current scope. PG-19 is optional. The Review 1 pilot is not the completed course project.
