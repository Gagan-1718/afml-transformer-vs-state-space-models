# Review 1 speaking notes

Aim for about five minutes. Use plain explanations and be clear that this is the first Transformer experiment.

| Slide | What to say |
| --- | --- |
| 1. Title | “We are comparing Transformer and Mamba-style small language models. Today we show our initial Transformer baseline.” |
| 2. Problem | “Both models predict the next token. We will compare prediction quality and efficiency using similar model sizes and the same training data.” |
| 3. Motivation | “Attention combines information from earlier tokens. Mamba updates a running state. We want to understand the trade-off experimentally.” |
| 4. Dataset | “For this pilot, we use only 200 source documents. We separate training and validation documents first and learn our tokenizer from training text.” |
| 5. Approach | “Text becomes token IDs. Targets are shifted by one position. The Transformer predicts the targets, and the optimizer updates its weights to reduce loss.” |
| 6. Results | Read the recorded values on the slide. “This is a short pilot. It checks that our pipeline learns. It does not establish a winner between architectures.” |
| 7. Progress | “The Transformer pipeline and demo are ready. Mamba code exists, but training and the controlled comparison are next.” |
| 8. Functionality | “These completion figures refer to the small pilot tasks. They are not percentages of the whole research project.” |
| 9. Learning | “We learned why causal masking and correct next-token targets matter, and why validation must be kept separate.” |
| 10. Challenges | “We still need matched runs, careful speed and memory measurements, and a larger body of evidence. Our current Mamba scan has implementation overhead.” |
| 11. References | “These papers provide the state-space background. FineWeb-Edu provides the training text.” |

## Short demo

Have the saved loss graph and checkpoint ready before the call. Show the model settings, the graph, then run:

```bash
python scripts/review1_demo.py generate --run-dir runs/review1 --prompt "The student learned that"
```

If using the delivered checkpoint archive, extract its contents into `runs/review1/`. It contains both `best.pt` and `tokenizer.json`. Use the same code branch. In Colab, the notebook saves a fresh checkpoint after training.

The model may produce rough text. Say: “This is a very small model trained briefly. The output demonstrates the generation path; fluent language is not established by this pilot.”

## Claims to avoid

- “Mamba is faster in our project.” We have not measured that comparison.
- “The whole model is finished.” Only the pilot Transformer experiment is complete.
- “Perplexity is accuracy.” It measures next-token prediction uncertainty.
- “We trained on ten billion tokens.” That is the dataset configuration name, not our sample size.
- “These validation scores are final test results.” Validation was used to choose the checkpoint.

## One-sentence project explanation

“We are testing whether a small model that updates an internal state can predict text as well as an attention-based model, while using less memory or running faster on longer inputs.”
