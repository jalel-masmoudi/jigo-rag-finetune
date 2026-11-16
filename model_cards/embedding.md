---
language: en
license: mit
base_model: BAAI/bge-small-en-v1.5
tags:
  - sentence-transformers
  - feature-extraction
---

# bge-small fine-tuned on the Jigo handbook corpus

Fine-tune of `BAAI/bge-small-en-v1.5` with MultipleNegativesRankingLoss.
The base model is MIT licensed. The numbers are from `results/embed.json`
on `data/test_reviewed.jsonl` (350 questions, 1,000 bootstrap resamples, seed 42).
That file is the output of `auto_filter_test.py`. It is not hand-reviewed.

## Training

- Loss: MultipleNegativesRankingLoss (in-batch negatives plus three hard negatives)
- Hard negatives: top hits from the baseline FAISS retriever, excluding the gold chunk and the rest of its document
- Grid: learning rate 2e-5 or 5e-5, epochs 1, 2, or 3
- Selection: highest validation Recall@5. The test set is scored once, after selection.
- Seed: 42

## Intended use

Retrieve passages from the GSA TTS Handbook and the Jigo AI Workspace markdown docs. Queries should use the prefix `Represent this sentence for searching relevant passages: `. Passages should not.

## Metrics

| | Recall@5 | 95% interval | n |
| --- | --- | --- | --- |
| Filtered test set | 0.851 | 0.814–0.886 | 350 |
| MRR@10 on the same set | 0.664 | 0.623–0.701 | 350 |
| Recall@5 gain vs baseline | +0.091 | 0.049–0.131 | 350 |

Validation Recall@5 for the grid, in `results/embed.json`: 2e-5 for 1, 2, and 3 epochs is 0.760, 0.760, and 0.765. 5e-5 for 1, 2, and 3 epochs is 0.785, 0.790, and 0.810. The selected run is 5e-5 for 3 epochs.

With gradient checkpointing, batch 16 peaked at 4324 MiB whole-GPU in nvidia-smi on an RTX 4060, desktop included (`results/run_memory.json`).

## Limitations

The questions are synthetic. Training used 1,166 of the planned 1,440 rows. The test set is 350 of 360 after the automatic filter, not a hand review. A document in the test set does not appear in training. The model still truncates a 512-token window by one or two tokens because `[CLS]` and `[SEP]` are added at embedding time.
