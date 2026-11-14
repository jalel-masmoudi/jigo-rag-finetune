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
The base model is MIT licensed. This card is a template. Fill the
metrics from `results/embed.json` after the sweep has run on the
reviewed test set. Do not write a smoke-run number here.

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
| Reviewed test set |  |  |  |

## Limitations

The questions are synthetic. The test questions are the ones a person kept in `data/test_reviewed.jsonl`. A document in the test set does not appear in training. The model still truncates a 512-token window by one or two tokens because `[CLS]` and `[SEP]` are added at embedding time.
