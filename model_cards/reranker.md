---
language: en
license: mit
base_model: BAAI/bge-reranker-base
tags:
  - sentence-transformers
  - cross-encoder
---

# bge-reranker-base fine-tuned on the Jigo handbook corpus

Fine-tune of `BAAI/bge-reranker-base`. The base model is MIT licensed.
The numbers are from `results/reranker.json` on 350 filtered test questions
(1,000 bootstrap resamples, seed 42). The test file is the output of
`auto_filter_test.py`. It is not hand-reviewed.

## Training

- Loss: binary cross-entropy
- Learning rate: 2e-5, one epoch, no learning-rate sweep
- Batch size: 4. Batch 8 spilled past 8 GB of VRAM on Windows, and there was no gradient accumulation.
- 1,166 steps, 375 seconds
- Seed: 42

## Intended use

Reorder the top 20 passages from the bi-encoder. The score stored on a hit is the cross-encoder score.

## Metrics

| Retriever under the reranker | Recall@5 | MRR@10 | n |
| --- | --- | --- | --- |
| Baseline | 0.877 (0.843–0.909) | 0.815 (0.778–0.850) | 350 |
| Fine-tuned embeddings | 0.929 (0.900–0.954) | 0.869 (0.838–0.899) | 350 |

Peak memory was 4619 MiB allocated and 5566 MiB reserved (`results/run_memory.json`).

## Limitations

Training used 1,166 of the planned 1,440 synthetic rows. The test set is 350 of 360 after the automatic filter, not a hand review. The reranker does not read the whole corpus. It only reorders the 20 passages the retriever already returned.
