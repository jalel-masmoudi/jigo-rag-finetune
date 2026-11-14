# jigo-rag-finetune

This repository fine-tunes a retriever and a generator for retrieval-augmented answering over enterprise-style documents. The corpus is the public-domain GSA TTS Handbook plus the markdown documentation from Jigo AI Workspace. The pipeline is reproducible from a fixed seed: documents are chunked, split into train, validation, and test with no document shared across splits, and a person reviews the test questions before any score is reported. The baseline is BAAI/bge-small-en-v1.5 with FAISS and Qwen2.5-1.5B-Instruct. Later steps fine-tune the embedding model, optionally add a cross-encoder reranker, and adapt the generator with QLoRA. A small FastAPI service exposes the finished pipeline. Every number in the results table comes from a real evaluation run.

## Data

How the corpus is pinned, filtered, chunked, and split is written up in [DATA.md](DATA.md).

```bash
python make_data.py
python generate_qa.py
python review_test.py
```

`generate_qa.py` reads `GEMINI_API_KEY` from the environment (see `.env.example`). It appends to `data/{train,val,test}.jsonl` and skips chunks that are already done, so a free-tier daily cap can be continued the next day. `review_test.py` is the hand check. Scores later use only `data/test_reviewed.jsonl`. The full generated files are gitignored. `data/sample/` is a small committed example of the row format.

## Baseline

`eval.py` embeds every chunk with `BAAI/bge-small-en-v1.5` and searches them with FAISS (exact inner product on normalized vectors). Queries use the prefix this model was trained with. The generator is `Qwen/Qwen2.5-1.5B-Instruct`. Scores are Recall@1, Recall@5, MRR@10, exact match, token F1, a Gemini judge when `GEMINI_API_KEY` is set, p50 and p95 latency, and tokens per query. Quality scores include a 95% bootstrap interval.

The official run reads `data/test_reviewed.jsonl` and will not score `data/test.jsonl`. There is no reviewed test file yet, so no baseline number below is an official result.

```bash
python eval.py --smoke --smoke-size 30 --retrieval-only
```

The smoke query is the first sentence of the gold chunk, searched over the full chunk index. The JSON it writes has `"smoke": true`. Do not copy it into the results table.

## Embedding fine-tune

`train_embed.py` fine-tunes `bge-small` with MultipleNegativesRankingLoss. Each training row is the question, the gold chunk, and three hard negatives mined by the baseline retriever. Negatives from the gold document are skipped, because those windows overlap and often contain the answer. The grid is learning rate {2e-5, 5e-5} and {1, 2, 3} epochs. The highest validation Recall@5 wins. That checkpoint is scored on the reviewed test set once.

```bash
python train_embed.py --smoke
```

The smoke flag is one optimizer step. It is not a sweep and not a test score.

## Reranker

`train_reranker.py` fine-tunes `BAAI/bge-reranker-base` with binary cross-entropy on the same hard negatives. At evaluation the first-stage retriever returns 20 passages and the cross-encoder reorders them. Recall@5 is measured on that order. There is no learning-rate sweep.

```bash
python train_reranker.py --smoke
```

## Generator fine-tune

`train_lora.py` adapts `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA: rank 16, alpha 32, 4-bit NF4. Each question is trained twice: once with the gold chunk and the gold answer, and once with a chunk from a different document and the target `not found`. The loss is only on the answer tokens.

4-bit training needs a CUDA GPU. `python train_lora.py --check-data` prints the examples on a CPU. `run_gpu.py` is the command to use on Kaggle or Colab.
