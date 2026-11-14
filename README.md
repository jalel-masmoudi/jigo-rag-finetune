# jigo-rag-finetune

This repository fine-tunes a retriever and a generator for retrieval-augmented answering over enterprise-style documents. The corpus is the public-domain GSA TTS Handbook plus the markdown documentation from Jigo AI Workspace. Documents are chunked, split so that no document is in two sets, and the test questions are the ones a person has reviewed. The baseline is `BAAI/bge-small-en-v1.5` with FAISS and `Qwen/Qwen2.5-1.5B-Instruct`. Later steps fine-tune the embedding model, add a cross-encoder reranker, and adapt the generator with QLoRA. A small FastAPI service exposes the pipeline. Every number in the table below has to come from a real run on `data/test_reviewed.jsonl`. The cells are empty because that run has not happened.

## Architecture

A question is embedded with bge-small and compared to every chunk (FAISS, exact inner product, normalized vectors). The top 20 chunks can be reordered by `bge-reranker-base`. The top 5 passages are given to Qwen2.5-1.5B-Instruct, which answers from that text or says `not found`. `/query` returns the answer, the source chunk ids, and the latency.

## Results

Fill this table from `results/*.json` after `python run_gpu.py` on a GPU, and only from files whose `"smoke"` field is absent or false. The smoke files are sanity checks. They are not results.

| Setup | Recall@5 | MRR@10 | Answer F1 | Judge score | p95 latency |
| --- | --- | --- | --- | --- | --- |
| Baseline | | | | | |
| + fine-tuned embeddings | | | | | |
| + reranker | | | | | |
| + LoRA generator (full) | | | | | |

## Data

Sources, the pinned commits, the filters, and the split counts are in [DATA.md](DATA.md).

```bash
python make_data.py
python generate_qa.py
python review_test.py
```

`generate_qa.py` reads `GEMINI_API_KEY` (see `.env.example`). It appends to `data/{train,val,test}.jsonl` and skips chunks already done. `review_test.py` is the hand check. Scores use only `data/test_reviewed.jsonl`.

## Baseline

`eval.py` reports Recall@1, Recall@5, MRR@10, exact match, token F1, a Gemini judge when `GEMINI_API_KEY` is set, p50 and p95 latency, and tokens per query. Quality scores include a 95% bootstrap interval (1,000 resamples, seed 42). Queries use the prefix bge was trained with. Passages do not. The judge rubric is `prompts/judge.txt`. A missing key leaves the judge empty. It is not stored as zero.

```bash
python eval.py --smoke --smoke-size 30 --retrieval-only
```

The smoke query is the first sentence of the gold chunk. The JSON has `"smoke": true`.

## Embedding fine-tune

`train_embed.py` fine-tunes bge-small with MultipleNegativesRankingLoss. Each row is the question, the gold chunk, and three hard negatives from the baseline retriever. Hits from the gold document are skipped, because the windows overlap. The grid is learning rate {2e-5, 5e-5} and {1, 2, 3} epochs. The best validation Recall@5 is scored on the reviewed test set once.

```bash
python train_embed.py --smoke
```

## Reranker

`train_reranker.py` fine-tunes `BAAI/bge-reranker-base` with binary cross-entropy on those pairs. Evaluation retrieves 20 passages and reorders them. Recall@5 is measured on the new order. There is no learning-rate sweep.

```bash
python train_reranker.py --smoke
```

## Generator fine-tune

`train_lora.py` adapts Qwen2.5-1.5B-Instruct with QLoRA: rank 16, alpha 32, 4-bit NF4. Each question is trained with the gold chunk, and again with a chunk from another document whose target is `not found`. The loss is only on the answer tokens. 4-bit training needs CUDA.

```bash
python train_lora.py --check-data
```

## Free GPU

`run_gpu.py` runs the baseline eval, the embedding sweep, the reranker, and the QLoRA step, in that order. It exits if CUDA is missing. `notebooks/free_gpu.ipynb` is the same command for Kaggle or Colab. Install `bitsandbytes` there; it is not required for the CPU checks. Model-card templates are in `model_cards/`. Their metric cells are empty on purpose.

## Service

One command, after `python make_data.py` so `data/chunks/` exists:

```bash
docker compose up --build
```

The container listens on `0.0.0.0` and `$PORT` (default 8000). `POST /query` with `{"question": "..."}` returns `answer`, `sources`, and `latency_ms`.

`JIGO_GENERATOR=extractive` skips Qwen and returns the opening of the top passage. Use it to try the service on a CPU. The default generator is Qwen.

## Tests

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest
```

## Limitations

The questions are written by a model from a single chunk, then filtered so the answer's words appear in that chunk. The test set is only as good as the hand review. Splitting by document stops a page from sitting in both train and test; it does not stop two pages from saying the same fact. The embedding model adds `[CLS]` and `[SEP]`, so the last token or two of a full 512-token window can be dropped at retrieval time. No official score is in this file yet.
