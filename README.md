[![CI](https://github.com/jalel-masmoudi/jigo-rag-finetune/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jalel-masmoudi/jigo-rag-finetune/actions/workflows/ci.yml)

# jigo-rag-finetune

This repository fine-tunes a retriever and a generator for retrieval-augmented answering over enterprise-style documents. The corpus is the public-domain GSA TTS Handbook plus the markdown documentation from Jigo AI Workspace. Documents are chunked, split so that no document is in two sets, and the test questions are the Gemini questions that pass `auto_filter_test.py`. They are not hand-reviewed. The baseline is `BAAI/bge-small-en-v1.5` with FAISS and `Qwen/Qwen2.5-1.5B-Instruct`. Later steps fine-tune the embedding model, add a cross-encoder reranker, and adapt the generator with QLoRA. A small FastAPI service exposes the pipeline. The numbers below are from one run on an RTX 4060 (8 GB, Windows), scored on `data/test_reviewed.jsonl` (350 questions). Smoke files under `results/smoke_*.json` are sanity checks and are not in the table.

## Architecture

A question is embedded with bge-small and compared to every chunk (FAISS, exact inner product, normalized vectors). The top 20 chunks can be reordered by `bge-reranker-base`. The top 5 passages are given to Qwen2.5-1.5B-Instruct, which answers from that text or says `not found`. `/query` returns the answer, the source chunk ids, and the latency.

## Results

From `results/table.json`. Each quality score is the mean and a 95% bootstrap interval (1,000 resamples, seed 42, n = 350). Latency is milliseconds. The judge score is not computed: no Gemini key was set on the training machine.

The LoRA-generator row is the adapted generator on the baseline retriever. That object in `table.json` stores the generator scores only, so Recall and MRR are not repeated there. The full row uses the fine-tuned retriever, the reranker, and the adapted generator. Its retrieval numbers match the reranker row because those two rows share that retriever.

| Setup | Recall@1 | Recall@5 | MRR@10 | Exact match | Token F1 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.463 (0.411–0.511) | 0.760 (0.717–0.803) | 0.586 (0.544–0.628) | 0.180 (0.137–0.220) | 0.452 (0.412–0.493) | 735.3 | 2408.1 |
| Fine-tuned embeddings | 0.526 (0.471–0.580) | 0.851 (0.814–0.886) | 0.664 (0.623–0.701) | 0.177 (0.134–0.217) | 0.465 (0.426–0.504) | 829.4 | 2195.1 |
| Reranker | 0.820 (0.780–0.857) | 0.929 (0.900–0.954) | 0.869 (0.838–0.899) | 0.197 (0.157–0.243) | 0.514 (0.475–0.550) | 1162.0 | 2156.8 |
| LoRA generator |  |  |  | 0.380 (0.323–0.431) | 0.583 (0.539–0.624) | 975.5 | 2212.3 |
| Full | 0.820 (0.780–0.857) | 0.929 (0.900–0.954) | 0.869 (0.838–0.899) | 0.454 (0.400–0.506) | 0.663 (0.621–0.701) | 1292.8 | 2512.8 |

### Embedding validation grid

From `results/embed.json`. The selected run is learning rate 5e-5 for 3 epochs (`models/embed-lr5e-05-ep3`). On the test set that checkpoint has Recall@5 0.851 (0.814–0.886) and MRR@10 0.664 (0.623–0.701). Against the baseline retriever the paired gain is +0.091 Recall@5 (0.049–0.131) and +0.078 MRR@10 (0.044–0.110).

| Learning rate | Epochs | Validation Recall@5 |
| ---: | ---: | ---: |
| 2e-5 | 1 | 0.760 |
| 2e-5 | 2 | 0.760 |
| 2e-5 | 3 | 0.765 |
| 5e-5 | 1 | 0.785 |
| 5e-5 | 2 | 0.790 |
| 5e-5 | 3 | 0.810 |

### Reranker on the two retrievers

From `results/reranker.json`. Depth 20. The reranker is the same model in both rows. The retriever underneath changes.

| Retriever | Recall@5 | MRR@10 |
| --- | ---: | ---: |
| Baseline | 0.877 (0.843–0.909) | 0.815 (0.778–0.850) |
| Fine-tuned | 0.929 (0.900–0.954) | 0.869 (0.838–0.899) |

Memory for the three training stages is in `results/run_memory.json`. Those values were entered by hand. No code measured or wrote them.

## Caveats

- Training used 1,166 of the planned 1,440 synthetic QA rows. Gemini free-tier generation stalled on quota and server errors.
- The test set is 350 of 360 after `auto_filter_test.py`. Five questions referred to the text, and five had answers that were not in the chunk. It was not reviewed by hand.
- The LLM-judge score is not computed because no Gemini key was set on the training PC.
- The reranker ran at batch 4 instead of 8, with no gradient accumulation, because batch 8 spilled past 8 GB of VRAM on Windows.
- LoRA peak GPU memory is approximate: about 4.2 GB was seen in nvidia-smi during training. It is not a logged peak (`results/run_memory.json`, entered by hand).
- The LoRA process exited with a Windows access violation (0xC0000005) after saving the adapter. The adapter was complete, so the eval was run separately.
- The reranker row and the full row share the same retriever, so their retrieval metrics are identical.

## Data

Sources, the pinned commits, the filters, and the split counts are in [DATA.md](DATA.md).

```bash
python make_data.py
python generate_qa.py
python auto_filter_test.py
```

`generate_qa.py` reads `GEMINI_API_KEY` (see `.env.example`). It appends to `data/{train,val,test}.jsonl` and skips chunks already done. `auto_filter_test.py` writes `data/test_reviewed.jsonl` from `data/test.jsonl`. Scores use only that file. The filter is deterministic and does not call an API. A row is dropped when it fails `validate_qa`, when its split is not `test`, when the answer is longer than 40 words, when the question uses a whole phrase such as "the passage" or "this document" (not a bare word), when the normalized answer is not a whole-word span of the normalized chunk, or when the same chunk already kept that normalized question. `DATA.md` lists the same rules.

## Baseline

`eval.py` reports Recall@1, Recall@5, MRR@10, exact match, token F1, a Gemini judge when `GEMINI_API_KEY` is set, p50 and p95 latency, and tokens per query. Quality scores include a 95% bootstrap interval (1,000 resamples, seed 42). Queries use the prefix bge was trained with. Passages do not. The judge rubric is `prompts/judge.txt`. A missing key leaves the judge empty. It is not stored as zero.

```bash
python eval.py --smoke --smoke-size 30 --retrieval-only
```

The smoke query is the first sentence of the gold chunk. The JSON has `"smoke": true`.

## Embedding fine-tune

`train_embed.py` fine-tunes bge-small with MultipleNegativesRankingLoss. Each row is the question, the gold chunk, and three hard negatives from the baseline retriever. Hits from the gold document are skipped, because the windows overlap. The grid is learning rate {2e-5, 5e-5} and {1, 2, 3} epochs. The best validation Recall@5 is scored on the filtered test set once.

```bash
python train_embed.py --smoke
```

## Reranker

`train_reranker.py` fine-tunes `BAAI/bge-reranker-base` with binary cross-entropy on those pairs. The training batch size is 4 (`--batch-size`). Batch 8 did not fit an 8 GB GPU. Evaluation retrieves 20 passages and reorders them. Recall@5 is measured on the new order. There is no learning-rate sweep.

```bash
python train_reranker.py --smoke
```

## Generator fine-tune

`train_lora.py` adapts Qwen2.5-1.5B-Instruct with QLoRA: rank 16, alpha 32, 4-bit NF4. Each question is trained with the gold chunk, and again with a chunk from another document whose target is `not found`. The loss is only on the answer tokens. 4-bit training needs CUDA. The per-step training loss from the RTX 4060 run is in `results/lora_loss.csv` (`step,loss,seed`; 2,332 steps, seed 42). It starts at 2.696 and ends at 0.050.

```bash
python train_lora.py --check-data
```

## Free GPU

On Kaggle or Colab, do not install torch. Those machines already ship a CUDA build. This repo was checked against torch 2.15.0. `requirements.txt` does not pin torch. `requirements-cpu.txt` is only for a CPU machine, and installing it on the GPU machine would replace the CUDA build.

```bash
pip install -r requirements.txt bitsandbytes
python run_gpu.py
```

`run_gpu.py` trains the embedding model, then the reranker, then the QLoRA generator. If one of those steps fails, it still runs `python eval.py --suite` so the rows for the steps that finished are written, and then it exits with the list of failures. `--suite` writes `results/table.json`. Any run that passes `--limit`, including `--retrieval-only` and each row inside the suite file, is marked `"smoke": true` and is not a full test-set result. `notebooks/free_gpu.ipynb` is the same command. Install `bitsandbytes` there. Model cards are in `model_cards/`. The numbers in this file and in those cards are from the run recorded under `results/`.

## Service

One command, after `python make_data.py` so `data/chunks/` exists:

```bash
docker compose up --build
```

The container listens on `0.0.0.0` and `$PORT` (default 8000). `POST /query` with `{"question": "..."}` returns `answer`, `sources`, and `latency_ms`.

`JIGO_GENERATOR=extractive` skips Qwen and returns the opening of the top passage. Use it to try the service on a CPU. The default generator is Qwen.

If `models/embed`, `models/reranker`, or `models/lora` exist, the service loads those instead of the public checkpoints. A checkout with none of those directories serves the baseline.

## Tests

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

## Limitations

The questions are written by a model from a single chunk. The test set is those questions after `auto_filter_test.py`, not a hand review. Splitting by document stops a page from sitting in both train and test; it does not stop two pages from saying the same fact. The embedding model adds `[CLS]` and `[SEP]`, so the last token or two of a full 512-token window can be dropped at retrieval time. The caveats above apply to the numbers in this file.
