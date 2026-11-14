---
language: en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - peft
  - lora
---

# QLoRA adapter for Qwen2.5-1.5B-Instruct

Adapter for `Qwen/Qwen2.5-1.5B-Instruct`. The base model is Apache 2.0.
This card is a template. Fill the metrics from a real `eval.py` run
that loads this adapter. Do not write a smoke-run number here.

## Training

- Method: QLoRA, 4-bit NF4, double quantization, float16 compute
- Rank: 16
- Alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Learning rate: 2e-4, one epoch
- Data: each question once with the gold chunk, and once with a chunk from another document labeled `not found`
- The loss is on the answer tokens only

## Intended use

Answer a question from retrieved handbook passages. Reply with exactly `not found` when the passages do not contain the answer.

## Metrics

| | Exact match | Token F1 | Judge | n |
| --- | --- | --- | --- | --- |
| Reviewed test set, same retrieved context as the baseline |  |  |  |  |

## Limitations

The adapter was trained to copy a short answer out of one passage, and to refuse on a passage from a different page. It was not trained on multi-hop questions. The judge, when it is run, is a Gemini model and is a second opinion, not a proof.
