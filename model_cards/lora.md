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
The numbers are from `results/table.json` on 350 filtered test questions
(1,000 bootstrap resamples, seed 42). The judge cell is empty because
no Gemini key was set for that run.

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
| LoRA on the baseline retriever | 0.380 (0.323–0.431) | 0.583 (0.539–0.624) | not computed | 350 |
| Full pipeline (fine-tuned retriever, reranker, LoRA) | 0.454 (0.400–0.506) | 0.663 (0.621–0.701) | not computed | 350 |

The training loss log (`results/lora_loss.csv`) starts at 2.696 and ends at 0.050 over 2,332 steps. About 4.2 GB was observed in nvidia-smi during training. That figure is approximate. It is not a logged peak, and the process did not spill into system RAM (`results/run_memory.json`).

## Limitations

The adapter was trained to copy a short answer out of one passage, and to refuse on a passage from a different page. It was not trained on multi-hop questions. Training used 1,166 of the planned 1,440 rows. The test set was not reviewed by hand. After the adapter was saved, the process exited with a Windows access violation (0xC0000005). The saved adapter was complete, and the scores above were measured in a separate eval. The judge, when it is run, is a Gemini model and is a second opinion, not a proof.
