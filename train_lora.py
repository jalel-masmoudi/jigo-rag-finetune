"""QLoRA fine-tune of Qwen2.5-1.5B-Instruct.

The adapter is the spec: rank 16, alpha 32, 4-bit NF4 weights.
The target modules are the attention projections (q, k, v, o) and the
MLP projections (gate, up, down). Those are the linear layers in a
Qwen2 block. The loss is only on the answer tokens.

4-bit training needs a CUDA GPU. bitsandbytes does not run this
quantization on CPU. On a machine without CUDA the script stops,
unless you pass --check-data, which only prints the examples.

    python train_lora.py --check-data
    python train_lora.py

The learning rate is 2e-4 for one epoch. The spec does not list a
sweep for the generator.
"""

import argparse
import json

from jigo_rag.generator import GENERATOR_MODEL
from jigo_rag.jsonl import read_jsonl
from jigo_rag.lora_data import build_examples

LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def four_bit_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_training_rows():
    rows = read_jsonl("data/train.jsonl")
    corpus = []
    for split in ("train", "val", "test"):
        corpus.extend(read_jsonl(f"data/chunks/{split}.jsonl"))
    if rows and corpus:
        return rows, corpus
    # A single literal example so --check-data works before Gemini has run.
    # It is not written to the training file.
    demo_row = {
        "question": "Where is travel booked?",
        "answer": "Book travel in Concur before the trip.",
        "chunk_text": "Book travel in Concur before the trip.",
        "doc_id": "demo/travel.md",
        "gold_chunk_id": "demo/travel.md#0",
    }
    demo_corpus = [
        {"chunk_id": "demo/travel.md#0", "doc_id": "demo/travel.md", "text": demo_row["chunk_text"]},
        {"chunk_id": "demo/leave.md#0", "doc_id": "demo/leave.md", "text": "Annual leave is requested in the leave system."},
    ]
    return [demo_row], demo_corpus


def check_data():
    rows, corpus = load_training_rows()
    examples = build_examples(rows, corpus)
    kinds = {example["kind"] for example in examples}
    counts = {
        "answer": sum(example["kind"] == "answer" for example in examples),
        "not_found": sum(example["kind"] == "not_found" for example in examples),
    }
    print(json.dumps({"examples": len(examples), "counts": counts, "kinds": sorted(kinds)}, indent=2))
    print("--- one not-found example ---")
    not_found = next(example for example in examples if example["kind"] == "not_found")
    for message in not_found["messages"]:
        print(f"[{message['role']}] {message['content'][:400]}")
    return examples


def train():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA in 4-bit needs a CUDA GPU. bitsandbytes does not run this "
            "quantization on CPU. On Kaggle or Colab, run python run_gpu.py. "
            "python train_lora.py --check-data prints the examples without a GPU."
        )
    from peft import get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, corpus = load_training_rows()
    if rows[0]["doc_id"].startswith("demo/"):
        raise SystemExit("Refusing to train on the demo example. Run generate_qa.py first.")
    examples = build_examples(rows, corpus)
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL,
        quantization_config=four_bit_config(),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()

    def collate(batch_examples):
        prompts = []
        fulls = []
        for example in batch_examples:
            messages = example["messages"]
            prompts.append(
                tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            )
            fulls.append(tokenizer.apply_chat_template(messages, tokenize=False))
        prompt_lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
        batch = tokenizer(fulls, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, :prompt_length] = -100
        batch["labels"] = labels
        return batch

    loader = DataLoader(examples, batch_size=1, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-4)
    model.train()
    for epoch in range(1):
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
    model.save_pretrained("models/lora")
    tokenizer.save_pretrained("models/lora")
    print("saved models/lora")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    if args.check_data:
        check_data()
        return
    train()


if __name__ == "__main__":
    main()
