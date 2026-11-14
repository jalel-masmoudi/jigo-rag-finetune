"""Not-found contexts and the collate the training loop actually calls."""

import torch

from jigo_rag.generator import NOT_FOUND
from jigo_rag.lora_data import build_examples, collate_for_training, contexts_for_row


def _row():
    return {
        "question": "Where is travel booked?",
        "answer": "In Concur.",
        "chunk_text": "Book travel in Concur.",
        "doc_id": "train-a",
        "gold_chunk_id": "train-a#0",
    }


def _train_corpus():
    return [
        {"chunk_id": "train-a#0", "doc_id": "train-a", "text": "Book travel in Concur.", "split": "train"},
        {"chunk_id": "train-b#0", "doc_id": "train-b", "text": "Leave is requested separately.", "split": "train"},
        {"chunk_id": "train-c#0", "doc_id": "train-c", "text": "Badges are issued at the desk.", "split": "train"},
    ]


def test_not_found_context_is_train_only_and_excludes_the_gold_chunk():
    hits = [
        {"chunk_id": "test-z#0", "doc_id": "test-z", "text": "A held-out page."},
        {"chunk_id": "train-a#1", "doc_id": "train-a", "text": "Same page, next window."},
        {"chunk_id": "train-b#0", "doc_id": "train-b", "text": "Leave is requested separately."},
        {"chunk_id": "val-y#0", "doc_id": "val-y", "text": "A validation page."},
    ]
    answer, refused = contexts_for_row(_row(), hits, _train_corpus(), k=2)
    assert answer[0] == "Book travel in Concur."
    assert "A held-out page." not in answer
    assert "A validation page." not in answer
    assert "A held-out page." not in refused
    assert "Book travel in Concur." not in refused
    assert "Same page, next window." not in refused
    assert "Leave is requested separately." in refused
    examples = build_examples([_row()], _train_corpus(), lambda question, k: hits, k=2)
    assert examples[0]["messages"][-1]["content"] == "In Concur."
    assert examples[1]["messages"][-1]["content"] == NOT_FOUND
    assert "Book travel in Concur." in examples[0]["messages"][1]["content"]
    assert "Book travel in Concur." not in examples[1]["messages"][1]["content"]


class _Tokenizer:
    """Splits on spaces. Enough to see which tokens are masked."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = " ".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += " ASSIST"
        return text

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [abs(hash(token)) % 1000 + 1 for token in text.split()]}


def test_gold_passage_position_depends_on_the_seed():
    hits = [
        {"chunk_id": "train-b#0", "doc_id": "train-b", "text": "Leave is requested separately."},
        {"chunk_id": "train-c#0", "doc_id": "train-c", "text": "Badges are issued at the desk."},
    ]
    positions = set()
    for seed in range(12):
        answer, _refused = contexts_for_row(_row(), hits, _train_corpus(), k=3, seed=seed)
        positions.add(answer.index("Book travel in Concur."))
    assert len(positions) > 1
    first = contexts_for_row(_row(), hits, _train_corpus(), k=3, seed=4)
    second = contexts_for_row(_row(), hits, _train_corpus(), k=3, seed=4)
    assert first[0] == second[0]


def test_loss_log_and_seed(tmp_path):
    import torch
    from train_lora import append_loss, set_training_seed, start_loss_log

    set_training_seed(7)
    first = torch.rand(2).tolist()
    set_training_seed(7)
    assert torch.rand(2).tolist() == first
    csv_path = start_loss_log(7, directory=tmp_path)
    append_loss(csv_path, 1, "0.500000", 7)
    assert "1,0.500000,7" in csv_path.read_text(encoding="utf-8")
    config = __import__("json").loads((tmp_path / "lora_config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 7 and config["r"] == 16 and config["lora_alpha"] == 32


def test_real_qwen_tokenizer_collates_long_passages():
    from pathlib import Path

    from transformers import AutoTokenizer

    from jigo_rag.jsonl import read_jsonl
    from jigo_rag.lora_data import collate_for_training, with_answer

    texts = []
    for split in ("train", "val", "test"):
        chunk_path = Path("data/chunks") / f"{split}.jsonl"
        if chunk_path.exists():
            for row in read_jsonl(chunk_path):
                if row.get("n_tokens", 0) >= 400:
                    texts.append(row["text"])
                if len(texts) >= 5:
                    break
        if len(texts) >= 5:
            break
    if len(texts) < 5:
        for split in ("train", "val", "test"):
            for row in read_jsonl(Path("data/sample") / f"{split}.jsonl"):
                texts.append(row["chunk_text"])
                if len(texts) >= 5:
                    break
    while len(texts) < 5:
        texts.append(texts[-1])
    example = {"messages": with_answer("Where is travel booked?", texts[:5], "In Concur.")}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    batch = collate_for_training([example], tokenizer)
    assert int(batch["input_ids"].shape[-1]) <= 3072
    assert any(token != -100 for token in batch["labels"][0].tolist())


def test_collate_masks_the_prompt_and_keeps_the_answer():
    example = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "Where?"},
            {"role": "assistant", "content": "In Concur."},
        ]
    }
    tokenizer = _Tokenizer()
    batch = collate_for_training([example], tokenizer)
    prompt = tokenizer.apply_chat_template(example["messages"][:-1], tokenize=False, add_generation_prompt=True)
    prompt_length = len(tokenizer(prompt)["input_ids"])
    labels = batch["labels"][0].tolist()
    assert labels[:prompt_length] == [-100] * prompt_length
    assert any(token != -100 for token in labels[prompt_length:])
