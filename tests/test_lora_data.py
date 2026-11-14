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
    """Splits on spaces and pads with zeros. Enough to see the mask."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = " ".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += " ASSIST"
        return text

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        if isinstance(text, str):
            return {"input_ids": text.split()}
        rows = [item.split() for item in text]
        width = max(len(row) for row in rows)
        ids = []
        mask = []
        for row in rows:
            ids.append([hash(token) % 1000 + 1 for token in row] + [0] * (width - len(row)))
            mask.append([1] * len(row) + [0] * (width - len(row)))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


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
