"""Training examples for the QLoRA generator.

Each training question becomes two examples, both using up to five
passages, the same shape eval and the HTTP service give the model.

- Answerable: the top passages from the training index, with the gold
  chunk kept in the list even if retrieval missed it. The target is
  the gold answer.
- Not found: the top passages from the training index with the gold
  chunk removed. The target is exactly "not found".

`train_corpus` is the only pool. A hit whose id is not in that list
is ignored, so a validation or test chunk cannot become context.
The test set is scored later; it is not a source of training context.

The collate function hides prompt tokens with -100. The loss then
falls only on the answer.
"""

import random

from jigo_rag.generator import NOT_FOUND, build_messages

MAX_TRAIN_LENGTH = 3072


def with_answer(question, passages, answer):
    """Messages the model trains on, including the assistant answer."""
    messages = build_messages(question, passages)
    messages.append({"role": "assistant", "content": answer})
    return messages


def contexts_for_row(row, hits, train_corpus, k=5, seed=42):
    """Return (answer_passages, not_found_passages).

    Both lists contain only text from `train_corpus`, except the gold
    chunk text, which is inserted into the answerable list.
    """
    train_ids = {chunk["chunk_id"] for chunk in train_corpus}
    usable = [hit for hit in hits if hit["chunk_id"] in train_ids and hit["chunk_id"] != row["gold_chunk_id"]]
    # Drop other windows of the gold document. They overlap the answer.
    usable = [hit for hit in usable if hit.get("doc_id") != row["doc_id"]]
    others = []
    for hit in usable:
        others.append(hit["text"])
        if len(others) == k - 1:
            break
    if len(others) < k - 1:
        for chunk in train_corpus:
            if chunk["doc_id"] == row["doc_id"] or chunk["text"] in others:
                continue
            others.append(chunk["text"])
            if len(others) == k - 1:
                break
    answer_passages = [row["chunk_text"]] + others
    # The gold passage is not always first. The position is fixed for a seed.
    random.Random(f"{seed}:{row.get('gold_chunk_id', row['doc_id'])}").shuffle(answer_passages)
    not_found = list(others)
    if len(not_found) < k:
        for chunk in train_corpus:
            if chunk["doc_id"] == row["doc_id"] or chunk["text"] in not_found:
                continue
            not_found.append(chunk["text"])
            if len(not_found) == k:
                break
    return answer_passages[:k], not_found[:k]


def build_examples(rows, train_corpus, search, k=5, seed=42):
    """`search(question, k)` returns hits. Only training chunks are kept."""
    if not train_corpus:
        raise ValueError("train corpus is empty")
    examples = []
    for index, row in enumerate(rows):
        hits = search(row["question"], k=max(k * 2, k))
        answer_passages, not_found_passages = contexts_for_row(
            row, hits, train_corpus, k=k, seed=seed + index
        )
        if not not_found_passages:
            raise ValueError(f"no training chunk left for a not-found example of {row['doc_id']}")
        examples.append(
            {
                "messages": with_answer(row["question"], answer_passages, row["answer"]),
                "kind": "answer",
                "doc_id": row["doc_id"],
            }
        )
        examples.append(
            {
                "messages": with_answer(row["question"], not_found_passages, NOT_FOUND),
                "kind": "not_found",
                "doc_id": row["doc_id"],
            }
        )
    return examples


def _as_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(encoded)


def collate_for_training(batch_examples, tokenizer, max_length=MAX_TRAIN_LENGTH):
    """Tokenize a batch and mask every prompt token.

    The cap is 3072 Qwen tokens. Five retrieved passages are often
    longer than 1024, which used to cut off the answer and then mask
    the whole row. The prompt is shortened so at least one answer
    token remains. Padding positions are masked too.
    """
    import torch

    rows = []
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    for example in batch_examples:
        messages = example["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_ids = _as_ids(tokenizer, prompt)
        full_ids = _as_ids(tokenizer, full)
        if full_ids[: len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
            answer_ids = full_ids[len(prompt_ids) :]
        else:
            overlap = min(len(prompt_ids), max(0, len(full_ids) - 1))
            answer_ids = full_ids[overlap:] or full_ids[-1:]
            prompt_ids = full_ids[:overlap]
        if not answer_ids:
            answer_ids = full_ids[-1:]
        if len(answer_ids) >= max_length:
            answer_ids = answer_ids[-(max_length - 1) :]
        prompt_ids = prompt_ids[: max_length - len(answer_ids)]
        rows.append((prompt_ids + answer_ids, len(prompt_ids)))
    width = max(len(ids) for ids, _prompt_len in rows)
    input_ids = []
    attention = []
    labels = []
    for ids, prompt_len in rows:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        lab = list(ids) + [-100] * pad
        for index in range(prompt_len):
            lab[index] = -100
        labels.append(lab)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention),
        "labels": torch.tensor(labels),
    }
