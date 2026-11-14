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

from jigo_rag.generator import NOT_FOUND, build_messages


def with_answer(question, passages, answer):
    """Messages the model trains on, including the assistant answer."""
    messages = build_messages(question, passages)
    messages.append({"role": "assistant", "content": answer})
    return messages


def contexts_for_row(row, hits, train_corpus, k=5):
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
    not_found = list(others)
    if len(not_found) < k:
        for chunk in train_corpus:
            if chunk["doc_id"] == row["doc_id"] or chunk["text"] in not_found:
                continue
            not_found.append(chunk["text"])
            if len(not_found) == k:
                break
    return answer_passages[:k], not_found[:k]


def build_examples(rows, train_corpus, search, k=5):
    """`search(question, k)` returns hits. Only training chunks are kept."""
    if not train_corpus:
        raise ValueError("train corpus is empty")
    examples = []
    for row in rows:
        hits = search(row["question"], k=max(k * 2, k))
        answer_passages, not_found_passages = contexts_for_row(row, hits, train_corpus, k=k)
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


def collate_for_training(batch_examples, tokenizer, max_length=1024):
    """Tokenize a batch and mask every prompt token.

    This is the collate the training loop uses. Padding positions are
    masked too. At least one answer token must remain.
    """
    prompts = []
    fulls = []
    for example in batch_examples:
        messages = example["messages"]
        prompts.append(
            tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        )
        fulls.append(tokenizer.apply_chat_template(messages, tokenize=False))
    prompt_lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
    batch = tokenizer(fulls, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    for index, prompt_length in enumerate(prompt_lengths):
        if prompt_length >= int(labels.size(1)):
            raise ValueError("prompt length left no answer token")
        labels[index, :prompt_length] = -100
        if bool((labels[index] == -100).all()):
            raise ValueError("every token in the example was masked")
    batch["labels"] = labels
    return batch
