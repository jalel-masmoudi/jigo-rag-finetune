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
falls only on the answer. A prompt that does not fit drops whole
non-gold passages, lowest rank first. The gold passage is cut only
when it alone is still over the cap. The question and the assistant
header stay whole.
"""

import random
import re

from jigo_rag.generator import NOT_FOUND, build_messages, format_context

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
                "passages": answer_passages,
                "gold_passage": row["chunk_text"],
            }
        )
        examples.append(
            {
                "messages": with_answer(row["question"], not_found_passages, NOT_FOUND),
                "kind": "not_found",
                "doc_id": row["doc_id"],
                "passages": not_found_passages,
                "gold_passage": None,
            }
        )
    return examples


def _as_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(encoded)


_QUESTION_MARK = "\n\nQuestion: "


def _passages_from_context(context):
    """Split a formatted context back into passage strings.

    Numbered blocks are separated by a blank line. The question is
    already removed by the caller.
    """
    body = context[len("Context:\n") :] if context.startswith("Context:\n") else context
    if not body.strip():
        return []
    passages = []
    for part in re.split(r"\n\n(?=\[\d+\] )", body):
        match = re.match(r"\[\d+\] (.*)\Z", part, re.S)
        passages.append(match.group(1) if match else part)
    return passages


def _fit_messages(tokenizer, messages, max_length, passages=None, gold=None):
    """Make the chat string fit by dropping passages, then cutting gold.

    Non-gold passages are removed whole, lowest rank first. Rank is the
    order in the list: the last passage is the lowest. The gold passage
    stays intact unless it, with the question and the assistant header,
    is still over the cap. The question itself is not cut.
    """
    if len(messages) < 2 or _QUESTION_MARK not in messages[1]["content"]:
        return messages
    context, question = messages[1]["content"].rsplit(_QUESTION_MARK, 1)
    if passages is None:
        passages = _passages_from_context(context)

    def assemble(kept):
        user = f"Context:\n{format_context(kept)}{_QUESTION_MARK}{question}"
        return [
            messages[0],
            {"role": "user", "content": user},
            *messages[2:],
        ]

    def fits(kept):
        full = tokenizer.apply_chat_template(assemble(kept), tokenize=False)
        return len(_as_ids(tokenizer, full)) <= max_length

    items = []
    gold_marked = False
    for text in passages:
        is_gold = gold is not None and not gold_marked and text == gold
        if is_gold:
            gold_marked = True
        items.append({"text": text, "gold": is_gold})

    def kept_texts():
        return [item["text"] for item in items]

    while items and not fits(kept_texts()):
        drop_at = None
        for index in range(len(items) - 1, -1, -1):
            if not items[index]["gold"]:
                drop_at = index
                break
        if drop_at is None:
            break
        del items[drop_at]

    if fits(kept_texts()) or not any(item["gold"] for item in items):
        return assemble(kept_texts())

    # The gold passage alone is over the cap. Shorten that text only.
    gold_item = next(item for item in items if item["gold"])
    original = gold_item["text"]
    lo, hi = 0, len(original)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        gold_item["text"] = original[:mid]
        if fits(kept_texts()):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    gold_item["text"] = original[:best]
    return assemble(kept_texts())


def collate_for_training(batch_examples, tokenizer, max_length=MAX_TRAIN_LENGTH):
    """Tokenize a batch and mask every prompt token.

    The cap is 3072 Qwen tokens. Five retrieved passages are often
    longer than 1024, which used to cut off the question and then mask
    the whole row. Whole non-gold passages are dropped first, lowest
    rank first. The gold passage is cut only if it alone still does not
    fit. The question and the assistant header stay in the window.
    Padding positions are masked too.
    """
    import torch

    rows = []
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    for example in batch_examples:
        messages = _fit_messages(
            tokenizer,
            example["messages"],
            max_length,
            passages=example.get("passages"),
            gold=example.get("gold_passage"),
        )
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
        # The prompt is already the fitted question and header. If the
        # answer is what overflows, drop answer tokens, not the question.
        if len(prompt_ids) < max_length:
            answer_ids = answer_ids[: max_length - len(prompt_ids)] or answer_ids[:1]
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
