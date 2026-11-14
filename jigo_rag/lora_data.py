"""Training examples for the QLoRA generator.

Each real question becomes two examples:

- The gold chunk is the context and the target is the gold answer.
- A chunk from a different document is the context and the target is
  exactly "not found".

The second kind is how the model learns to refuse when the retrieved
text does not contain the answer. Using a different document, not the
next window of the same page, matters: the next window overlaps this
one and often does contain the answer.

Prompt tokens are masked with -100 so the loss is only on the answer.
"""

import random

from jigo_rag.generator import NOT_FOUND, build_messages


def with_answer(question, passages, answer):
    """Messages the model trains on, including the assistant answer."""
    messages = build_messages(question, passages)
    messages.append({"role": "assistant", "content": answer})
    return messages


def mask_prompt_labels(input_ids, prompt_length):
    """Copy input ids into labels and hide the prompt.

    The causal LM loss ignores -100. If the mask covers the whole
    sequence there is nothing to learn, so that case is an error.
    """
    if prompt_length <= 0 or prompt_length >= len(input_ids):
        raise ValueError("prompt length must leave at least one answer token")
    labels = list(input_ids)
    for index in range(prompt_length):
        labels[index] = -100
    return labels


def build_examples(rows, corpus, seed=42):
    """Return answer examples and not-found examples, one pair per row."""
    rng = random.Random(seed)
    others_ok = [chunk for chunk in corpus]
    if not others_ok:
        raise ValueError("corpus is empty")
    examples = []
    for row in rows:
        examples.append(
            {
                "messages": with_answer(row["question"], [row["chunk_text"]], row["answer"]),
                "kind": "answer",
                "doc_id": row["doc_id"],
            }
        )
        pool = [chunk for chunk in corpus if chunk["doc_id"] != row["doc_id"]]
        if not pool:
            raise ValueError(f"no other document to build a not-found example for {row['doc_id']}")
        other = rng.choice(pool)
        examples.append(
            {
                "messages": with_answer(row["question"], [other["text"]], NOT_FOUND),
                "kind": "not_found",
                "doc_id": other["doc_id"],
            }
        )
    return examples
