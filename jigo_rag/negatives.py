"""Hard negatives for the embedding fine-tune.

For each training question the baseline retriever returns its top hits.
A hit is kept when it is not the gold chunk and not from the same
document. Same-document chunks overlap by 64 tokens, so they often
contain the answer and would teach the model to push the right page
away.

The loss reads columns, and every row must have the same number of
columns. If the retriever does not return enough negatives, the rest
are random chunks from other documents. Those fillers are marked by
being whatever is left after the mined ones; the column is still a
negative. Three negatives is the default: enough to be harder than
in-batch negatives alone, small enough for a free GPU.
"""

import random


def mine_hard_negatives(examples, search, corpus, n_negatives=3, search_k=20, seed=42):
    """Return rows with question, positive, negative_1 .. negative_n.

    `search(question, k)` returns hits with chunk_id, doc_id, and text,
    best first. `examples` need question, gold_chunk_id, doc_id, and
    chunk_text. An example that cannot fill `n_negatives` is skipped.
    """
    rng = random.Random(seed)
    mined = []
    for example in examples:
        hits = search(example["question"], k=search_k)
        negatives = []
        used = {example["gold_chunk_id"]}
        for hit in hits:
            if hit["chunk_id"] in used or hit["doc_id"] == example["doc_id"]:
                continue
            negatives.append(hit["text"])
            used.add(hit["chunk_id"])
            if len(negatives) == n_negatives:
                break
        if len(negatives) < n_negatives:
            fillers = [
                chunk
                for chunk in corpus
                if chunk["doc_id"] != example["doc_id"] and chunk["chunk_id"] not in used
            ]
            rng.shuffle(fillers)
            for chunk in fillers:
                negatives.append(chunk["text"])
                used.add(chunk["chunk_id"])
                if len(negatives) == n_negatives:
                    break
        if len(negatives) < n_negatives:
            continue
        row = {
            "question": example["question"],
            "positive": example["chunk_text"],
            "gold_chunk_id": example["gold_chunk_id"],
            "doc_id": example["doc_id"],
        }
        for index, text in enumerate(negatives, start=1):
            row[f"negative_{index}"] = text
        mined.append(row)
    return mined


def training_columns(rows, n_negatives=3):
    """Column order is the loss input order: query, positive, then negatives.

    Names are for people reading the file. The trainer uses the order.
    """
    columns = {
        "question": [row["question"] for row in rows],
        "positive": [row["positive"] for row in rows],
    }
    for index in range(1, n_negatives + 1):
        key = f"negative_{index}"
        columns[key] = [row[key] for row in rows]
    return columns


def select_best(records):
    """Pick the run with the highest validation Recall@5.

    A tie goes to fewer epochs, then the smaller learning rate. The
    test set is not an input to this function. Call it on validation
    scores only, then score the winner on the test set once.
    """
    if not records:
        raise ValueError("no runs to choose from")
    return min(records, key=lambda record: (-record["val_recall@5"], record["epochs"], record["learning_rate"]))
