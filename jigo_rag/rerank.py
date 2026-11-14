"""Rerank the top hits of the first-stage retriever.

The cross-encoder reads the question and one passage together and
returns a relevance score. It is slower than the bi-encoder, so it
only sees the top 20 passages, not the whole handbook. Recall@5 is
then computed on that new order.

`bge-reranker-base` is the model named in the project spec.
"""

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_DEPTH = 20


def rerank(model, question, hits):
    """Return a new list of hits, highest cross-encoder score first.

    `model.predict` takes (question, passage) pairs. The score stored
    on each returned hit is that cross-encoder score. The bi-encoder
    score used to retrieve the hit is not kept.
    """
    if not hits:
        return []
    pairs = [(question, hit["text"]) for hit in hits]
    scores = model.predict(pairs, show_progress_bar=False)
    ordered = sorted(zip(scores, hits), key=lambda item: float(item[0]), reverse=True)
    return [hit for _score, hit in ordered]


def labeled_pairs(mined_rows):
    """One positive pair and one pair per hard negative.

    The label column is named `label` because that is the name the
    cross-encoder loss looks for. 1 is the gold passage, 0 is a negative.
    """
    questions = []
    passages = []
    labels = []
    for row in mined_rows:
        questions.append(row["question"])
        passages.append(row["positive"])
        labels.append(1.0)
        index = 1
        while f"negative_{index}" in row:
            questions.append(row["question"])
            passages.append(row[f"negative_{index}"])
            labels.append(0.0)
            index += 1
    return {"question": questions, "passage": passages, "label": labels}
