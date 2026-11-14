"""Hard-negative mining and the validation-based model choice."""

from jigo_rag.negatives import mine_hard_negatives, select_best, training_columns
from jigo_rag.retriever import QUERY_PREFIX


def test_mining_skips_the_gold_chunk_and_the_same_document():
    examples = [
        {
            "question": "Where is travel booked?",
            "gold_chunk_id": "doc-a#0",
            "doc_id": "doc-a",
            "chunk_text": "Book travel in Concur.",
        }
    ]
    corpus = [
        {"chunk_id": "doc-a#0", "doc_id": "doc-a", "text": "Book travel in Concur."},
        {"chunk_id": "doc-a#1", "doc_id": "doc-a", "text": "Same page, next window."},
        {"chunk_id": "doc-b#0", "doc_id": "doc-b", "text": "A different page about leave."},
        {"chunk_id": "doc-c#0", "doc_id": "doc-c", "text": "A page about pay."},
        {"chunk_id": "doc-d#0", "doc_id": "doc-d", "text": "A page about badges."},
    ]

    def search(question, k):
        assert "travel" in question
        return [
            {"chunk_id": "doc-a#0", "doc_id": "doc-a", "text": "Book travel in Concur."},
            {"chunk_id": "doc-a#1", "doc_id": "doc-a", "text": "Same page, next window."},
            {"chunk_id": "doc-b#0", "doc_id": "doc-b", "text": "A different page about leave."},
        ]

    rows = mine_hard_negatives(examples, search, corpus, n_negatives=3, seed=1)
    assert len(rows) == 1
    texts = [rows[0]["negative_1"], rows[0]["negative_2"], rows[0]["negative_3"]]
    assert "Book travel in Concur." not in texts
    assert "Same page, next window." not in texts
    assert "A different page about leave." in texts
    columns = training_columns(rows)
    assert list(columns) == ["question", "positive", "negative_1", "negative_2", "negative_3"]


def test_val_and_test_chunk_ids_never_become_training_negatives():
    examples = [
        {
            "question": "Where is travel booked?",
            "gold_chunk_id": "train-a#0",
            "doc_id": "train-a",
            "chunk_text": "Book travel in Concur.",
        }
    ]
    train_corpus = [
        {"chunk_id": "train-a#0", "doc_id": "train-a", "text": "Book travel in Concur.", "split": "train"},
        {"chunk_id": "train-b#0", "doc_id": "train-b", "text": "A training page about leave.", "split": "train"},
        {"chunk_id": "train-c#0", "doc_id": "train-c", "text": "A training page about pay.", "split": "train"},
        {"chunk_id": "train-d#0", "doc_id": "train-d", "text": "A training page about badges.", "split": "train"},
    ]

    def search(question, k):
        return [
            {"chunk_id": "val-x#0", "doc_id": "val-x", "text": "Validation page."},
            {"chunk_id": "test-y#0", "doc_id": "test-y", "text": "Test page."},
            {"chunk_id": "train-b#0", "doc_id": "train-b", "text": "A training page about leave."},
        ]

    rows = mine_hard_negatives(examples, search, train_corpus, n_negatives=3, seed=1)
    forbidden = {"val-x#0", "test-y#0"}
    used = {rows[0]["negative_1_id"], rows[0]["negative_2_id"], rows[0]["negative_3_id"]}
    assert used.isdisjoint(forbidden)
    assert used <= {chunk["chunk_id"] for chunk in train_corpus}


def test_query_prefix_is_on_the_question_only():
    columns = training_columns(
        [{"question": "Where?", "positive": "In Concur.", "negative_1": "Leave."}],
        n_negatives=1,
        query_prefix=QUERY_PREFIX,
    )
    assert columns["question"][0].startswith(QUERY_PREFIX)
    assert not columns["positive"][0].startswith(QUERY_PREFIX)
    assert not columns["negative_1"][0].startswith(QUERY_PREFIX)


def test_select_best_uses_validation_only():
    records = [
        {"learning_rate": 5e-5, "epochs": 3, "val_recall@5": 0.4},
        {"learning_rate": 2e-5, "epochs": 2, "val_recall@5": 0.5},
        {"learning_rate": 5e-5, "epochs": 1, "val_recall@5": 0.5},
    ]
    winner = select_best(records)
    # Same validation score: the run with fewer epochs wins.
    assert winner["epochs"] == 1
    assert winner["learning_rate"] == 5e-5
