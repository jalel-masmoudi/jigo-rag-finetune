"""Hard-negative mining and the validation-based model choice."""

from jigo_rag.negatives import mine_hard_negatives, select_best, training_columns


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
