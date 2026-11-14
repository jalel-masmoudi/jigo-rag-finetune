"""Reranking order and the labeled pairs the cross-encoder trains on."""

from jigo_rag.rerank import labeled_pairs, rerank


class _FakeEncoder:
    def predict(self, pairs, show_progress_bar=False):
        # Higher score when the passage mentions Concur.
        return [1.0 if "Concur" in passage else 0.1 for _question, passage in pairs]


def test_rerank_puts_the_higher_score_first():
    hits = [
        {"chunk_id": "other#0", "text": "The office closes on federal holidays."},
        {"chunk_id": "gold#0", "text": "Book travel in Concur before the trip."},
    ]
    ordered = rerank(_FakeEncoder(), "Where is travel booked?", hits)
    assert [hit["chunk_id"] for hit in ordered] == ["gold#0", "other#0"]


def test_labeled_pairs_mark_the_gold_passage():
    columns = labeled_pairs(
        [
            {
                "question": "Where?",
                "positive": "In Concur.",
                "negative_1": "On leave.",
                "negative_2": "By email.",
            }
        ]
    )
    assert columns["label"] == [1.0, 0.0, 0.0]
    assert columns["passage"][0] == "In Concur."
    assert columns["question"] == ["Where?"] * 3
