"""The /query response shape. The models are fakes, so this does not score retrieval."""

from fastapi.testclient import TestClient

from serve.app import create_app


class _Retriever:
    def search(self, query, k=5):
        assert query
        return [
            {
                "chunk_id": "gsa-handbook/pages/travel.md#0",
                "doc_id": "gsa-handbook/pages/travel.md",
                "text": "Book travel in Concur before the trip.",
                "score": 0.82,
            }
        ]


class _Generator:
    def answer(self, question, passages, max_new_tokens=128):
        assert passages
        return {"answer": "Book travel in Concur before the trip.", "prompt_tokens": 12, "completion_tokens": 6}


def test_query_returns_answer_sources_and_latency():
    client = TestClient(create_app(_Retriever(), _Generator()))
    response = client.post("/query", json={"question": "Where is travel booked?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Book travel")
    assert body["sources"][0]["chunk_id"].endswith("#0")
    assert body["sources"][0]["score"] == 0.82
    assert body["latency_ms"] >= 0


def test_empty_question_is_rejected():
    client = TestClient(create_app(_Retriever(), _Generator()))
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_health():
    client = TestClient(create_app(_Retriever(), _Generator()))
    assert client.get("/health").json() == {"ok": True}
