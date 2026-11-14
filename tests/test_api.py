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
    blank = client.post("/query", json={"question": "   "})
    assert blank.status_code == 422


def test_reranker_reorders_sources_before_the_answer():
    class _WideRetriever:
        def search(self, query, k=5):
            return [
                {"chunk_id": "low#0", "doc_id": "low", "text": "Unrelated leave policy.", "score": 0.9},
                {"chunk_id": "high#0", "doc_id": "high", "text": "Book travel in Concur.", "score": 0.2},
            ]

    class _Reranker:
        def predict(self, pairs, show_progress_bar=False):
            return [1.0 if "Concur" in passage else 0.0 for _question, passage in pairs]

    seen = {}

    class _Generator:
        def answer(self, question, passages, max_new_tokens=128):
            seen["first"] = passages[0]
            return {"answer": "Book travel in Concur.", "prompt_tokens": 1, "completion_tokens": 1}

    client = TestClient(create_app(_WideRetriever(), _Generator(), _Reranker()))
    body = client.post("/query", json={"question": "Where is travel booked?"}).json()
    assert body["sources"][0]["chunk_id"] == "high#0"
    assert "Concur" in seen["first"]


def test_saved_model_dir_must_contain_files(tmp_path):
    from serve.app import saved_model

    assert saved_model("embed", root=tmp_path) is None
    folder = tmp_path / "models" / "embed"
    folder.mkdir(parents=True)
    assert saved_model("embed", root=tmp_path) is None
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert saved_model("embed", root=tmp_path) == folder


def test_health():
    client = TestClient(create_app(_Retriever(), _Generator()))
    assert client.get("/health").json() == {"ok": True}
