"""POST /query returns an answer, the source chunks, and the latency.

The default stack is the bge-small FAISS index plus Qwen2.5-1.5B-Instruct.
Set JIGO_GENERATOR=extractive to skip Qwen and answer with the first
sentence of the top passage. That mode is for checking the HTTP service
on a CPU. It is not the fine-tuned pipeline.

    uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from jigo_rag.generator import QwenGenerator
from jigo_rag.jsonl import read_jsonl
from jigo_rag.retriever import FaissRetriever


class QueryIn(BaseModel):
    question: str = Field(min_length=1)


class SourceOut(BaseModel):
    chunk_id: str
    doc_id: str
    score: float


class QueryOut(BaseModel):
    answer: str
    sources: list[SourceOut]
    latency_ms: float


class ExtractiveGenerator:
    """Answer with the opening of the top passage. No language model."""

    def answer(self, question, passages, max_new_tokens=128):
        text = passages[0].strip() if passages else "not found"
        sentence = text.split(". ", 1)[0].strip()
        return {"answer": sentence or "not found", "prompt_tokens": 0, "completion_tokens": 0}


def load_chunks():
    chunks = []
    for split in ("train", "val", "test"):
        chunks.extend(read_jsonl(f"data/chunks/{split}.jsonl"))
    if not chunks:
        raise RuntimeError("data/chunks/ is empty. Run python make_data.py before starting the service.")
    return chunks


def load_stack():
    retriever = FaissRetriever().build(load_chunks())
    if os.environ.get("JIGO_GENERATOR", "qwen") == "extractive":
        return retriever, ExtractiveGenerator()
    return retriever, QwenGenerator()


def create_app(retriever=None, generator=None):
    app = FastAPI(title="jigo-rag-finetune")
    holder = {"retriever": retriever, "generator": generator}

    def ready():
        if holder["retriever"] is None or holder["generator"] is None:
            holder["retriever"], holder["generator"] = load_stack()
        return holder["retriever"], holder["generator"]

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/query", response_model=QueryOut)
    def query(body: QueryIn):
        retriever, generator = ready()
        started = time.perf_counter()
        hits = retriever.search(body.question, k=5)
        generated = generator.answer(body.question, [hit["text"] for hit in hits])
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "answer": generated["answer"],
            "sources": [
                {"chunk_id": hit["chunk_id"], "doc_id": hit["doc_id"], "score": hit["score"]}
                for hit in hits
            ],
            "latency_ms": latency_ms,
        }

    return app


app = create_app()
