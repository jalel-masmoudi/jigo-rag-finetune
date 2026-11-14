"""POST /query returns an answer, the source chunks, and the latency.

The default stack is the bge-small FAISS index plus Qwen2.5-1.5B-Instruct.
Set JIGO_GENERATOR=extractive to skip Qwen and answer with the first
sentence of the top passage. That mode is for checking the HTTP service
on a CPU. It is not the fine-tuned pipeline.

    uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from jigo_rag.generator import QwenGenerator
from jigo_rag.jsonl import read_jsonl
from jigo_rag.rerank import rerank
from jigo_rag.retriever import EMBED_MODEL, FaissRetriever


class QueryIn(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value):
        # Spaces have length, so min_length does not catch a blank question.
        if not value.strip():
            raise ValueError("question is empty")
        return value


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


def saved_model(name, root="."):
    """Return models/<name> when a training script has written it, else None."""
    path = Path(root) / "models" / name
    if path.is_dir() and any(path.iterdir()):
        return path
    return None


def load_stack():
    # Fine-tuned weights are optional. A fresh checkout serves the baseline.
    embed = saved_model("embed")
    retriever = FaissRetriever(model_name=str(embed) if embed else EMBED_MODEL)
    retriever.build(load_chunks())
    reranker = None
    reranker_dir = saved_model("reranker")
    if reranker_dir is not None:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(str(reranker_dir))
    if os.environ.get("JIGO_GENERATOR", "qwen") == "extractive":
        generator = ExtractiveGenerator()
    else:
        generator = QwenGenerator()
        lora = saved_model("lora")
        generator.load(adapter_path=str(lora) if lora else None)
    return retriever, reranker, generator


def create_app(retriever=None, generator=None, reranker=None):
    app = FastAPI(title="jigo-rag-finetune")
    holder = {"retriever": retriever, "generator": generator, "reranker": reranker}

    def ready():
        if holder["retriever"] is None or holder["generator"] is None:
            loaded_retriever, loaded_reranker, loaded_generator = load_stack()
            holder["retriever"] = loaded_retriever
            holder["generator"] = loaded_generator
            holder["reranker"] = loaded_reranker
        return holder["retriever"], holder["generator"], holder["reranker"]

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/query", response_model=QueryOut)
    def query(body: QueryIn):
        retriever, generator, reranker_model = ready()
        started = time.perf_counter()
        # The cross-encoder only looks at the top 20. Without it, five
        # passages are enough for the generator.
        depth = 20 if reranker_model is not None else 5
        hits = retriever.search(body.question, k=depth)
        if reranker_model is not None:
            hits = rerank(reranker_model, body.question, hits)[:5]
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
