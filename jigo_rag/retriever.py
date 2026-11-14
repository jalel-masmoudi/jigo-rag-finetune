"""FAISS retriever over the bge-small-en-v1.5 embeddings.

Passages are encoded as they are. Queries get the prefix this model was
trained with: "Represent this sentence for searching relevant passages: ".
Without that prefix the query vector sits in a different part of the space
and retrieval is worse. The prefix is not added to passages.

Vectors are L2-normalized, so inner product is cosine similarity.
FAISS IndexFlatIP is an exact search, not an approximate one. The corpus
is a few hundred chunks, so exact search is simpler and does not lose hits.
"""

import numpy as np

from jigo_rag.config import TOKENIZER_NAME

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_MODEL = TOKENIZER_NAME


class FaissRetriever:
    def __init__(self, model_name=EMBED_MODEL):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build(self, chunks, batch_size=16):
        """Embed every chunk and add it to a flat inner-product index."""
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("cannot build an index with no chunks")
        texts = [chunk["text"] for chunk in self.chunks]
        vectors = self._load_model().encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        import faiss

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self

    def search(self, query, k=10):
        """Return the top k chunks for one question, best score first."""
        if self.index is None:
            raise RuntimeError("call build() before search()")
        vector = self._load_model().encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = np.ascontiguousarray(vector, dtype="float32")
        k = min(k, len(self.chunks))
        scores, indexes = self.index.search(vector, k)
        hits = []
        for score, index in zip(scores[0], indexes[0]):
            if int(index) < 0:
                continue
            chunk = self.chunks[int(index)]
            hits.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )
        return hits
