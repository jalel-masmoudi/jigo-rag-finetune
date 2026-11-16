"""Chunk windows stay within 512 tokens and share a 64-token overlap."""

import os

import pytest

from jigo_rag.chunk import chunk_text, chunk_token_ids, load_tokenizer
from jigo_rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def test_windows_cover_the_document_and_overlap():
    token_ids = list(range(2000))
    windows = chunk_token_ids(token_ids, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    assert windows
    assert all(len(window) <= CHUNK_SIZE for window in windows)
    assert len(windows[0]) == CHUNK_SIZE
    for left, right in zip(windows, windows[1:]):
        assert len(left) == CHUNK_SIZE
        assert left[-CHUNK_OVERLAP:] == right[:CHUNK_OVERLAP]
    # The step is size - overlap, so the tail of a long document is kept.
    assert windows[-1][-1] == token_ids[-1]
    covered = {token for window in windows for token in window}
    assert covered == set(token_ids)


def test_short_and_empty_documents():
    assert chunk_token_ids([]) == []
    short = list(range(10))
    assert chunk_token_ids(short) == [short]


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="CI does not download model files; this test loads BAAI/bge-small-en-v1.5 from Hugging Face",
)
def test_real_tokenizer_reports_the_cut_size():
    tokenizer = load_tokenizer()
    text = "Employees request travel in Concur before they leave. " * 400
    chunks = chunk_text(text, tokenizer)
    assert len(chunks) > 1
    assert all(1 <= chunk["n_tokens"] <= CHUNK_SIZE for chunk in chunks)
    # Stored text is a slice of the page, not a detokenized string.
    assert chunks[0]["text"] in text
    assert "Concur" in chunks[0]["text"]
    # Full windows are exactly the cap. The last window may be shorter.
    assert chunks[0]["n_tokens"] == CHUNK_SIZE
    ids = tokenizer.encode(text, add_special_tokens=False)
    windows = chunk_token_ids(ids)
    for chunk, window in zip(chunks, windows):
        assert chunk["n_tokens"] == len(window)
        assert chunk["text"]
