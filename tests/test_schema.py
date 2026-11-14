"""JSONL rows have the fields the rest of the pipeline reads."""

import pytest

from jigo_rag.config import SOURCES
from jigo_rag.jsonl import read_jsonl, write_jsonl
from jigo_rag.qa import answer_supported
from jigo_rag.schema import validate_chunk, validate_qa


def _chunk(**overrides):
    row = {
        "chunk_id": "gsa-handbook/pages/travel.md#0",
        "doc_id": "gsa-handbook/pages/travel.md",
        "source": "gsa-handbook",
        "doc_path": "pages/travel.md",
        "text": "Book travel in Concur before the trip starts.",
        "n_tokens": 12,
        "split": "train",
    }
    row.update(overrides)
    return row


def _qa(**overrides):
    row = {
        "row_id": "gsa-handbook/pages/travel.md#0/q0",
        "question": "Where is travel booked?",
        "answer": "Book travel in Concur before the trip starts.",
        "gold_chunk_id": "gsa-handbook/pages/travel.md#0",
        "doc_id": "gsa-handbook/pages/travel.md",
        "chunk_text": "Book travel in Concur before the trip starts.",
        "source": "gsa-handbook",
        "split": "train",
    }
    row.update(overrides)
    return row


def test_valid_rows_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [validate_chunk(_chunk()), validate_qa(_qa())])
    loaded = read_jsonl(path)
    assert loaded[0]["chunk_id"].endswith("#0")
    assert loaded[1]["gold_chunk_id"] == loaded[0]["chunk_id"]


@pytest.mark.parametrize(
    "bad",
    [
        {"n_tokens": 513},
        {"n_tokens": 0},
        {"split": "holdout"},
        {"chunk_id": "other#0"},
        {"text": "   "},
    ],
)
def test_chunk_schema_rejects_bad_rows(bad):
    with pytest.raises(ValueError):
        validate_chunk(_chunk(**bad))


def test_qa_schema_rejects_a_mismatched_gold_id():
    with pytest.raises(ValueError):
        validate_qa(_qa(gold_chunk_id="gsa-handbook/pages/other.md#1"))


def test_sample_files_match_the_schema_and_do_not_leak():
    from pathlib import Path

    from jigo_rag.split import assert_no_doc_leakage

    rows = []
    sample_dir = Path("data/sample")
    for split in ("train", "val", "test"):
        file_rows = read_jsonl(sample_dir / f"{split}.jsonl")
        assert file_rows, f"data/sample/{split}.jsonl is empty"
        for row in file_rows:
            assert row["split"] == split
            validate_qa(row)
            assert row["answer"] in row["chunk_text"]
            assert answer_supported(row["answer"], row["chunk_text"])
            rows.append(row)
    assert 15 <= len(rows) <= 30
    assert_no_doc_leakage(rows)


def test_data_md_names_the_pinned_commits():
    from pathlib import Path

    text = Path("DATA.md").read_text(encoding="utf-8")
    for source in SOURCES:
        assert source["sha"] in text
        assert source["url"] in text
