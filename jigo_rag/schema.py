"""Shape of the JSONL rows.

A chunk row is one window of one document. A QA row is one question
whose answer is supposed to come from that window. `gold_chunk_id`
is the chunk id, so a retriever can be scored against it later.
"""

from jigo_rag.config import CHUNK_SIZE

CHUNK_FIELDS = (
    "chunk_id",
    "doc_id",
    "source",
    "doc_path",
    "text",
    "n_tokens",
    "split",
)
QA_FIELDS = (
    "row_id",
    "question",
    "answer",
    "gold_chunk_id",
    "doc_id",
    "chunk_text",
    "source",
    "split",
)
SPLITS = ("train", "val", "test")


def _require(row, fields, kind):
    missing = [name for name in fields if name not in row]
    if missing:
        raise ValueError(f"{kind} row is missing {missing}")


def _check_ids(chunk_id, doc_id):
    prefix, sep, number = chunk_id.rpartition("#")
    if sep != "#" or prefix != doc_id or not number.isdigit():
        raise ValueError(
            f"chunk id {chunk_id!r} must be {doc_id!r} plus '#' plus a window number"
        )


def validate_chunk(row):
    _require(row, CHUNK_FIELDS, "chunk")
    if row["split"] not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if not isinstance(row["text"], str) or not row["text"].strip():
        raise ValueError("chunk text is empty")
    if not isinstance(row["n_tokens"], int) or not 1 <= row["n_tokens"] <= CHUNK_SIZE:
        raise ValueError(f"n_tokens must be an int from 1 to {CHUNK_SIZE}")
    _check_ids(row["chunk_id"], row["doc_id"])
    if row["source"] not in row["doc_id"]:
        raise ValueError("doc_id must start with the source name")
    return row


def validate_qa(row):
    _require(row, QA_FIELDS, "qa")
    if row["split"] not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if not str(row["question"]).strip() or not str(row["answer"]).strip():
        raise ValueError("question and answer must be non-empty")
    if not str(row["chunk_text"]).strip():
        raise ValueError("chunk_text is empty")
    _check_ids(row["gold_chunk_id"], row["doc_id"])
    if not str(row["row_id"]).startswith(row["gold_chunk_id"] + "/q"):
        raise ValueError("row_id must start with the gold chunk id and '/q'")
    return row
