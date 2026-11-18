"""Build the test set with a fixed filter. No API calls.

Reads data/test.jsonl, the raw Gemini output. Writes the rows that
pass to data/test_filtered.jsonl and the rest to data/test_dropped.jsonl.
This is not a hand review. The rules below are the whole decision.

A row is dropped for the first reason that applies:

1. It does not pass validate_qa (`invalid_qa`).
2. Its split is not "test" (`wrong_split`).
3. The answer is more than 40 words (`answer_too_long`).
4. The question refers to the text itself (`refers_to_text`). The match
   is a whole phrase, such as "the passage" or "this document", not a
   bare word. "document retention period" is kept.
5. After the same SQuAD normalization used for exact match, the answer
   is not a whole-word substring of the chunk (`answer_not_in_chunk`).
   "30" does not match "300".
6. The same chunk already kept a question with the same normalized
   text (`duplicate_question`).

Kept rows get review_action "auto_keep". Dropped rows get
review_action "auto_drop" and a drop_reason. An existing
data/test_filtered.jsonl is left alone unless --force is passed.

    python auto_filter_test.py
    python auto_filter_test.py --force
"""

import argparse
import re
from pathlib import Path

from jigo_rag.jsonl import read_jsonl, write_jsonl
from jigo_rag.metrics import normalize
from jigo_rag.schema import validate_qa

TEST_PATH = Path("data/test.jsonl")
KEPT_PATH = Path("data/test_filtered.jsonl")
DROPPED_PATH = Path("data/test_dropped.jsonl")

# Printed in this order, including zeros, so a run is easy to compare.
DROP_REASONS = (
    "invalid_qa",
    "wrong_split",
    "answer_too_long",
    "refers_to_text",
    "answer_not_in_chunk",
    "duplicate_question",
)

MAX_ANSWER_WORDS = 40

# Whole phrases only. A bare word such as "passage" or "document" is not
# enough, and a longer name such as "document retention" is not either.
_SELF_REFERENCE = (
    r"according to the text",
    r"as mentioned above",
    r"the text above",
    r"this passage",
    r"this text",
    r"this section",
    r"this document",
    r"the article",
    r"the excerpt",
    r"the passage(?!\s+of\b)",
    r"the above(?!\s+and\s+beyond\b)",
    r"the document(?!\s+retention\b)",
    r"the chunk",
)
_SELF_REFERENCE_RE = re.compile(
    "|".join(rf"\b{phrase}\b" for phrase in _SELF_REFERENCE),
    re.IGNORECASE,
)


def answer_in_chunk(answer, chunk_text):
    """True when the normalized answer is a whole span of the chunk.

    Normalization is the SQuAD function in jigo_rag.metrics, the same one
    exact match uses. Spaces are added on both sides so "30" does not
    match inside "300". An answer that normalizes to nothing does not match.
    """
    normalized_answer = normalize(answer)
    if not normalized_answer:
        return False
    return f" {normalized_answer} " in f" {normalize(chunk_text)} "


def refers_to_text(question):
    """True when the question uses one of the self-reference phrases."""
    return _SELF_REFERENCE_RE.search(question) is not None


def drop_reason(row, seen):
    """Return a drop reason, or None when the row is kept.

    `seen` holds (gold_chunk_id, normalized question) for rows already
    kept. A later copy in that chunk is a duplicate. A copy in another
    chunk is not.
    """
    try:
        validate_qa(row)
    except (TypeError, ValueError, KeyError):
        return "invalid_qa"
    if row.get("split") != "test":
        return "wrong_split"
    if len(str(row["answer"]).split()) > MAX_ANSWER_WORDS:
        return "answer_too_long"
    if refers_to_text(str(row["question"])):
        return "refers_to_text"
    if not answer_in_chunk(str(row["answer"]), str(row["chunk_text"])):
        return "answer_not_in_chunk"
    key = (row["gold_chunk_id"], normalize(str(row["question"])))
    if key in seen:
        return "duplicate_question"
    seen.add(key)
    return None


def filter_rows(rows):
    """Split rows into kept, dropped, and a count per drop reason."""
    seen = set()
    kept = []
    dropped = []
    counts = {reason: 0 for reason in DROP_REASONS}
    for row in rows:
        reason = drop_reason(row, seen)
        if reason is None:
            kept_row = dict(row)
            kept_row["review_action"] = "auto_keep"
            kept.append(kept_row)
            continue
        counts[reason] += 1
        dropped_row = dict(row)
        dropped_row["review_action"] = "auto_drop"
        dropped_row["drop_reason"] = reason
        dropped.append(dropped_row)
    return kept, dropped, counts


def write_filter(rows, kept_path, dropped_path, force):
    """Write both files. Refuse to replace an existing kept file."""
    kept_path = Path(kept_path)
    dropped_path = Path(dropped_path)
    if kept_path.exists() and not force:
        raise SystemExit(
            f"{kept_path} already exists. Pass --force to overwrite it. "
            "The file was not changed."
        )
    kept, dropped, counts = filter_rows(rows)
    write_jsonl(kept_path, kept)
    write_jsonl(dropped_path, dropped)
    return kept, dropped, counts


def print_counts(kept, counts):
    print(f"kept {len(kept)}")
    for reason in DROP_REASONS:
        print(f"{reason} {counts[reason]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite an existing test_filtered.jsonl")
    args = parser.parse_args()
    if not TEST_PATH.exists():
        raise SystemExit("data/test.jsonl does not exist. Run python generate_qa.py first.")
    kept, _dropped, counts = write_filter(read_jsonl(TEST_PATH), KEPT_PATH, DROPPED_PATH, args.force)
    print_counts(kept, counts)


if __name__ == "__main__":
    main()
