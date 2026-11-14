"""Hand-check test questions, one at a time.

Only rows you keep or fix are written to data/test_reviewed.jsonl.
That file is the test set. data/test.jsonl is the raw model output
and is not used for scores. Dropped rows go to data/test_dropped.jsonl
so a later session does not show them again.

    python review_test.py

Keys: k keep, f fix, d drop, q quit. Progress is saved after each row.
"""

from pathlib import Path

from jigo_rag.jsonl import append_jsonl, read_jsonl
from jigo_rag.schema import validate_qa

TEST_PATH = Path("data/test.jsonl")
KEPT_PATH = Path("data/test_reviewed.jsonl")
DROPPED_PATH = Path("data/test_dropped.jsonl")


def apply_review(row, action, new_question="", new_answer=""):
    """Return the reviewed row, or None when the row is dropped.

    `action` is k/keep, f/fix, or d/drop. A fix with a blank field
    leaves that field as it was.
    """
    action = action.strip().lower()
    reviewed = dict(row)
    if action in ("k", "keep"):
        reviewed["review_action"] = "keep"
        return reviewed
    if action in ("d", "drop"):
        reviewed["review_action"] = "drop"
        return None
    if action in ("f", "fix"):
        if new_question.strip():
            reviewed["question"] = new_question.strip()
        if new_answer.strip():
            reviewed["answer"] = new_answer.strip()
        reviewed["review_action"] = "fix"
        return reviewed
    raise ValueError("action must be keep, fix, or drop")


def pending_rows():
    if not TEST_PATH.exists():
        raise SystemExit("data/test.jsonl does not exist. Run python generate_qa.py first.")
    rows = read_jsonl(TEST_PATH)
    seen = set()
    for path in (KEPT_PATH, DROPPED_PATH):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            seen.add(row["row_id"])
    return [row for row in rows if row["row_id"] not in seen]


def prompt_line(label):
    try:
        return input(label)
    except EOFError:
        return "q"


def review_one(row):
    print()
    print(f"id: {row['row_id']}")
    print(f"question: {row['question']}")
    print(f"answer: {row['answer']}")
    print("chunk:")
    print(row["chunk_text"])
    print()
    choice = prompt_line("[k]eep  [f]ix  [d]rop  [q]uit: ").strip().lower()
    if choice in ("q", "quit", ""):
        return "quit", None
    if choice in ("f", "fix"):
        new_question = prompt_line("New question (blank keeps the current one): ")
        new_answer = prompt_line("New answer (blank keeps the current one): ")
        return "fix", apply_review(row, "fix", new_question, new_answer)
    if choice in ("k", "keep", "d", "drop"):
        return choice, apply_review(row, choice)
    print("Type k, f, d, or q.")
    return review_one(row)


def main():
    rows = pending_rows()
    print(f"{len(rows)} test rows left to review. Kept rows go to {KEPT_PATH}.")
    for index, row in enumerate(rows, start=1):
        print(f"\n--- {index}/{len(rows)} ---")
        action, reviewed = review_one(row)
        if action == "quit":
            print("Stopped. Rerun to continue.")
            return
        if reviewed is None:
            dropped = dict(row)
            dropped["review_action"] = "drop"
            append_jsonl(DROPPED_PATH, [dropped])
            print("dropped")
            continue
        validate_qa(reviewed)
        append_jsonl(KEPT_PATH, [reviewed])
        print(reviewed["review_action"])
    print("No test rows left.")


if __name__ == "__main__":
    main()
