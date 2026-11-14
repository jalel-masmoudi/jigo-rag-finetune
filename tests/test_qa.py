"""Question planning, parsing, and the 'answer is in the chunk' check."""

from pathlib import Path

import pytest

from jigo_rag.config import DEFAULT_GEMINI_MODEL

from jigo_rag.qa import answer_supported, make_qa_row, parse_triples, plan_questions
from jigo_rag.schema import validate_qa
from review_test import apply_review


def test_plan_reaches_the_test_target_when_there_are_enough_chunks():
    per_chunk, enough = plan_questions(n_chunks=200, target=360, cap=4)
    assert enough
    assert per_chunk == 2
    assert 200 * per_chunk >= 360


def test_plan_reports_a_short_split():
    per_chunk, enough = plan_questions(n_chunks=50, target=360, cap=4)
    assert per_chunk == 4
    assert not enough


def test_parse_triples_accepts_a_fenced_list():
    text = """```json
[{"question": "Where is travel booked?", "answer": "In Concur."}]
```"""
    triples = parse_triples(text)
    assert triples == [{"question": "Where is travel booked?", "answer": "In Concur."}]


def test_answer_must_be_in_the_chunk():
    chunk = "Book travel in Concur before the trip starts."
    assert answer_supported("Book travel in Concur before the trip starts.", chunk)
    assert not answer_supported("The office is closed on federal holidays.", chunk)


def test_prompt_file_and_env_example():
    prompt = Path("prompts/qa_generate.txt").read_text(encoding="utf-8")
    assert "{{N_QUESTIONS}}" in prompt
    assert "{{CHUNK_ID}}" in prompt
    assert "{{CHUNK_TEXT}}" in prompt
    assert "this passage alone" in prompt
    example = Path(".env.example").read_text(encoding="utf-8")
    assert f"GEMINI_MODEL={DEFAULT_GEMINI_MODEL}" in example
    assert "GEMINI_API_KEY=\n" in example
    # The example file must not ship a real key.
    assert "AIza" not in example


def test_review_keep_fix_and_drop():
    row = make_qa_row(
        {
            "chunk_id": "jigo/README.md#0",
            "doc_id": "jigo/README.md",
            "text": "Jigo syncs notes from Slack.",
            "source": "jigo",
            "split": "test",
        },
        "What does Jigo sync from Slack?",
        "notes",
        0,
    )
    validate_qa(row)
    kept = apply_review(row, "k")
    assert kept["review_action"] == "keep"
    assert kept["answer"] == "notes"
    fixed = apply_review(row, "fix", new_answer="Jigo syncs notes from Slack.")
    assert fixed["review_action"] == "fix"
    assert fixed["question"] == row["question"]
    assert "Slack" in fixed["answer"]
    assert apply_review(row, "d") is None
    with pytest.raises(ValueError):
        apply_review(row, "maybe")
