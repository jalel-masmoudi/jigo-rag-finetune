"""Each automatic test-set rule, and the refusal to overwrite."""

from jigo_rag.jsonl import read_jsonl
from auto_filter_test import drop_reason, filter_rows, write_filter


def _row(**overrides):
    row = {
        "row_id": "gsa-handbook/pages/travel.md#0/q0",
        "question": "Where is travel booked?",
        "answer": "Book travel in Concur",
        "gold_chunk_id": "gsa-handbook/pages/travel.md#0",
        "doc_id": "gsa-handbook/pages/travel.md",
        "chunk_text": "Book travel in Concur before the trip starts.",
        "source": "gsa-handbook",
        "split": "test",
    }
    row.update(overrides)
    return row


def test_normalized_answer_must_be_a_substring_of_the_chunk():
    # Punctuation is deleted, so "U.S." matches "us" inside the chunk.
    kept, dropped, counts = filter_rows(
        [
            _row(answer="U.S. office", chunk_text="The U.S. office opens at nine."),
            _row(
                row_id="gsa-handbook/pages/travel.md#0/q1",
                question="When is the office closed?",
                answer="federal holidays",
            ),
        ]
    )
    assert len(kept) == 1
    assert kept[0]["review_action"] == "auto_keep"
    assert dropped[0]["drop_reason"] == "answer_not_in_chunk"
    assert dropped[0]["review_action"] == "auto_drop"
    assert counts["answer_not_in_chunk"] == 1


def test_duplicate_normalized_questions_in_one_chunk_are_dropped():
    first = _row(question="Where is travel booked?")
    second = _row(
        row_id="gsa-handbook/pages/travel.md#0/q1",
        question="Where is travel booked!",
    )
    other_chunk = _row(
        row_id="gsa-handbook/pages/leave.md#0/q0",
        question="Where is travel booked?",
        gold_chunk_id="gsa-handbook/pages/leave.md#0",
        doc_id="gsa-handbook/pages/leave.md",
        chunk_text="Book travel in Concur before the trip starts.",
    )
    kept, dropped, counts = filter_rows([first, second, other_chunk])
    assert [row["row_id"] for row in kept] == [first["row_id"], other_chunk["row_id"]]
    assert dropped[0]["drop_reason"] == "duplicate_question"
    assert counts["duplicate_question"] == 1


def test_questions_that_refer_to_the_text_are_dropped():
    questions = [
        "What does the passage say about Concur?",
        "What does THIS TEXT say about Concur?",
        "What does the above say about Concur?",
        "What does this section say about Concur?",
        "What does the document say about Concur?",
        "What does the chunk say about Concur?",
    ]
    rows = [
        _row(row_id=f"gsa-handbook/pages/travel.md#0/q{index}", question=question)
        for index, question in enumerate(questions)
    ]
    # "documentary" contains "document" but is not the phrase.
    rows.append(
        _row(
            row_id="gsa-handbook/pages/travel.md#0/q9",
            question="Where is the documentary booked in Concur?",
        )
    )
    kept, dropped, counts = filter_rows(rows)
    assert len(kept) == 1
    assert kept[0]["row_id"].endswith("q9")
    assert counts["refers_to_text"] == 6
    assert {row["drop_reason"] for row in dropped} == {"refers_to_text"}


def test_answers_over_40_words_are_dropped_and_40_are_kept():
    short = " ".join(["word"] * 40)
    long = " ".join(["word"] * 41)
    chunk = short + " and then one more word."
    kept, dropped, counts = filter_rows(
        [
            _row(answer=short, chunk_text=chunk),
            _row(
                row_id="gsa-handbook/pages/travel.md#0/q1",
                question="What is the long form?",
                answer=long,
                chunk_text=chunk,
            ),
        ]
    )
    assert len(kept) == 1
    assert kept[0]["answer"] == short
    assert dropped[0]["drop_reason"] == "answer_too_long"
    assert counts["answer_too_long"] == 1


def test_a_row_that_fails_the_schema_is_dropped():
    broken = _row()
    del broken["gold_chunk_id"]
    kept, dropped, counts = filter_rows([broken, _row()])
    assert len(kept) == 1
    assert dropped[0]["drop_reason"] == "invalid_qa"
    assert counts["invalid_qa"] == 1
    assert drop_reason(broken, set()) == "invalid_qa"


def test_existing_reviewed_file_is_kept_unless_force(tmp_path):
    source = [_row()]
    kept_path = tmp_path / "test_reviewed.jsonl"
    dropped_path = tmp_path / "test_dropped.jsonl"
    kept_path.write_text('{"row_id": "already"}\n', encoding="utf-8")
    try:
        write_filter(source, kept_path, dropped_path, force=False)
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("an existing reviewed file was overwritten")
    assert "already exists" in message
    assert kept_path.read_text(encoding="utf-8") == '{"row_id": "already"}\n'
    assert not dropped_path.exists()

    write_filter(source, kept_path, dropped_path, force=True)
    written = read_jsonl(kept_path)
    assert written[0]["review_action"] == "auto_keep"
    assert read_jsonl(dropped_path) == []
