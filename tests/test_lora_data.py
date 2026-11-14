"""Not-found examples and the prompt mask. No model is loaded."""

from jigo_rag.generator import NOT_FOUND
from jigo_rag.lora_data import build_examples, mask_prompt_labels


def test_every_question_gets_an_answer_and_a_not_found():
    rows = [
        {
            "question": "Where is travel booked?",
            "answer": "In Concur.",
            "chunk_text": "Book travel in Concur.",
            "doc_id": "doc-a",
        }
    ]
    corpus = [
        {"chunk_id": "doc-a#0", "doc_id": "doc-a", "text": "Book travel in Concur."},
        {"chunk_id": "doc-b#0", "doc_id": "doc-b", "text": "Leave is requested separately."},
    ]
    examples = build_examples(rows, corpus, seed=1)
    assert [example["kind"] for example in examples] == ["answer", "not_found"]
    answer = examples[0]["messages"][-1]["content"]
    refusal = examples[1]["messages"][-1]["content"]
    assert answer == "In Concur."
    assert refusal == NOT_FOUND
    # The not-found context is the other document, not the gold page.
    assert "Leave is requested separately." in examples[1]["messages"][1]["content"]
    assert "Concur" not in examples[1]["messages"][1]["content"]


def test_mask_hides_the_prompt_and_keeps_the_answer():
    labels = mask_prompt_labels([1, 2, 3, 4, 5], prompt_length=3)
    assert labels == [-100, -100, -100, 4, 5]
    try:
        mask_prompt_labels([1, 2], prompt_length=2)
    except ValueError:
        pass
    else:
        raise AssertionError("a fully masked sequence was accepted")
