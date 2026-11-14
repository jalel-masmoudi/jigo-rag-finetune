"""Retrieval and answer metrics, including the bootstrap interval."""

from jigo_rag.generator import NOT_FOUND, build_messages
from jigo_rag.judge import fill_judge_prompt, parse_judge_score
from jigo_rag.metrics import (
    bootstrap_mean_ci,
    exact_match,
    mrr_at_k,
    percentile,
    recall_at_k,
    token_f1,
)
from eval import score_retrieval, summarize


def test_recall_and_mrr():
    ranked = ["a", "b", "c", "d", "gold"]
    assert recall_at_k(ranked, "gold", 1) == 0
    assert recall_at_k(ranked, "gold", 5) == 1
    assert mrr_at_k(ranked, "gold", 10) == 1 / 5
    assert mrr_at_k(ranked, "missing", 10) == 0
    assert mrr_at_k(["gold"], "gold", 10) == 1


def test_exact_match_and_token_f1():
    assert exact_match("Hello, World!", "hello world") == 1
    assert exact_match("not found", "Concur") == 0
    # two of three tokens overlap on each side: F1 is 2/3
    assert abs(token_f1("a b c", "b c d") - (2 / 3)) < 1e-9
    assert token_f1("", "") == 1
    assert token_f1("answer", "") == 0


def test_percentile_and_bootstrap_of_a_constant():
    # 4 values, 50th percentile lands on the third sorted value (index 2).
    assert percentile([1, 2, 3, 4], 50) == 3
    assert percentile([10], 95) == 10
    summary = bootstrap_mean_ci([1, 1, 1], n_resamples=50, seed=1)
    assert summary["mean"] == 1
    assert summary["ci_low"] == 1
    assert summary["ci_high"] == 1


def test_score_retrieval_uses_the_gold_id():
    hits = [{"chunk_id": "other#0"}, {"chunk_id": "gold#1"}, {"chunk_id": "gold#1"}]
    scores = score_retrieval("gold#1", hits)
    assert scores["recall@1"] == 0
    assert scores["recall@5"] == 1
    assert scores["mrr@10"] == 0.5
    report = summarize(
        [
            {"recall@1": 1, "recall@5": 1, "mrr@10": 1.0},
            {"recall@1": 0, "recall@5": 1, "mrr@10": 0.5},
        ],
        n_resamples=100,
        seed=42,
    )
    assert report["recall@5"]["mean"] == 1
    assert report["recall@1"]["mean"] == 0.5
    assert report["recall@1"]["ci_low"] <= 0.5 <= report["recall@1"]["ci_high"]


def test_generator_prompt_requires_not_found_and_numbers_passages():
    messages = build_messages("Where is travel booked?", ["Book it in Concur.", "Bring a card."])
    assert messages[0]["role"] == "system"
    assert NOT_FOUND in messages[0]["content"]
    assert "[1] Book it in Concur." in messages[1]["content"]
    assert "[2] Bring a card." in messages[1]["content"]
    assert "Where is travel booked?" in messages[1]["content"]


def test_judge_prompt_and_score_parser():
    filled = fill_judge_prompt(
        "Q {{QUESTION}} C {{CONTEXT}} R {{REFERENCE}} P {{PREDICTION}}",
        "q",
        "c",
        "r",
        "p",
    )
    assert filled == "Q q C c R r P p"
    assert parse_judge_score('{"score": 1, "reason": "same fact"}') == 1
    assert parse_judge_score('{"score": 0.5, "reason": "partial"}') == 0.5
    try:
        parse_judge_score('{"score": 2, "reason": "no"}')
    except ValueError:
        pass
    else:
        raise AssertionError("a score of 2 was accepted")
