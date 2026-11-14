"""Retrieval and answer metrics, including the bootstrap interval."""

from jigo_rag.generator import NOT_FOUND, build_messages
from jigo_rag.judge import fill_judge_prompt, judge_with_retries, parse_judge_score
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
    assert exact_match("The cat.", "cat") == 1
    assert token_f1("the cat", "cat") == 1.0
    assert token_f1("don't", "dont") == 1
    from jigo_rag.metrics import normalize

    assert normalize("U.S.") == "us"
    assert normalize("A-1") == "a1"
    # café is a letter, not punctuation, so it survives normalization.
    assert exact_match("Café!", "café") == 1
    assert exact_match("not found", "Concur") == 0
    # two of three content tokens overlap: F1 is 2/3
    assert abs(token_f1("alpha beta gamma", "beta gamma delta") - (2 / 3)) < 1e-9
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


def test_unparseable_judge_replies_are_not_retried():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        # The digits are text in the reply, not an HTTP status.
        raise ValueError("not json, server said 500")

    try:
        judge_with_retries(bad, attempts=3)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("a bad judge reply was accepted")
    assert calls["n"] == 1
    assert "could not be parsed" in message
    assert "judge request failed" not in message


def test_permission_errors_are_not_retried():
    calls = {"n": 0}

    def denied():
        calls["n"] += 1
        raise PermissionError("denied")

    try:
        judge_with_retries(denied, attempts=3)
    except PermissionError:
        pass
    else:
        raise AssertionError("a permission error was retried")
    assert calls["n"] == 1


def test_http_server_errors_are_retried():
    calls = {"n": 0}

    def down():
        calls["n"] += 1
        error = OSError("unavailable")
        error.code = 503
        raise error

    try:
        judge_with_retries(down, attempts=2)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("an HTTP 503 was accepted")
    assert calls["n"] == 2
    assert "judge request failed" in message
    assert "could not be parsed" not in message


def test_network_errors_are_retried():
    calls = {"n": 0}

    def down():
        calls["n"] += 1
        raise TimeoutError("timed out")

    try:
        judge_with_retries(down, attempts=2)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("a network failure was accepted")
    assert calls["n"] == 2
    assert "judge request failed" in message
    assert "could not be parsed" not in message


def test_rate_limit_then_success_returns_the_score(monkeypatch):
    from google.genai.errors import ClientError

    slept = []
    monkeypatch.setattr("jigo_rag.judge.time.sleep", lambda seconds: slept.append(seconds))
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] == 1:
            response = type("Response", (), {"headers": {"Retry-After": "7"}, "status_code": 429})()
            raise ClientError(429, {"error": {"message": "rate limit"}}, response)
        return 1.0

    assert judge_with_retries(call, attempts=3) == 1.0
    assert calls["n"] == 2
    assert slept == [7.0]


def test_persistent_rate_limit_skips_the_judge_and_the_loop_continues(monkeypatch):
    from google.genai.errors import ClientError

    from eval import run

    monkeypatch.setattr("jigo_rag.judge.time.sleep", lambda seconds: None)
    seen = {"n": 0}

    def judge(question, passages, reference, prediction):
        seen["n"] += 1

        def call():
            if seen["n"] == 1:
                raise ClientError(429, {"error": {"message": "rate limit"}}, None)
            return 1.0

        return judge_with_retries(call, attempts=2)

    class Retriever:
        chunks = [None, None]

        def search(self, query, k=10):
            return [{"chunk_id": "doc#0", "text": "Book travel in Concur.", "doc_id": "doc"}]

    class Generator:
        def answer(self, question, passages, max_new_tokens=128):
            return {"answer": "In Concur.", "prompt_tokens": 1, "completion_tokens": 1}

    summary = run(
        [
            {"question": "Where?", "answer": "In Concur.", "gold_chunk_id": "doc#0"},
            {"question": "Also?", "answer": "In Concur.", "gold_chunk_id": "doc#0"},
        ],
        Retriever(),
        Generator(),
        judge,
        smoke=True,
        n_resamples=10,
        seed=1,
    )
    assert summary["n"] == 2
    assert summary["judge"]["skipped"] == 1
    assert summary["judge"]["mean"] == 1.0
    assert seen["n"] == 2


def test_not_found_is_zero_end_to_end_and_separate_if_retrieval_missed():
    from jigo_rag.judge import end_to_end_not_found, not_found_score

    assert end_to_end_not_found("not found") == 0.0
    assert end_to_end_not_found("Not found.") == 0.0
    assert end_to_end_not_found("In Concur.") is None
    # The retrieval-conditioned column is not the table judge.
    assert not_found_score("not found", "Book travel in Concur.", ["The office is closed."]) == 1.0
    assert not_found_score("Not found.", "Book travel in Concur.", ["Book travel in Concur before the trip."]) == 0.0


def test_latency_does_not_include_the_judge():
    import time

    from eval import run

    class Retriever:
        chunks = [None]

        def search(self, query, k=10):
            return [{"chunk_id": "doc#0", "text": "Book travel in Concur.", "doc_id": "doc"}]

    class Generator:
        def answer(self, question, passages, max_new_tokens=128):
            return {"answer": "somewhere else", "prompt_tokens": 1, "completion_tokens": 1}

    def slow_judge(question, passages, reference, prediction):
        time.sleep(0.2)
        return 0.0

    summary = run(
        [{"question": "Where?", "answer": "In Concur.", "gold_chunk_id": "doc#0"}],
        Retriever(),
        Generator(),
        slow_judge,
        smoke=True,
        n_resamples=10,
        seed=1,
    )
    assert summary["latency_ms"]["p50"] < 150


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
    for bad in ('{"score": 2, "reason": "no"}', '{"score": true, "reason": "no"}'):
        try:
            parse_judge_score(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
