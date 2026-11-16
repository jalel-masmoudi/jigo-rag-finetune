"""Retrieval and answer metrics, including the bootstrap interval."""

from jigo_rag.generator import NOT_FOUND, build_messages
from jigo_rag.judge import fill_judge_prompt, judge_with_retries, parse_judge_score
from jigo_rag.metrics import (
    bootstrap_mean_ci,
    exact_match,
    mrr_at_k,
    paired_bootstrap_diff,
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


def test_base_rows_use_the_disabled_adapter_on_one_model(monkeypatch):
    import eval as ev

    calls = []

    class Shared:
        def __init__(self, adapter_path=None):
            calls.append(("init", adapter_path))

        def answer(self, question, passages, max_new_tokens=128, disable_adapter=False):
            calls.append(("answer", disable_adapter, max_new_tokens))
            return {"answer": "base" if disable_adapter else "lora", "prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(ev, "QwenGenerator", Shared)
    base, lora = ev.suite_generators("models/lora", retrieval_only=False)
    assert calls == [("init", "models/lora")]
    assert base.answer("q", ["p"])["answer"] == "base"
    assert lora.answer("q", ["p"], max_new_tokens=128)["answer"] == "lora"
    assert calls[1:] == [("answer", True, 128), ("answer", False, 128)]
    plain, no_lora = ev.suite_generators(None, retrieval_only=True)
    assert plain is None and no_lora is None


def test_suite_limit_is_marked_smoke():
    from eval import suite_is_smoke

    assert suite_is_smoke(5) is True
    assert suite_is_smoke(0) is False


def test_limit_marks_retrieval_only_and_every_suite_row(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    import sentence_transformers

    import eval as ev

    chunks = [
        {
            "chunk_id": "a#0",
            "doc_id": "a",
            "text": "Book travel in Concur before the trip.",
            "split": "test",
        }
    ]
    reviewed = [
        {"question": "Where?", "answer": "In Concur.", "gold_chunk_id": "a#0"},
        {"question": "When?", "answer": "Before the trip.", "gold_chunk_id": "a#0"},
    ]
    monkeypatch.setattr(ev, "load_corpus", lambda: chunks)
    monkeypatch.setattr(ev, "load_reviewed_test", lambda: reviewed)
    monkeypatch.setattr(ev, "_present", lambda path: Path("models/embed"))

    class Retriever:
        def __init__(self, model_name=None):
            self.chunks = []

        def build(self, found):
            self.chunks = list(found)
            return self

        def search(self, query, k=10):
            return [{"chunk_id": "a#0", "text": chunks[0]["text"], "doc_id": "a", "score": 0.5}]

    class Generator:
        def __init__(self, *args, **kwargs):
            return None

        def answer(self, question, passages, max_new_tokens=128):
            return {"answer": "In Concur.", "prompt_tokens": 2, "completion_tokens": 1}

    class Cross:
        def __init__(self, *args, **kwargs):
            return None

        def predict(self, pairs, show_progress_bar=False):
            return [0.2 for _pair in pairs]

    monkeypatch.setattr(ev, "FaissRetriever", Retriever)
    monkeypatch.setattr(ev, "QwenGenerator", Generator)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Cross)

    retrieval_out = tmp_path / "retrieval.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval.py", "--retrieval-only", "--limit", "1", "--bootstrap", "5", "--output", str(retrieval_out)],
    )
    ev.main()
    retrieval = json.loads(retrieval_out.read_text(encoding="utf-8"))
    assert retrieval["smoke"] is True
    assert retrieval["n"] == 1

    suite_out = tmp_path / "table.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval.py", "--suite", "--retrieval-only", "--limit", "1", "--bootstrap", "5", "--output", str(suite_out)],
    )
    ev.main()
    table = json.loads(suite_out.read_text(encoding="utf-8"))
    assert table["smoke"] is True
    for name, row in table["rows"].items():
        assert row is not None, name
        assert row["smoke"] is True, name


def test_suite_plan_covers_the_full_pipeline_only_when_every_weight_exists():
    from eval import planned_rows

    assert planned_rows(False, False, False) == ["baseline"]
    assert planned_rows(True, True, True) == [
        "baseline",
        "finetuned_embeddings",
        "reranker",
        "lora_generator",
        "full",
    ]
    assert "reranker" not in planned_rows(False, True, True)
    assert "full" not in planned_rows(True, False, True)


def test_paired_bootstrap_is_the_gain_on_the_same_questions():
    baseline = [0, 0, 1, 1]
    new = [1, 1, 1, 1]
    summary = paired_bootstrap_diff(baseline, new, n_resamples=200, seed=1)
    assert summary["mean_gain"] == 0.5
    assert summary["ci_low"] <= 0.5 <= summary["ci_high"]
    # Identical systems have a gain of zero, so the interval is a point.
    flat = paired_bootstrap_diff([0.5, 0.5], [0.5, 0.5], n_resamples=20, seed=1)
    assert flat["mean_gain"] == 0
    assert flat["ci_low"] == 0
    assert flat["ci_high"] == 0


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


def test_retry_after_is_capped_and_non_finite_values_use_backoff():
    from jigo_rag.judge import _retry_delay

    def delay(raw, attempt=0):
        response = type("Response", (), {"headers": {"Retry-After": raw}})()
        error = type("Error", (), {"response": response})()
        return _retry_delay(error, attempt)

    assert delay("86400") == 60.0
    assert delay("7") == 7.0
    # attempt 0 waits 2**0 seconds when the header cannot be used.
    assert delay("inf") == 1.0
    assert delay("nan") == 1.0
    assert delay("-5") == 1.0
    assert delay("soon") == 1.0
    assert delay("inf", attempt=3) == 8.0


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


def test_lora_summary_keeps_the_retrieval_conditioned_score():
    from eval import run

    class Retriever:
        chunks = [None]

        def search(self, query, k=10):
            return [{"chunk_id": "doc#0", "text": "The office is closed.", "doc_id": "doc"}]

    class Generator:
        def answer(self, question, passages, max_new_tokens=128):
            return {"answer": "not found", "prompt_tokens": 1, "completion_tokens": 1}

    summary = run(
        [{"question": "Where?", "answer": "In Concur.", "gold_chunk_id": "doc#0"}],
        Retriever(),
        Generator(),
        lambda question, passages, reference, prediction: 1.0,
        smoke=False,
        n_resamples=10,
        seed=1,
        lora_generator=Generator(),
    )
    assert summary["lora"]["judge"]["mean"] == 0.0
    conditioned = summary["lora"]["lora_retrieval_conditioned_judge"]
    assert conditioned["mean"] == 1.0
    assert "Not-found" in conditioned["label"]


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
