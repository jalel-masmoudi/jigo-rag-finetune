"""LLM judge for answer correctness.

The rubric and the wording live in prompts/judge.txt. The judge is
shown the passages the retriever actually returned, not the gold chunk.
The score is 0, 0.5, or 1, and it must be a number. A JSON boolean is
rejected. The model is the same free-tier Gemini default as the
question generator. If the key is missing the judge is skipped and the
score is left empty. A missing key is not a zero.

Every question in the test set is answerable, so a reply of "not found"
scores 0 on the end-to-end judge. not_found_score is a separate value:
1 when the retrieved passages do not contain the reference, and 0 when
they do. That value is not the judge score.
"""

import json
import math
import os
import re
import time

from jigo_rag.config import DEFAULT_GEMINI_MODEL


def fill_judge_prompt(template, question, context, reference, prediction):
    required = ("{{QUESTION}}", "{{CONTEXT}}", "{{REFERENCE}}", "{{PREDICTION}}")
    missing = [token for token in required if token not in template]
    if missing:
        raise ValueError(f"judge prompt is missing {missing}")
    return (
        template.replace("{{QUESTION}}", question)
        .replace("{{CONTEXT}}", context)
        .replace("{{REFERENCE}}", reference)
        .replace("{{PREDICTION}}", prediction)
    )


def parse_judge_score(text):
    """Pull a numeric score of 0, 0.5, or 1 out of a JSON object.

    `true` is rejected. In Python `True == 1`, so a boolean would
    otherwise sneak through the membership test.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    score = data["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"judge score must be a number, got {score!r}")
    if float(score) not in (0.0, 0.5, 1.0):
        raise ValueError(f"judge score must be 0, 0.5, or 1, got {score!r}")
    return float(score)


def format_passages(passages):
    if not passages:
        return "(no passages were retrieved)"
    return "\n\n".join(f"[{index}] {text}" for index, text in enumerate(passages, start=1))


def end_to_end_not_found(prediction):
    """Return 0.0 when the reply is 'not found', else None.

    Every question in the test set has an answer, so refusing is wrong
    on the end-to-end judge. This does not look at retrieval.
    """
    from jigo_rag.metrics import normalize

    if normalize(prediction) == "not found":
        return 0.0
    return None


def not_found_score(prediction, reference, passages):
    """Retrieval-conditioned score for a 'not found' reply.

    1.0 when the retrieved passages do not contain the reference, 0.0
    when they do. None when the reply is not 'not found'. This is not
    the end-to-end judge.
    """
    from jigo_rag.metrics import normalize
    from jigo_rag.qa import answer_supported

    if normalize(prediction) != "not found":
        return None
    supported = any(answer_supported(reference, passage) for passage in passages)
    return 0.0 if supported else 1.0


# httpx and requests raise their own connection and timeout types. They are
# not the builtin ConnectionError, and the message text is not a status code.
_NETWORK_TYPES = {
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "NetworkError",
    "RemoteProtocolError",
    "Timeout",
    "ProxyError",
}


def _status_code(exc):
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return None


_GEMINI_API_ERRORS = {"APIError", "ClientError", "ServerError"}


def _gemini_api_error(exc):
    return any(cls.__name__ in _GEMINI_API_ERRORS for cls in type(exc).__mro__)


def _retryable(exc):
    """Network failures, HTTP 429, and HTTP 5xx.

    PermissionError is an OSError, and a reply that merely contains the
    digits 500 is not an HTTP error. Neither is retried. A 429 is, because
    the free-tier judge is rate limited.
    """
    if isinstance(exc, PermissionError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if any(cls.__name__ in _NETWORK_TYPES for cls in type(exc).__mro__):
        return True
    status = _status_code(exc)
    if status == 429:
        return True
    return status is not None and 500 <= status <= 599


def _retry_delay(exc, attempt):
    """Seconds to wait.

    A finite Retry-After is used, and never more than 60 seconds.
    A negative, non-finite, or unparseable value falls back to the
    exponential wait. `inf` must not reach time.sleep, which raises
    OverflowError on it.
    """
    fallback = float(min(2 ** attempt, 8))
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw = None
    if headers is not None and hasattr(headers, "get"):
        raw = headers.get("Retry-After")
        if raw is None:
            raw = headers.get("retry-after")
    if raw is None:
        return fallback
    try:
        delay = float(raw)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(delay) or delay < 0:
        return fallback
    return min(delay, 60.0)


def judge_failure_skips(exc):
    """A parse failure or a Gemini API error is recorded, not fatal."""
    return isinstance(exc, ValueError) or _gemini_api_error(exc)


def judge_with_retries(call, attempts=3):
    """Retry network errors, HTTP 429, and HTTP 5xx.

    A bad JSON reply is a parse failure. A rate limit or a dropped
    connection is a request failure. The two messages stay different so
    a log can tell them apart.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            return call()
        except ValueError as exc:
            raise ValueError(f"judge reply could not be parsed: {exc}") from exc
        except Exception as exc:
            if not _retryable(exc):
                if _gemini_api_error(exc):
                    raise ValueError(f"judge request failed: {exc}") from exc
                raise
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(_retry_delay(exc, attempt))
    raise ValueError(f"judge request failed: {last_error}") from last_error


def judge_available():
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def judge_one(client, model, template, question, context, reference, prediction):
    from google.genai import types

    prompt = fill_judge_prompt(template, question, context, reference, prediction)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=256,
            response_mime_type="application/json",
        ),
    )
    return parse_judge_score(response.text or "")


def default_model():
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
