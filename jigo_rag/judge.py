"""LLM judge for answer correctness.

The rubric and the wording live in prompts/judge.txt. The judge is
shown the passages the retriever actually returned, not the gold chunk.
The score is 0, 0.5, or 1, and it must be a number. A JSON boolean is
rejected. The model is the same free-tier Gemini default as the
question generator. If the key is missing the judge is skipped and the
score is left empty. A missing key is not a zero.

Saying "not found" is correct when none of the retrieved passages
support the reference answer, and wrong when one of them does. That
case is decided here so a retrieval miss cannot be marked wrong.
"""

import json
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


def _retryable(exc):
    if isinstance(exc, (ValueError, json.JSONDecodeError, OSError, TimeoutError, ConnectionError)):
        return True
    message = str(exc).lower()
    return any(token in message for token in ("timeout", "timed out", "unavailable", "connection", "429", "500", "502", "503"))


def judge_with_retries(call, attempts=3):
    """Retry parse failures and network errors. Raise the last one."""
    last_error = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            if not _retryable(exc):
                raise
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise ValueError(f"judge reply could not be parsed: {last_error}") from last_error


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
