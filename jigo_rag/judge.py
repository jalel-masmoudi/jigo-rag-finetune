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


def not_found_score(prediction, reference, passages):
    """Score a 'not found' reply from the retrieved passages alone.

    Returns 1.0, 0.0, or None. None means the reply is not 'not found'
    and the model judge should score it.
    """
    from jigo_rag.metrics import normalize
    from jigo_rag.qa import answer_supported

    if normalize(prediction) != "not found":
        return None
    supported = any(answer_supported(reference, passage) for passage in passages)
    return 0.0 if supported else 1.0


def judge_with_retries(call, attempts=3):
    """Call `call` up to `attempts` times. Raise the last parse error."""
    last_error = None
    for _ in range(attempts):
        try:
            return call()
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
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
