"""Write synthetic (question, answer, gold_chunk_id) rows with Gemini.

The key is read from GEMINI_API_KEY. A `.env` file in the repo root is
loaded if that variable is not already set. The model id defaults to
the free-tier Flash-Lite id in jigo_rag/config.py and can be overridden
with GEMINI_MODEL.

Each request asks for several questions about one chunk. Finished
chunks are appended to data/{split}.jsonl and skipped on the next run,
so a daily free-tier cap can be continued the next day.

The test split is generated first. Those are the rows a person reviews.

    python generate_qa.py
"""

import os
import sys
import time
from pathlib import Path

from jigo_rag.config import DEFAULT_GEMINI_MODEL, QA_MAX_PER_CHUNK, QA_TARGETS
from jigo_rag.jsonl import append_jsonl, read_jsonl
from jigo_rag.qa import answer_supported, make_qa_row, parse_triples, plan_questions
from jigo_rag.schema import validate_qa

PROMPT_PATH = Path("prompts/qa_generate.txt")
MAX_ATTEMPTS = 6


def load_dotenv(path=".env"):
    """Set variables from a .env file without overriding the real environment."""
    file_path = Path(path)
    if not file_path.exists():
        return
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fill_prompt(template, chunk, n_questions):
    required = ("{{N_QUESTIONS}}", "{{CHUNK_ID}}", "{{CHUNK_TEXT}}")
    missing = [token for token in required if token not in template]
    if missing:
        raise SystemExit(f"{PROMPT_PATH} is missing {missing}")
    return (
        template.replace("{{N_QUESTIONS}}", str(n_questions))
        .replace("{{CHUNK_ID}}", chunk["chunk_id"])
        .replace("{{CHUNK_TEXT}}", chunk["text"])
    )


def is_daily_quota(exc):
    message = str(exc).lower()
    if "per minute" in message or "rate limit" in message:
        return False
    return "per day" in message or "daily quota" in message or "quota exceeded" in message


def call_model(client, model, prompt):
    from google.genai import types

    delay_seconds = 2.0
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    seed=42,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            return parse_triples(response.text or "")
        except Exception as exc:
            last_error = exc
            if is_daily_quota(exc):
                raise SystemExit(
                    "Daily Gemini free-tier quota was reached. "
                    "Rows already written were kept. Rerun the same command later."
                ) from exc
            print(
                f"attempt {attempt}/{MAX_ATTEMPTS} failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 60)
    raise RuntimeError(f"chunk failed after {MAX_ATTEMPTS} attempts: {last_error}")


def generate_split(client, model, template, split, sleep_seconds):
    chunk_path = Path("data/chunks") / f"{split}.jsonl"
    out_path = Path("data") / f"{split}.jsonl"
    chunks = read_jsonl(chunk_path)
    if not chunks:
        raise SystemExit(f"No chunks in {chunk_path}. Run python make_data.py first.")
    chunks = sorted(chunks, key=lambda row: row["chunk_id"])
    target = QA_TARGETS[split]
    per_chunk, enough = plan_questions(len(chunks), target, QA_MAX_PER_CHUNK)
    if not enough:
        print(
            f"warning: {split} has {len(chunks)} chunks, which at "
            f"{QA_MAX_PER_CHUNK} questions each cannot reach {target}",
            file=sys.stderr,
        )
    existing = read_jsonl(out_path)
    done = {row["gold_chunk_id"] for row in existing}
    written = len(existing)
    print(f"{split}: {written} rows already on disk, target {target}, {per_chunk} per chunk")
    for chunk in chunks:
        if written >= target:
            break
        if chunk["chunk_id"] in done:
            continue
        n_questions = min(per_chunk, target - written)
        prompt = fill_prompt(template, chunk, n_questions)
        try:
            triples = call_model(client, model, prompt)
        except RuntimeError as exc:
            print(f"skip {chunk['chunk_id']}: {exc}", file=sys.stderr)
            continue
        rows = []
        for triple in triples:
            if not answer_supported(triple["answer"], chunk["text"]):
                continue
            rows.append(make_qa_row(chunk, triple["question"], triple["answer"], len(rows)))
            if len(rows) >= n_questions:
                break
        if not rows:
            print(f"skip {chunk['chunk_id']}: no answer was supported by the chunk", file=sys.stderr)
            continue
        for row in rows:
            validate_qa(row)
        append_jsonl(out_path, rows)
        written += len(rows)
        done.add(chunk["chunk_id"])
        print(f"{split} {written}/{target} {chunk['chunk_id']}")
        time.sleep(sleep_seconds)
    return written


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY. See .env.example. The key is not written to disk by this script.")
    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    sleep_seconds = float(os.environ.get("GEMINI_SLEEP_SECONDS", "4"))
    template = PROMPT_PATH.read_text(encoding="utf-8")
    from google import genai

    client = genai.Client(api_key=api_key)
    # Test first: those rows are the ones a person reviews, and a daily
    # cap may stop the run before the training split is finished.
    for split in ("test", "val", "train"):
        generate_split(client, model, template, split, sleep_seconds)


if __name__ == "__main__":
    main()
