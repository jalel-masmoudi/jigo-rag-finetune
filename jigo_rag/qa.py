"""Plan and check synthetic questions.

The generator asks for several questions per chunk so a free-tier
daily request cap can cover about 2,000 questions. The gold chunk id
is assigned by the script from the chunk that was sent, not copied
from the model's reply.

`answer_supported` drops an answer whose words are mostly absent from
that chunk. It is a filter, not a proof: a person still reviews the
test rows.
"""

import json
import math
import re


def plan_questions(n_chunks, target, cap):
    """Return (questions_per_chunk, enough).

    `enough` is false when even `cap` questions on every chunk cannot
    reach `target`. The caller still generates what it can.
    """
    if target < 0 or cap < 1:
        raise ValueError("target must be >= 0 and cap must be >= 1")
    if n_chunks <= 0 or target == 0:
        return 0, True
    per_chunk = min(cap, max(1, math.ceil(target / n_chunks)))
    enough = n_chunks * per_chunk >= target
    return per_chunk, enough


def parse_triples(text):
    """Parse a JSON list of {question, answer} objects.

    A surrounding ```json fence is removed if the model added one.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("questions", "triples", "items"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("model output is not a JSON list")
    triples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            triples.append({"question": question, "answer": answer})
    if not triples:
        raise ValueError("model output had no usable questions")
    return triples


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def answer_supported(answer, chunk):
    """True when the answer is mostly made of words from the chunk."""
    answer_words = _words(answer)
    if not answer_words:
        return False
    if " ".join(answer_words) in " ".join(_words(chunk)):
        return True
    content_words = [word for word in answer_words if len(word) > 3]
    if not content_words:
        return False
    chunk_words = set(_words(chunk))
    hits = sum(1 for word in content_words if word in chunk_words)
    return hits / len(content_words) >= 0.6


def make_qa_row(chunk, question, answer, index):
    """Build one QA row. `index` is 0-based within this chunk."""
    return {
        "row_id": f"{chunk['chunk_id']}/q{index}",
        "question": question,
        "answer": answer,
        "gold_chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "chunk_text": chunk["text"],
        "source": chunk["source"],
        "split": chunk["split"],
    }
