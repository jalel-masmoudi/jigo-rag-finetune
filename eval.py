"""Score the baseline retriever and, when asked, the baseline generator.

The test set is data/test_reviewed.jsonl only. data/test.jsonl is raw
model output and is refused here.

    python eval.py --retrieval-only
    python eval.py
    python eval.py --smoke --smoke-size 30 --retrieval-only

`--smoke` does not read the reviewed test set. It uses the first
sentence of a chunk as the query and the chunk as the gold hit. That
is a sanity check that the index can find a passage from its own
words. It is not a number for the results table. The JSON it writes
has "smoke": true.

A full run writes results/baseline.json. Recall@1, Recall@5, MRR@10,
exact match, and token F1 each include a 95% bootstrap interval.
Latency is the per-query p50 and p95. Tokens per query are prompt
tokens plus completion tokens from the generator. The judge runs only
when GEMINI_API_KEY is set; otherwise its score is null.
"""

import argparse
import json
import time
from pathlib import Path

from jigo_rag.generator import QwenGenerator
from jigo_rag.jsonl import read_jsonl
from jigo_rag.judge import default_model, judge_available, judge_one
from jigo_rag.metrics import (
    bootstrap_mean_ci,
    exact_match,
    mrr_at_k,
    percentile,
    recall_at_k,
    token_f1,
)
from jigo_rag.retriever import FaissRetriever

REVIEWED_TEST = Path("data/test_reviewed.jsonl")
JUDGE_PROMPT = Path("prompts/judge.txt")


def load_corpus():
    chunks = []
    for split in ("train", "val", "test"):
        chunks.extend(read_jsonl(Path("data/chunks") / f"{split}.jsonl"))
    if not chunks:
        raise SystemExit("No chunks in data/chunks/. Run python make_data.py first.")
    return chunks


def load_reviewed_test():
    if not REVIEWED_TEST.exists():
        raise SystemExit(
            "data/test_reviewed.jsonl does not exist. "
            "Review the test rows with python review_test.py. "
            "data/test.jsonl is not the test set."
        )
    rows = read_jsonl(REVIEWED_TEST)
    if not rows:
        raise SystemExit("data/test_reviewed.jsonl is empty.")
    return rows


def first_sentence(text):
    """A short query taken from the chunk itself, for the smoke check."""
    flat = " ".join(text.split())
    for mark in (". ", "? ", "! "):
        if mark in flat:
            sentence = flat.split(mark, 1)[0].strip()
            if len(sentence) >= 40:
                return sentence
    return flat[:200].strip()


def smoke_rows(chunks, n, seed):
    import random

    pool = [chunk for chunk in chunks if chunk["split"] == "test"] or list(chunks)
    rng = random.Random(seed)
    rng.shuffle(pool)
    rows = []
    for chunk in pool:
        query = first_sentence(chunk["text"])
        if len(query) < 20:
            continue
        rows.append(
            {
                "question": query,
                "answer": query,
                "gold_chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["text"],
            }
        )
        if len(rows) >= n:
            break
    if not rows:
        raise SystemExit("smoke check found no usable chunks")
    return rows


def score_retrieval(gold_chunk_id, hits):
    ranked = [hit["chunk_id"] for hit in hits]
    return {
        "recall@1": recall_at_k(ranked, gold_chunk_id, 1),
        "recall@5": recall_at_k(ranked, gold_chunk_id, 5),
        "mrr@10": mrr_at_k(ranked, gold_chunk_id, 10),
    }


QUALITY_KEYS = ("recall@1", "recall@5", "mrr@10", "exact_match", "token_f1", "judge")


def summarize(per_question, n_resamples, seed):
    report = {}
    for key in QUALITY_KEYS:
        if key not in per_question[0]:
            continue
        report[key] = bootstrap_mean_ci(
            [row[key] for row in per_question], n_resamples=n_resamples, seed=seed
        )
    return report


def run(rows, retriever, generator, judge, smoke, n_resamples, seed):
    per_question = []
    latencies_ms = []
    for row in rows:
        started = time.perf_counter()
        hits = retriever.search(row["question"], k=10)
        retrieval_seconds = time.perf_counter() - started
        record = score_retrieval(row["gold_chunk_id"], hits)
        record["retrieval_latency_ms"] = retrieval_seconds * 1000
        if generator is not None:
            gen_started = time.perf_counter()
            generated = generator.answer(row["question"], [hit["text"] for hit in hits[:5]])
            record["exact_match"] = exact_match(generated["answer"], row["answer"])
            record["token_f1"] = token_f1(generated["answer"], row["answer"])
            record["tokens"] = generated["prompt_tokens"] + generated["completion_tokens"]
            record["generation_latency_ms"] = (time.perf_counter() - gen_started) * 1000
            if judge is not None:
                record["judge"] = judge(
                    row["question"],
                    row.get("chunk_text", hits[0]["text"] if hits else ""),
                    row["answer"],
                    generated["answer"],
                )
        total_ms = (time.perf_counter() - started) * 1000
        record["latency_ms"] = total_ms
        latencies_ms.append(total_ms)
        per_question.append(record)
    summary = summarize(per_question, n_resamples, seed)
    summary["latency_ms"] = {
        "p50": percentile(latencies_ms, 50),
        "p95": percentile(latencies_ms, 95),
        "n": len(latencies_ms),
    }
    if generator is not None and "tokens" in per_question[0]:
        summary["tokens_per_query"] = sum(row["tokens"] for row in per_question) / len(per_question)
    summary["smoke"] = smoke
    summary["n"] = len(rows)
    summary["n_chunks_indexed"] = len(retriever.chunks)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-size", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N reviewed rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    chunks = load_corpus()
    retriever = FaissRetriever().build(chunks)
    if args.smoke:
        rows = smoke_rows(chunks, args.smoke_size, args.seed)
        output = Path(args.output or "results/smoke_retrieval.json")
    else:
        rows = load_reviewed_test()
        if args.limit:
            rows = rows[: args.limit]
        output = Path(args.output or "results/baseline.json")

    generator = None if args.retrieval_only else QwenGenerator()
    judge = None
    if generator is not None and judge_available():
        from generate_qa import load_dotenv
        from google import genai

        load_dotenv()
        template = JUDGE_PROMPT.read_text(encoding="utf-8")
        client = genai.Client(api_key=__import__("os").environ["GEMINI_API_KEY"])
        model = default_model()

        def judge(question, context, reference, prediction, _client=client, _model=model, _template=template):
            return judge_one(_client, _model, _template, question, context, reference, prediction)

    elif generator is not None:
        print("GEMINI_API_KEY is not set. The judge score will be null.")

    summary = run(rows, retriever, generator, judge, smoke=args.smoke, n_resamples=args.bootstrap, seed=args.seed)
    if args.smoke:
        summary["query_source"] = "first_sentence_of_gold_chunk"
        summary["note"] = (
            "Smoke check only. The query is a sentence from the gold chunk. "
            "Do not copy these numbers into the results table."
        )
    if generator is not None and not judge_available():
        summary["judge"] = None
        summary["judge_note"] = "GEMINI_API_KEY was not set, so no judge score was computed."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "note"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
