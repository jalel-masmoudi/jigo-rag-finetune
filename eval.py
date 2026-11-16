"""Score the baseline retriever and, when asked, the baseline generator.

The test set is data/test_reviewed.jsonl only. data/test.jsonl is raw
model output and is refused here.

    python eval.py --retrieval-only
    python eval.py
    python eval.py --smoke --smoke-size 30 --retrieval-only

`--smoke` does not read data/test_reviewed.jsonl. It uses the first
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

from jigo_rag.generator import GENERATOR_MODEL, QwenGenerator
from jigo_rag.jsonl import read_jsonl
from jigo_rag.judge import (
    default_model,
    end_to_end_not_found,
    format_passages,
    judge_available,
    judge_failure_skips,
    judge_one,
    judge_with_retries,
    not_found_score,
)
from jigo_rag.metrics import (
    bootstrap_mean_ci,
    exact_match,
    mrr_at_k,
    percentile,
    recall_at_k,
    token_f1,
)
from jigo_rag.retriever import FaissRetriever
from jigo_rag.schema import require_review_actions

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
            "Run python auto_filter_test.py. "
            "data/test.jsonl is not the test set."
        )
    rows = read_jsonl(REVIEWED_TEST)
    if not rows:
        raise SystemExit("data/test_reviewed.jsonl is empty.")
    return require_review_actions(rows)


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
        if key == "judge":
            if not any("judge" in row or row.get("judge_skipped") for row in per_question):
                continue
            scored = [row["judge"] for row in per_question if "judge" in row]
            skipped = sum(1 for row in per_question if row.get("judge_skipped"))
            if scored:
                judge_summary = bootstrap_mean_ci(scored, n_resamples=n_resamples, seed=seed)
            else:
                judge_summary = {"mean": None, "n": 0}
            judge_summary["skipped"] = skipped
            report["judge"] = judge_summary
            conditioned = [
                row["retrieval_conditioned_judge"]
                for row in per_question
                if "retrieval_conditioned_judge" in row
            ]
            if conditioned:
                report["retrieval_conditioned_judge"] = bootstrap_mean_ci(
                    conditioned, n_resamples=n_resamples, seed=seed
                )
                report["retrieval_conditioned_judge"]["label"] = (
                    "Not-found replies only. 1 when the retrieved passages do not "
                    "contain the reference, else 0. The judge mean scores not-found as 0."
                )
            continue
        if key not in per_question[0]:
            continue
        report[key] = bootstrap_mean_ci(
            [row[key] for row in per_question], n_resamples=n_resamples, seed=seed
        )
    return report


def run(rows, retriever, generator, judge, smoke, n_resamples, seed, lora_generator=None, reranker=None):
    per_question = []
    latencies_ms = []
    for row in rows:
        started = time.perf_counter()
        # MRR@10 needs ten hits. The reranker looks at twenty, then we
        # score the reordered list.
        depth = 20 if reranker is not None else 10
        hits = retriever.search(row["question"], k=depth)
        if reranker is not None:
            from jigo_rag.rerank import rerank

            hits = rerank(reranker, row["question"], hits)
        hits = hits[:10]
        retrieval_seconds = time.perf_counter() - started
        record = score_retrieval(row["gold_chunk_id"], hits)
        record["retrieval_latency_ms"] = retrieval_seconds * 1000
        passages = [hit["text"] for hit in hits[:5]]
        generated = None
        if generator is not None:
            gen_started = time.perf_counter()
            generated = generator.answer(row["question"], passages)
            record["exact_match"] = exact_match(generated["answer"], row["answer"])
            record["token_f1"] = token_f1(generated["answer"], row["answer"])
            record["tokens"] = generated["prompt_tokens"] + generated["completion_tokens"]
            record["generation_latency_ms"] = (time.perf_counter() - gen_started) * 1000
        # The clock stops before the judge. The judge is not part of
        # the system a user would wait on.
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        latencies_ms.append(record["latency_ms"])
        lora_answer = None
        if lora_generator is not None:
            # Same passages as the baseline generator. Its own clock is
            # retrieval plus this generation, not the baseline generation.
            lora_started = time.perf_counter()
            lora_out = lora_generator.answer(row["question"], passages)
            lora_gen_ms = (time.perf_counter() - lora_started) * 1000
            lora_answer = lora_out["answer"]
            record["lora_exact_match"] = exact_match(lora_answer, row["answer"])
            record["lora_token_f1"] = token_f1(lora_answer, row["answer"])
            record["lora_tokens"] = lora_out["prompt_tokens"] + lora_out["completion_tokens"]
            record["lora_latency_ms"] = record["retrieval_latency_ms"] + lora_gen_ms
        if generator is not None and judge is not None:
            record.update(_judge_fields(judge, row, passages, generated["answer"], prefix=""))
        if lora_answer is not None and judge is not None:
            record.update(_judge_fields(judge, row, passages, lora_answer, prefix="lora_"))
        per_question.append(record)
    summary = summarize(per_question, n_resamples, seed)
    summary["latency_ms"] = {
        "p50": percentile(latencies_ms, 50),
        "p95": percentile(latencies_ms, 95),
        "n": len(latencies_ms),
    }
    if generator is not None and "tokens" in per_question[0]:
        summary["tokens_per_query"] = sum(row["tokens"] for row in per_question) / len(per_question)
    if lora_generator is not None:
        summary["lora"] = {
            "seed": seed,
            "exact_match": bootstrap_mean_ci(
                [row["lora_exact_match"] for row in per_question], n_resamples=n_resamples, seed=seed
            ),
            "token_f1": bootstrap_mean_ci(
                [row["lora_token_f1"] for row in per_question], n_resamples=n_resamples, seed=seed
            ),
            "latency_ms": {
                "p50": percentile([row["lora_latency_ms"] for row in per_question], 50),
                "p95": percentile([row["lora_latency_ms"] for row in per_question], 95),
                "n": len(per_question),
            },
        }
        lora_judged = [row["lora_judge"] for row in per_question if "lora_judge" in row]
        if lora_judged:
            summary["lora"]["judge"] = bootstrap_mean_ci(lora_judged, n_resamples=n_resamples, seed=seed)
        skipped = sum(1 for row in per_question if row.get("lora_judge_skipped"))
        if skipped:
            summary["lora"]["judge_skipped"] = skipped
        conditioned = [
            row["lora_retrieval_conditioned_judge"]
            for row in per_question
            if "lora_retrieval_conditioned_judge" in row
        ]
        if conditioned:
            summary["lora"]["lora_retrieval_conditioned_judge"] = bootstrap_mean_ci(
                conditioned, n_resamples=n_resamples, seed=seed
            )
            summary["lora"]["lora_retrieval_conditioned_judge"]["label"] = (
                "Not-found replies only. 1 when the retrieved passages do not "
                "contain the reference, else 0. The judge mean scores not-found as 0."
            )
    summary["seed"] = seed
    summary["smoke"] = smoke
    summary["n"] = len(rows)
    summary["n_chunks_indexed"] = len(retriever.chunks)
    return summary


def _make_judge():
    """The Gemini judge, or None when the key is missing.

    One function, so main does not define `judge` a second time.
    This is the only place that reads .env.
    """
    from generate_qa import load_dotenv

    load_dotenv()
    if not judge_available():
        print("GEMINI_API_KEY is not set. The judge score will be null.")
        return None
    from google import genai

    template = JUDGE_PROMPT.read_text(encoding="utf-8")
    client = genai.Client(api_key=__import__("os").environ["GEMINI_API_KEY"])
    model = default_model()

    def judge(question, passages, reference, prediction, _client=client, _model=model, _template=template):
        context = format_passages(passages)

        def call():
            return judge_one(_client, _model, _template, question, context, reference, prediction)

        return judge_with_retries(call)

    return judge


def _judge_fields(judge, row, passages, prediction, prefix):
    """Score one answer. A parse failure is counted, not fatal.

    'not found' is 0 on the end-to-end judge. The retrieval-conditioned
    value is a separate field and is not that score.
    """
    key = f"{prefix}judge" if prefix else "judge"
    skip_key = f"{prefix}judge_skipped" if prefix else "judge_skipped"
    cond_key = f"{prefix}retrieval_conditioned_judge" if prefix else "retrieval_conditioned_judge"
    forced = end_to_end_not_found(prediction)
    if forced is not None:
        return {key: forced, cond_key: not_found_score(prediction, row["answer"], passages)}
    try:
        return {key: judge(row["question"], passages, row["answer"], prediction)}
    except Exception as exc:
        if not judge_failure_skips(exc):
            raise
        return {skip_key: True}


def read_adapter_config(path):
    """The rank and alpha stored with the adapter, when the file is there."""
    config_path = Path(path) / "adapter_config.json"
    if not config_path.exists():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    keys = ("r", "lora_alpha", "target_modules", "base_model_name_or_path")
    return {key: data[key] for key in keys if key in data}


def suite_is_smoke(limit):
    """Any --limit run is a partial file, not the full test set."""
    return bool(limit)


def planned_rows(has_embed, has_reranker, has_lora):
    """Which table rows a suite can produce from the weights that exist.

    The reranker row and the full pipeline both need the fine-tuned
    retriever, because that is the retriever they sit on. The LoRA row
    is the generator comparison on the baseline retrieval.
    """
    rows = ["baseline"]
    if has_embed:
        rows.append("finetuned_embeddings")
    if has_embed and has_reranker:
        rows.append("reranker")
    if has_lora:
        rows.append("lora_generator")
    if has_embed and has_reranker and has_lora:
        rows.append("full")
    return rows


def _present(path):
    directory = Path(path)
    return directory if directory.is_dir() and any(directory.iterdir()) else None


class _BaseGenerator:
    """The shared Qwen model with its LoRA adapter turned off."""

    def __init__(self, shared):
        self.shared = shared

    def answer(self, question, passages, max_new_tokens=128):
        return self.shared.answer(
            question, passages, max_new_tokens=max_new_tokens, disable_adapter=True
        )


def suite_generators(lora_path, retrieval_only):
    """One Qwen for the whole suite when an adapter is on disk.

    Loading the base and the base-plus-adapter as two models does not
    fit an 8 GB card. Base rows run inside disable_adapter(). The LoRA
    rows use the same object with the adapter on. Prompts and greedy
    decoding stay the ones in QwenGenerator.answer.
    """
    if retrieval_only:
        # No base generator. The LoRA row, if the adapter is on disk,
        # still needs the one adapted model.
        if not lora_path:
            return None, None
        return None, QwenGenerator(adapter_path=str(lora_path))
    if not lora_path:
        return QwenGenerator(), None
    shared = QwenGenerator(adapter_path=str(lora_path))
    return _BaseGenerator(shared), shared


def run_suite(args):
    """One command for every results-table row the weights can support.

    Missing weights leave that row null. Nothing is filled in with a guess.
    """
    chunks = load_corpus()
    rows = load_reviewed_test()
    if args.limit:
        rows = rows[: args.limit]
    embed = _present(args.embed_model or "models/embed")
    reranker_dir = _present(args.reranker or "models/reranker")
    lora = _present(args.lora or "models/lora")
    names = planned_rows(bool(embed), bool(reranker_dir), bool(lora))
    judge = None if args.retrieval_only else _make_judge()
    baseline = FaissRetriever().build(chunks)
    tuned = FaissRetriever(model_name=str(embed)).build(chunks) if embed else None
    reranker_model = None
    if reranker_dir is not None:
        from sentence_transformers import CrossEncoder

        reranker_model = CrossEncoder(str(reranker_dir))
    generator, lora_generator = suite_generators(lora, args.retrieval_only)

    # --limit covers the table and every row, including a retrieval-only suite.
    smoke = bool(args.smoke) or suite_is_smoke(args.limit)

    def one(retriever, generator_model, lora_model=None, reranker=None):
        return run(
            rows,
            retriever,
            generator_model,
            judge,
            smoke=smoke,
            n_resamples=args.bootstrap,
            seed=args.seed,
            lora_generator=lora_model,
            reranker=reranker,
        )

    # A limited run is not the full test set, so it is marked as a smoke file.
    table = {"smoke": smoke, "seed": args.seed, "rows": {}}
    if args.limit:
        table["limit"] = args.limit
    table["rows"]["baseline"] = one(baseline, generator)
    if "finetuned_embeddings" in names:
        table["rows"]["finetuned_embeddings"] = one(tuned, generator)
    else:
        table["rows"]["finetuned_embeddings"] = None
    if "reranker" in names:
        table["rows"]["reranker"] = one(tuned, generator, reranker=reranker_model)
    else:
        table["rows"]["reranker"] = None
    if "lora_generator" in names:
        compared = one(baseline, generator, lora_model=lora_generator)
        table["rows"]["lora_generator"] = compared["lora"]
        table["rows"]["lora_generator"]["smoke"] = smoke
        table["rows"]["lora_generator"]["adapter"] = str(lora)
        table["rows"]["lora_generator"]["config"] = read_adapter_config(lora)
        table["rows"]["lora_generator"]["base_model"] = GENERATOR_MODEL
    else:
        table["rows"]["lora_generator"] = None
    if "full" in names:
        table["rows"]["full"] = one(tuned, lora_generator, reranker=reranker_model)
    else:
        table["rows"]["full"] = None
    table["note"] = (
        "A null row means the weights for that setup were not on disk. "
        "It is not a score of zero."
    )
    output = Path(args.output or "results/table.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("rows:", ", ".join(name for name, value in table["rows"].items() if value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-size", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--embed-model", default="", help="Fine-tuned embedding directory. Default for --suite is models/embed.")
    parser.add_argument("--reranker", default="", help="Cross-encoder directory. Default for --suite is models/reranker.")
    parser.add_argument("--lora", default="", help="LoRA adapter directory. Scored on the same retrieved passages as the baseline generator.")
    parser.add_argument("--suite", action="store_true", help="Write every table row the weights on disk can support, including the full pipeline.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.suite:
        run_suite(args)
        return

    chunks = load_corpus()
    if args.embed_model:
        if not _present(args.embed_model):
            raise SystemExit(f"Embedding model not found: {args.embed_model}")
        retriever = FaissRetriever(model_name=args.embed_model).build(chunks)
    else:
        retriever = FaissRetriever().build(chunks)
    reranker_model = None
    if args.reranker:
        if not _present(args.reranker):
            raise SystemExit(f"Reranker not found: {args.reranker}")
        from sentence_transformers import CrossEncoder

        reranker_model = CrossEncoder(args.reranker)
    if args.smoke:
        rows = smoke_rows(chunks, args.smoke_size, args.seed)
        output = Path(args.output or "results/smoke_retrieval.json")
    else:
        rows = load_reviewed_test()
        if args.limit:
            rows = rows[: args.limit]
        output = Path(args.output or ("results/e3.json" if args.lora else "results/baseline.json"))

    lora_generator = None
    if args.lora:
        if args.retrieval_only:
            raise SystemExit("--lora scores a generator, so it cannot be combined with --retrieval-only.")
        if not Path(args.lora).is_dir():
            raise SystemExit(f"LoRA adapter not found: {args.lora}")
        generator, lora_generator = suite_generators(args.lora, retrieval_only=False)
    else:
        generator = None if args.retrieval_only else QwenGenerator()
    judge = _make_judge() if generator is not None else None

    summary = run(
        rows,
        retriever,
        generator,
        judge,
        smoke=bool(args.smoke) or suite_is_smoke(args.limit),
        n_resamples=args.bootstrap,
        seed=args.seed,
        lora_generator=lora_generator,
        reranker=reranker_model,
    )
    if lora_generator is not None:
        summary["lora"]["adapter"] = args.lora
        summary["lora"]["config"] = read_adapter_config(args.lora)
        summary["lora"]["base_model"] = GENERATOR_MODEL
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
