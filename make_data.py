"""Download the pinned corpora, chunk them, and split by document.

Rebuild from scratch with:

    python make_data.py

The seed and the fractions live in jigo_rag/config.py. The same seed
and the same documents always write the same split. Generated files
under data/chunks/ and data/raw/ are gitignored.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from jigo_rag.chunk import chunk_text, load_tokenizer
from jigo_rag.config import QA_MAX_PER_CHUNK, QA_TARGETS, SEED
from jigo_rag.corpus import load_documents
from jigo_rag.jsonl import write_jsonl
from jigo_rag.qa import plan_questions
from jigo_rag.schema import validate_chunk
from jigo_rag.split import assert_no_doc_leakage, assign_splits


def build(raw_root, out_dir, seed):
    documents, dropped = load_documents(raw_root)
    tokenizer = load_tokenizer()
    mapping = assign_splits([document["doc_id"] for document in documents], seed=seed)
    chunks = []
    for document in documents:
        for part in chunk_text(document["text"], tokenizer):
            row = {
                "chunk_id": f"{document['doc_id']}#{part['n']}",
                "doc_id": document["doc_id"],
                "source": document["source"],
                "doc_path": document["doc_path"],
                "text": part["text"],
                "n_tokens": part["n_tokens"],
                "split": mapping[document["doc_id"]],
            }
            validate_chunk(row)
            chunks.append(row)
    # This is the leakage guard. It runs on the rows that are written.
    assert_no_doc_leakage(chunks)

    out_dir = Path(out_dir)
    by_split = {}
    for split in ("train", "val", "test"):
        rows = [row for row in chunks if row["split"] == split]
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        by_split[split] = rows

    docs_per_split = Counter(mapping.values())
    chunks_per_split = {split: len(rows) for split, rows in by_split.items()}
    source_docs = Counter(document["source"] for document in documents)
    drop_reasons = Counter(item["reason"] for item in dropped)
    qa_plan = {}
    for split, target in QA_TARGETS.items():
        per_chunk, enough = plan_questions(
            chunks_per_split[split], target, QA_MAX_PER_CHUNK
        )
        qa_plan[split] = {
            "target_questions": target,
            "questions_per_chunk": per_chunk,
            "enough_chunks": enough,
        }
    stats = {
        "seed": seed,
        "documents_kept": len(documents),
        "documents_per_source": dict(source_docs),
        "documents_dropped": len(dropped),
        "drop_reasons": dict(drop_reasons),
        "dropped": dropped,
        "docs_per_split": dict(docs_per_split),
        "chunks_per_split": chunks_per_split,
        "qa_plan": qa_plan,
    }
    by_source = {}
    for doc_id, split_name in mapping.items():
        by_source.setdefault(split_name, Counter())[doc_id.split("/", 1)[0]] += 1
    stats["docs_per_split_by_source"] = {name: dict(counts) for name, counts in by_source.items()}
    # Stats sit next to the chunk files so --out is the only output path.
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def print_stats(stats):
    print(f"seed {stats['seed']}")
    print(
        f"documents kept {stats['documents_kept']} "
        f"{stats['documents_per_source']}"
    )
    print(
        f"documents dropped {stats['documents_dropped']} "
        f"{stats['drop_reasons']}"
    )
    print(f"docs per split {stats['docs_per_split']}")
    print(f"docs per split by source {stats['docs_per_split_by_source']}")
    print(f"chunks per split {stats['chunks_per_split']}")
    for split, plan in stats["qa_plan"].items():
        flag = "ok" if plan["enough_chunks"] else "SHORT"
        print(
            f"  {split}: target {plan['target_questions']} questions, "
            f"{plan['questions_per_chunk']} per chunk ({flag})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw", help="where to clone the corpora")
    parser.add_argument("--out", default="data/chunks", help="where to write split chunk files")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print_stats(build(args.raw, args.out, args.seed))


if __name__ == "__main__":
    main()
