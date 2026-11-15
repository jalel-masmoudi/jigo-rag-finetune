"""Fine-tune bge-small with MultipleNegativesRankingLoss.

The grid is the one in the project spec: learning rate 2e-5 or 5e-5,
and 1, 2, or 3 epochs. Each run starts from the public bge checkpoint,
not from the previous run. The winner is the highest Recall@5 on the
validation questions. That one model is then scored on the filtered
test set, once, and saved to models/embed.

Hard negatives are mined by eval.py's retriever (the baseline), three
per question, excluding the gold chunk and the rest of its document.

    python train_embed.py
    python train_embed.py --smoke

`--smoke` trains one step on eight sentence-from-the-chunk pairs.
It does not sweep and it does not score the test set. The file it
writes has "smoke": true.

There is no training file until generate_qa.py has been run. The
official command exits with that message instead of inventing pairs.
"""

import argparse
import json
from pathlib import Path

from jigo_rag.config import SEED
from jigo_rag.jsonl import read_jsonl
from jigo_rag.metrics import bootstrap_mean_ci, mrr_at_k, paired_bootstrap_diff, recall_at_k
from jigo_rag.negatives import mine_hard_negatives, select_best, training_columns
from jigo_rag.retriever import EMBED_MODEL, QUERY_PREFIX, FaissRetriever
from jigo_rag.schema import require_review_actions

GRID = (
    {"learning_rate": 2e-5, "epochs": 1},
    {"learning_rate": 2e-5, "epochs": 2},
    {"learning_rate": 2e-5, "epochs": 3},
    {"learning_rate": 5e-5, "epochs": 1},
    {"learning_rate": 5e-5, "epochs": 2},
    {"learning_rate": 5e-5, "epochs": 3},
)


def load_split(name):
    path = Path("data") / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run python generate_qa.py first. "
            "The smoke flag can exercise the loop without those files."
        )
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"{path} is empty.")
    return rows


def per_query_scores(model, questions, corpus):
    """Recall@5 and MRR@10 for each question, in the same order.

    The index is the full corpus. `search` adds QUERY_PREFIX, the same
    prefix the training rows carry on the question column.
    """
    retriever = FaissRetriever(model_name=EMBED_MODEL)
    retriever.model = model
    retriever.build(corpus)
    recall_scores = []
    mrr_scores = []
    for row in questions:
        ranked = [hit["chunk_id"] for hit in retriever.search(row["question"], k=10)]
        recall_scores.append(recall_at_k(ranked, row["gold_chunk_id"], 5))
        mrr_scores.append(mrr_at_k(ranked, row["gold_chunk_id"], 10))
    return recall_scores, mrr_scores


def train_one(rows, learning_rate, epochs, output_dir, max_steps=None, batch_size=16):
    """Train one grid point.

    Batch size 16 is the real run. MultipleNegativesRankingLoss treats
    the other rows in the batch as negatives, so a bigger batch is a
    bigger set of negatives. The smoke step uses 2: a batch of 8 was
    killed for memory on a 16 GB CPU machine.
    """
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.base.sampler import BatchSamplers
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import (
        SentenceTransformerTrainingArguments,
    )

    model = SentenceTransformer(EMBED_MODEL)
    # Same query prefix the retriever uses at evaluation. Passages stay bare.
    dataset = Dataset.from_dict(training_columns(rows, query_prefix=QUERY_PREFIX))
    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        # A duplicate query in one batch would be treated as a negative of itself.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Batch 16 at 512 tokens needs more than 6.5 GB of activations
        # without this, which does not fit an 8 GB card. Same gradients, slower step.
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_steps=max_steps if max_steps else -1,
        # A float here is a ratio of the run (this Transformers version).
        warmup_steps=0.1,
        seed=SEED,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=10,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()
    # save_strategy is "no" so the trainer does not write a checkpoint
    # every step. One save at the end is the run we might select.
    model.save(output_dir)
    return model


def smoke_examples(chunks, n=8):
    """Eight pairs whose question is a sentence from the positive chunk.

    Used only to prove the training step runs. Not a fine-tune result.
    """
    from eval import first_sentence

    picked = [chunk for chunk in chunks if len(chunk["text"]) > 200][:n]
    examples = []
    for chunk in picked:
        examples.append(
            {
                "question": first_sentence(chunk["text"]),
                "gold_chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "chunk_text": chunk["text"],
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    corpus = []
    for split in ("train", "val", "test"):
        corpus.extend(read_jsonl(Path("data/chunks") / f"{split}.jsonl"))
    if not corpus:
        raise SystemExit("No chunks in data/chunks/. Run python make_data.py first.")

    train_chunks = [chunk for chunk in corpus if chunk.get("split") == "train"]
    if args.smoke:
        examples = smoke_examples(train_chunks)
        # Index a small train-only slice so the smoke step stays small.
        slice_ids = {example["doc_id"] for example in examples}
        small = [chunk for chunk in train_chunks if chunk["doc_id"] in slice_ids][:40]
        baseline = FaissRetriever().build(small)

        def search(question, k):
            return baseline.search(question, k=k)

        rows = mine_hard_negatives(examples, search, small, seed=SEED)
        if len(rows) < 2:
            raise SystemExit("smoke mining produced fewer than 2 rows")
        train_one(
            rows,
            learning_rate=2e-5,
            epochs=1,
            output_dir="models/smoke-embed",
            max_steps=1,
            batch_size=2,
        )
        report = {
            "smoke": True,
            "n_rows": len(rows),
            "max_steps": 1,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "note": "One optimizer step on sentence-from-the-chunk pairs. Not a validation or test score.",
        }
        path = Path("results/smoke_embed.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    train_rows = load_split("train")
    val_rows = load_split("val")
    reviewed = Path("data/test_reviewed.jsonl")
    if not reviewed.exists():
        raise SystemExit("data/test_reviewed.jsonl does not exist. Run python auto_filter_test.py.")
    test_rows = require_review_actions(read_jsonl(reviewed))
    if not test_rows:
        raise SystemExit("data/test_reviewed.jsonl is empty.")
    # Negatives come from training chunks only. The test index is separate.
    miner = FaissRetriever().build(train_chunks)

    def search(question, k):
        return miner.search(question, k=k)

    train_mined = mine_hard_negatives(train_rows, search, train_chunks, seed=SEED)
    records = []
    for spec in GRID:
        folder = f"models/embed-lr{spec['learning_rate']}-ep{spec['epochs']}"
        model = train_one(train_mined, spec["learning_rate"], spec["epochs"], folder)
        val_scores, _val_mrr = per_query_scores(model, val_rows, corpus)
        record = {
            **spec,
            "val_recall@5": sum(val_scores) / len(val_scores),
            "dir": folder,
        }
        records.append(record)
        print(record)
    winner = select_best(records)
    from sentence_transformers import SentenceTransformer

    # Score the selected checkpoint on the filtered test set once.
    chosen = SentenceTransformer(winner["dir"])
    # The grid directories stay so a run can be inspected. models/embed
    # is the checkpoint that was selected.
    chosen.save("models/embed")
    new_recall, new_mrr = per_query_scores(chosen, test_rows, corpus)
    base_model = SentenceTransformer(EMBED_MODEL)
    base_recall, base_mrr = per_query_scores(base_model, test_rows, corpus)
    report = {
        "smoke": False,
        "seed": SEED,
        "runs": records,
        "winner": {key: winner[key] for key in ("learning_rate", "epochs", "val_recall@5", "dir")},
        "test_recall@5": bootstrap_mean_ci(new_recall, seed=SEED),
        "test_mrr@10": bootstrap_mean_ci(new_mrr, seed=SEED),
        "recall@5_gain_vs_baseline": paired_bootstrap_diff(base_recall, new_recall, seed=SEED),
        "mrr@10_gain_vs_baseline": paired_bootstrap_diff(base_mrr, new_mrr, seed=SEED),
    }
    path = Path("results/embed.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
