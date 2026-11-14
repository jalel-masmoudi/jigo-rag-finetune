"""Fine-tune bge-small with MultipleNegativesRankingLoss.

The grid is the one in the project spec: learning rate 2e-5 or 5e-5,
and 1, 2, or 3 epochs. Each run starts from the public bge checkpoint,
not from the previous run. The winner is the highest Recall@5 on the
validation questions. That one model is then scored on the reviewed
test set, once.

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
from jigo_rag.metrics import bootstrap_mean_ci, recall_at_k
from jigo_rag.negatives import mine_hard_negatives, select_best, training_columns
from jigo_rag.retriever import EMBED_MODEL, FaissRetriever

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
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(
            f"{path} is missing or empty. Run python generate_qa.py first. "
            "The smoke flag can exercise the loop without those files."
        )
    return rows


def recall_at_5_scores(model, questions, corpus):
    """One Recall@5 value (0 or 1) per question."""
    retriever = FaissRetriever(model_name=EMBED_MODEL)
    retriever.model = model
    retriever.build(corpus)
    scores = []
    for row in questions:
        ranked = [hit["chunk_id"] for hit in retriever.search(row["question"], k=5)]
        scores.append(recall_at_k(ranked, row["gold_chunk_id"], 5))
    return scores


def train_one(rows, learning_rate, epochs, output_dir, max_steps=None, batch_size=16):
    """Train one grid point.

    Batch size 16 is the real run. MultipleNegativesRankingLoss treats
    the other rows in the batch as negatives, so a bigger batch is a
    bigger set of negatives. The smoke step uses 2: a batch of 8 was
    killed for memory on a 16 GB CPU machine.
    """
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import (
        SentenceTransformerTrainingArguments,
    )

    model = SentenceTransformer(EMBED_MODEL)
    dataset = Dataset.from_dict(training_columns(rows))
    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
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

    if args.smoke:
        examples = smoke_examples(corpus)
        # Index a small slice so the smoke step does not re-embed the whole handbook.
        slice_ids = {example["doc_id"] for example in examples}
        small = [chunk for chunk in corpus if chunk["doc_id"] in slice_ids][:40]
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
    test_rows = read_jsonl("data/test_reviewed.jsonl")
    if not test_rows:
        raise SystemExit("data/test_reviewed.jsonl is empty. Review the test set before scoring it.")
    baseline = FaissRetriever().build(corpus)

    def search(question, k):
        return baseline.search(question, k=k)

    train_mined = mine_hard_negatives(train_rows, search, corpus, seed=SEED)
    records = []
    for spec in GRID:
        folder = f"models/embed-lr{spec['learning_rate']}-ep{spec['epochs']}"
        model = train_one(train_mined, spec["learning_rate"], spec["epochs"], folder)
        val_scores = recall_at_5_scores(model, val_rows, corpus)
        record = {
            **spec,
            "val_recall@5": sum(val_scores) / len(val_scores),
            "dir": folder,
        }
        records.append(record)
        print(record)
    winner = select_best(records)
    from sentence_transformers import SentenceTransformer

    # Score the selected checkpoint on the reviewed test set once.
    chosen = SentenceTransformer(winner["dir"])
    test_scores = recall_at_5_scores(chosen, test_rows, corpus)
    report = {
        "smoke": False,
        "runs": records,
        "winner": {key: winner[key] for key in ("learning_rate", "epochs", "val_recall@5", "dir")},
        "test_recall@5": bootstrap_mean_ci(test_scores, seed=SEED),
    }
    path = Path("results/embed.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
