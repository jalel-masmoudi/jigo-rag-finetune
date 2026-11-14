"""Fine-tune bge-reranker-base and measure Recall@5 after reranking.

Training pairs are the same hard negatives as the embedding fine-tune:
the gold passage is label 1, each negative is label 0. The loss is
binary cross-entropy on the cross-encoder logit.

At evaluation the baseline retriever returns 20 passages and the
cross-encoder reorders them. Recall@5 is computed on that order, on
the reviewed test set, once. There is no learning-rate sweep; the
spec does not ask for one. The learning rate is 2e-5 for one epoch.

    python train_reranker.py
    python train_reranker.py --smoke

`--smoke` scores two pairs and, when it can, runs one optimizer step.
The file it writes has "smoke": true. It is not a Recall@5.
"""

import argparse
import json
from pathlib import Path

from jigo_rag.config import SEED
from jigo_rag.jsonl import read_jsonl
from jigo_rag.metrics import bootstrap_mean_ci, mrr_at_k, recall_at_k
from jigo_rag.negatives import mine_hard_negatives
from jigo_rag.rerank import RERANK_DEPTH, RERANKER_MODEL, labeled_pairs, rerank
from jigo_rag.retriever import FaissRetriever


def load_corpus():
    corpus = []
    for split in ("train", "val", "test"):
        corpus.extend(read_jsonl(Path("data/chunks") / f"{split}.jsonl"))
    if not corpus:
        raise SystemExit("No chunks in data/chunks/. Run python make_data.py first.")
    return corpus


def scores_after_rerank(encoder, retriever, questions):
    """Recall@5 and MRR@10 after reordering each question's top 20."""
    recall_scores = []
    mrr_scores = []
    for row in questions:
        hits = retriever.search(row["question"], k=RERANK_DEPTH)
        ordered = rerank(encoder, row["question"], hits)
        ranked = [hit["chunk_id"] for hit in ordered]
        recall_scores.append(recall_at_k(ranked, row["gold_chunk_id"], 5))
        mrr_scores.append(mrr_at_k(ranked, row["gold_chunk_id"], 10))
    return recall_scores, mrr_scores


def train_one(pair_columns, output_dir, max_steps=None, batch_size=8):
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    model = CrossEncoder(RERANKER_MODEL, num_labels=1)
    dataset = Dataset.from_dict(pair_columns)
    loss = BinaryCrossEntropyLoss(model)
    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_steps=max_steps if max_steps else -1,
        warmup_steps=0.1,
        seed=SEED,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=10,
        report_to="none",
    )
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()
    model.save(output_dir)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(RERANKER_MODEL, num_labels=1)
        demo = rerank(
            model,
            "Where is travel booked?",
            [
                {"chunk_id": "other#0", "text": "The office closes on federal holidays."},
                {"chunk_id": "gold#0", "text": "Book travel in Concur before the trip."},
            ],
        )
        # One step on four labeled pairs built from those two sentences.
        columns = {
            "question": ["Where is travel booked?"] * 4,
            "passage": [
                "Book travel in Concur before the trip.",
                "The office closes on federal holidays.",
                "Book travel in Concur before the trip.",
                "Bring a receipt for the hotel.",
            ],
            "label": [1.0, 0.0, 1.0, 0.0],
        }
        train_one(columns, "models/smoke-reranker", max_steps=1, batch_size=2)
        report = {
            "smoke": True,
            "top_chunk_after_rerank": demo[0]["chunk_id"],
            "max_steps": 1,
            "batch_size": 2,
            "note": "Two-pair reorder plus one optimizer step. Not a Recall@5 on the test set.",
        }
        path = Path("results/smoke_reranker.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    train_path = Path("data/train.jsonl")
    reviewed = Path("data/test_reviewed.jsonl")
    if not train_path.exists() or not reviewed.exists():
        raise SystemExit(
            "Need data/train.jsonl and data/test_reviewed.jsonl. "
            "Generate questions, then review the test set."
        )
    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(reviewed)
    corpus = load_corpus()
    train_chunks = [chunk for chunk in corpus if chunk.get("split") == "train"]
    # Negatives are training chunks. The test set is only for scoring.
    miner = FaissRetriever().build(train_chunks)

    def search(question, k):
        return miner.search(question, k=k)

    mined = mine_hard_negatives(train_rows, search, train_chunks, seed=SEED)
    model = train_one(labeled_pairs(mined), "models/reranker")
    baseline = FaissRetriever().build(corpus)
    base_recall, base_mrr = scores_after_rerank(model, baseline, test_rows)
    report = {
        "smoke": False,
        "seed": SEED,
        "model": RERANKER_MODEL,
        "rerank_depth": RERANK_DEPTH,
        "baseline_retriever": {
            "test_recall@5": bootstrap_mean_ci(base_recall, seed=SEED),
            "test_mrr@10": bootstrap_mean_ci(base_mrr, seed=SEED),
        },
    }
    embed_dir = Path("models/embed")
    if embed_dir.is_dir() and any(embed_dir.iterdir()):
        tuned = FaissRetriever(model_name=str(embed_dir)).build(corpus)
        tuned_recall, tuned_mrr = scores_after_rerank(model, tuned, test_rows)
        report["finetuned_retriever"] = {
            "test_recall@5": bootstrap_mean_ci(tuned_recall, seed=SEED),
            "test_mrr@10": bootstrap_mean_ci(tuned_mrr, seed=SEED),
        }
    else:
        report["finetuned_retriever"] = None
        report["finetuned_retriever_note"] = (
            "models/embed is not present, so the reranker was not scored on the fine-tuned retriever."
        )
    path = Path("results/reranker.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
