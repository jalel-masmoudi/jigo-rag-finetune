"""Scores for one retrieval question and for a whole run.

Retrieval
- Recall@k is 1 when the gold chunk id is in the top k, else 0.
- MRR@10 is 1/rank when the gold chunk is in the top 10, else 0.
  Rank 1 is the first hit.

Answers
- Exact match and token F1 use SQuAD normalization: lowercase, strip
  punctuation, drop the articles a/an/the, collapse whitespace. Letters
  outside ASCII are kept, so "café" stays "café".
- Token F1 is the F1 of those tokens. Repeated words count with their
  multiplicity (a multiset), which is the SQuAD token F1.

Uncertainty
- A 95% confidence interval is a percentile bootstrap of the mean.
  Each resample draws the questions with replacement and takes the mean.
  The reported interval is the 2.5th and 97.5th percentiles of those means.
  The seed makes the interval reproducible. It is not a model of the
  handbook; it only says how much the mean would move if we had drawn a
  different set of the same questions.
"""

import random
import re
from collections import Counter


def normalize(text):
    """SQuAD normalization. Unicode letters are kept."""
    text = text.lower()
    text = "".join(character if character.isalnum() or character.isspace() else " " for character in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, gold):
    return int(normalize(prediction) == normalize(gold))


def token_f1(prediction, gold):
    predicted = normalize(prediction).split()
    reference = normalize(gold).split()
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(ranked_ids, gold_id, k):
    return int(gold_id in list(ranked_ids)[:k])


def mrr_at_k(ranked_ids, gold_id, k):
    for rank, chunk_id in enumerate(list(ranked_ids)[:k], start=1):
        if chunk_id == gold_id:
            return 1.0 / rank
    return 0.0


def percentile(values, percent):
    """Nearest-rank percentile. `percent` is 0-100."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = int(round((percent / 100) * (len(ordered) - 1)))
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def bootstrap_mean_ci(scores, n_resamples=1000, seed=42, alpha=0.05):
    """Return the mean and a percentile bootstrap interval."""
    if not scores:
        raise ValueError("no scores to summarize")
    rng = random.Random(seed)
    n_scores = len(scores)
    means = []
    for _ in range(n_resamples):
        sample = [scores[rng.randrange(n_scores)] for _ in range(n_scores)]
        means.append(sum(sample) / n_scores)
    means.sort()
    low_index = int((alpha / 2) * n_resamples)
    high_index = int((1 - alpha / 2) * n_resamples) - 1
    return {
        "mean": sum(scores) / n_scores,
        "ci_low": means[low_index],
        "ci_high": means[high_index],
        "n": n_scores,
        "n_resamples": n_resamples,
        "seed": seed,
    }
