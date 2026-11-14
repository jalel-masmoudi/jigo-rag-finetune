"""Split documents into train, validation, and test.

The split is by document, not by chunk. Neighboring chunks of the
same page share 64 tokens and usually the same facts. If one of them
is in train and the other is in test, a model can memorize the page
and look strong on the test set without answering a new page. Keeping
every chunk of a document in one split blocks that leak.

The lists are built from a sorted id list shuffled with `random.Random(seed)`,
so the same documents and the same seed always give the same split.

The split is done separately inside each source (the part of the doc id
before the first slash). The Jigo set is only nine documents. If those
nine were mixed with the handbook and then cut 70/10/20, validation can
easily get none of them. Cutting each source on its own keeps the same
fractions and, for nine Jigo documents, puts one of them in validation
and two in test.
"""

import random

from jigo_rag.config import SEED, TEST_FRACTION, TRAIN_FRACTION, VAL_FRACTION


def split_doc_ids(
    doc_ids,
    seed=SEED,
    train_fraction=TRAIN_FRACTION,
    val_fraction=VAL_FRACTION,
    test_fraction=TEST_FRACTION,
):
    """Return (train_ids, val_ids, test_ids). Every input id is in one list."""
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train, val, and test fractions must sum to 1")
    ids = sorted(set(doc_ids))
    random.Random(seed).shuffle(ids)
    n_docs = len(ids)
    if n_docs == 0:
        return [], [], []
    n_test = int(round(n_docs * test_fraction))
    n_val = int(round(n_docs * val_fraction))
    if n_test + n_val > n_docs:
        n_test = min(n_test, n_docs)
        n_val = n_docs - n_test
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    return train_ids, val_ids, test_ids


def source_of(doc_id):
    """The corpus name is the first path segment of the doc id."""
    return doc_id.split("/", 1)[0]


def split_doc_ids_stratified(doc_ids, **kwargs):
    """Split each source with `split_doc_ids`, then combine the lists.

    A source with only a few documents can round to an empty validation
    slice (10% of 4 is 0). Nine documents round to one validation
    document, which is why the Jigo docs are split on their own.
    """
    groups = {}
    for doc_id in doc_ids:
        groups.setdefault(source_of(doc_id), []).append(doc_id)
    train_ids, val_ids, test_ids = [], [], []
    for source in sorted(groups):
        train_part, val_part, test_part = split_doc_ids(groups[source], **kwargs)
        train_ids.extend(train_part)
        val_ids.extend(val_part)
        test_ids.extend(test_part)
    return train_ids, val_ids, test_ids


def assign_splits(doc_ids, **kwargs):
    """Map each doc_id to 'train', 'val', or 'test'."""
    train_ids, val_ids, test_ids = split_doc_ids_stratified(doc_ids, **kwargs)
    mapping = {}
    for doc_id in train_ids:
        mapping[doc_id] = "train"
    for doc_id in val_ids:
        mapping[doc_id] = "val"
    for doc_id in test_ids:
        mapping[doc_id] = "test"
    return mapping


def assert_no_doc_leakage(rows):
    """Raise if any doc_id is present in more than one split.

    Called on the rows that are about to be written, so a bug in the
    assignment cannot silently produce a leaked file.
    """
    owner = {}
    for row in rows:
        doc_id = row["doc_id"]
        split = row["split"]
        previous = owner.get(doc_id)
        if previous is not None and previous != split:
            raise AssertionError(f"{doc_id} is in both {previous} and {split}")
        owner[doc_id] = split
    return owner
