"""The document split is deterministic and does not leak a document."""

from jigo_rag.split import (
    assert_no_doc_leakage,
    assign_splits,
    split_doc_ids,
    split_doc_ids_stratified,
)


def test_split_is_deterministic_and_exhaustive():
    doc_ids = [f"doc-{i}" for i in range(100)]
    first = split_doc_ids(doc_ids, seed=42)
    second = split_doc_ids(doc_ids, seed=42)
    assert first == second
    assert split_doc_ids(doc_ids, seed=42) != split_doc_ids(doc_ids, seed=7)
    train_ids, val_ids, test_ids = first
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(val_ids) | set(test_ids) == set(doc_ids)
    # 100 docs at 70/10/20, rounded.
    assert len(test_ids) == 20
    assert len(val_ids) == 10
    assert len(train_ids) == 70


def test_each_source_is_split_on_its_own():
    # Nine documents is enough for 10% to round to one validation doc.
    jigo = [f"jigo/doc-{i}.md" for i in range(9)]
    handbook = [f"gsa-handbook/page-{i}.md" for i in range(40)]
    train_ids, val_ids, test_ids = split_doc_ids_stratified(jigo + handbook, seed=42)
    assert any(doc_id.startswith("jigo/") for doc_id in val_ids)
    assert any(doc_id.startswith("jigo/") for doc_id in test_ids)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(val_ids) | set(test_ids) == set(jigo + handbook)


def test_chunk_rows_cannot_put_one_doc_in_two_splits():
    doc_ids = [f"handbook/page-{i}.md" for i in range(30)]
    mapping = assign_splits(doc_ids, seed=42)
    rows = []
    for doc_id, split in mapping.items():
        # Two chunks of the same document must share its split.
        for index in (0, 1):
            rows.append({"doc_id": doc_id, "split": split, "chunk_id": f"{doc_id}#{index}"})
    owners = assert_no_doc_leakage(rows)
    assert set(owners) == set(doc_ids)

    leaked = [dict(rows[0]), dict(rows[1])]
    leaked[1]["split"] = "test" if leaked[0]["split"] != "test" else "train"
    try:
        assert_no_doc_leakage(leaked)
    except AssertionError as exc:
        assert leaked[0]["doc_id"] in str(exc)
    else:
        raise AssertionError("leakage was not detected")
