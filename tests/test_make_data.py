"""make_data writes where --out says, not to a fixed directory."""

from make_data import build


def test_build_honors_the_output_directory(tmp_path, monkeypatch):
    documents = [
        {
            "doc_id": f"gsa-handbook/page-{i}.md",
            "source": "gsa-handbook",
            "doc_path": f"page-{i}.md",
            "text": "Employees book travel in Concur before the trip starts.",
        }
        for i in range(10)
    ]
    monkeypatch.setattr("make_data.load_documents", lambda raw_root: (documents, []))
    monkeypatch.setattr("make_data.load_tokenizer", lambda: None)
    monkeypatch.setattr(
        "make_data.chunk_text",
        lambda text, tokenizer: [{"n": 0, "text": text, "n_tokens": 8}],
    )
    out = tmp_path / "custom-chunks"
    build(tmp_path / "raw", out, seed=1)
    assert (out / "train.jsonl").exists()
    assert (out / "val.jsonl").exists()
    assert (out / "test.jsonl").exists()
    assert (out / "stats.json").stat().st_size > 0
