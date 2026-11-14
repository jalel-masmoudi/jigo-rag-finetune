# Data

This file is the record of where the text came from, how it was cut up, and how to rebuild the files. The numbers below are from one real run of `python make_data.py` on the pinned commits.

## Sources

| Source | URL | Pinned commit | License |
| --- | --- | --- | --- |
| GSA TTS Handbook | https://github.com/GSA-TTS/handbook | `c220f93896b39e9e4ab0f486a1d5bf346ac06e3e` | CC0 1.0, and a work of the United States Government (public domain in the United States). See `LICENSE.md` in that repo. |
| Jigo AI Workspace | https://github.com/jalel-masmoudi/Jigo-AI-workspace | `e9d7d9eab21db78211c7c26b3831632d6e31147f` | MIT. Copyright 2026 Jalel Masmoudi. |

The same two commits are the constants in `jigo_rag/config.py`. `make_data.py` checks them out. A later commit on either GitHub repo is not picked up until those constants change, so a rebuild stays on the same text.

Jigo has no chunker, no embedding step, and no vector store in that repo, so this pipeline uses the fallback in the project spec: 512-token windows with a 64-token overlap.

At the pinned Jigo commit there are 10 markdown files, not 11. One is `.github/PULL_REQUEST_TEMPLATE.md`. That path is skipped. The other nine are kept:

- `README.md`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `docs/README.md`
- `docs/superpowers/plans/2026-07-09-production-integrations.md`
- `docs/superpowers/plans/2026-07-28-persistent-workspace.md`
- `docs/superpowers/specs/2026-07-28-persistent-workspace-design.md`
- `src/routes/README.md`

The handbook repo has 250 markdown files. Only `pages/**/*.md` (242 files) is treated as handbook content. The other eight are repository notes, not chapters: `README.md`, `CONTRIBUTING.md`, `LICENSE.md`, `SECURITY.md`, three files under `.github/`, and `terraform/teams-structure.md`.

## Filtering

`jigo_rag/corpus.py` cleans each file before chunking.

1. A leading YAML front matter block (`---` … `---`) is removed. Titles and permalinks live there, and the body is the page.
2. Liquid comments (`{% comment %}…{% endcomment %}`), HTML comments, and other Liquid tags (`{% … %}`) are removed. The handbook is an Eleventy site. Those tags are navigation and shortcodes, not sentences. A shortcode that only named a Slack channel disappears, so a few sentences have a blank where the channel name was.
3. Markdown link URLs are dropped. The link label is kept, because the label is usually the words a person would read.
4. **Empty.** After that cleanup, a page with fewer than 40 letters or digits is dropped. These seven handbook pages are empty (a redirect, or a title that existed only in the front matter):
   - `pages/about-us/mission.md`
   - `pages/general-information-and-resources/business-and-ops-policies/billing_policy.md`
   - `pages/general-information-and-resources/tech-policies/open-source.md`
   - `pages/redirects/buddy-handbook.md`
   - `pages/redirects/employee-assistance-program.md`
   - `pages/tools/vmware-fusion.md`
   - `pages/tools/vmware-horizon.md`
5. **Nav stub.** A page with at least 8 markdown links, and fewer than 200 letters or digits outside those links, is a table of contents. One page matched: `pages/travel-and-leave/travel-guide-table-of-contents.md`. A short real page with a few links is kept, because the prose outside the links is what a question would be about.

Kept: 234 handbook pages + 9 Jigo docs = **243 documents**. Dropped: 7 empty + 1 nav stub = **8 documents**.

## Chunking

- Tokenizer: `BAAI/bge-small-en-v1.5` (the baseline retriever). "Token" means one id from that tokenizer, not one word.
- Window: **512 tokens**. That is the longest sequence the embedding model accepts, so a chunk is one full model input.
- Overlap: **64 tokens**. The next window starts 448 tokens later (512 − 64). A sentence that sits on the cut is still whole in the next window. 64 tokens is about one or two sentences, so the index does not nearly double.
- The stored text is a slice of the original page, using the tokenizer's character offsets. The tokenizer lowercases when it counts, but the file keeps the original wording. `n_tokens` is the size of the window at the cut, not a second tokenization of the decoded string.
- `[CLS]` and `[SEP]` are not part of the 512. The retriever adds them later, so the last token or two of a full window can be truncated when the vector is built.
- Chunk id: `<source>/<doc_path>#<n>`, for example `gsa-handbook/pages/about.md#0`. `n` is the window number from the start of the document. The doc id is the same string without `#<n>`. The same page and the same tokenizer always produce the same ids.

## Split

Documents are shuffled with `random.Random(42)` after sorting the ids, then cut into 70% train, 10% validation, 20% test. The fractions are of **documents**, not of chunks. Rounding on this corpus gives:

| Split | Documents | Chunks |
| --- | ---: | ---: |
| train | 170 | 519 |
| val | 24 | 81 |
| test | 49 | 175 |
| total | 243 | 775 |

`make_data.py` asserts that no `doc_id` appears in two splits, and the tests check the same rule.

Why not split by chunk: neighboring windows of one page share 64 tokens and usually the same facts. If one window is in train and the next is in test, a model can memorize the page and score well on a question it has essentially already seen. Keeping every chunk of a document in one split blocks that leak. The retrieval index at evaluation time still contains every chunk. The split only decides which documents may be used to train.

The test split has 175 chunks. The question generator asks for at most 4 questions per chunk, and for this count it asks for 3 (`ceil(360 / 175) = 3`). 175 × 3 = 525, which is more than the 360 test questions, so 70/10/20 already leaves enough test material for a 300–400 question review set. The fractions were not changed after the count.

## Synthetic questions

`generate_qa.py` calls the Gemini API with the `google-genai` SDK. The key is read from the environment variable `GEMINI_API_KEY` (a `.env` file is loaded only to fill variables that are not already set). Nothing in the repo contains a key. `.env` is gitignored. `.env.example` shows the variable names.

The model id is `GEMINI_MODEL`, default **`gemini-3.5-flash-lite`**. On the Gemini API pricing page that model is a Flash-Lite model whose input and output are free of charge on the free tier. If Google renames it, set `GEMINI_MODEL`. The prompt is `prompts/qa_generate.txt` and is not built in Python, so the wording stays fixed.

Each request asks for several questions about **one** chunk. The script sets `gold_chunk_id` to that chunk. It does not trust the model to copy ids. An answer is kept only when most of its words appear in the chunk (`jigo_rag/qa.py`). The prompt says the question must be answerable from that passage alone.

Targets: 1,440 train + 200 validation + 360 test = 2,000. Test is generated first, because those are the rows a person reviews, and a free-tier daily cap may stop the run. Finished chunks are appended to `data/{train,val,test}.jsonl` and skipped on the next run. Wait between calls with `GEMINI_SLEEP_SECONDS` (default 4). A daily quota error stops the process without deleting what was already written. Run the same command again the next day.

This workspace did not have a `GEMINI_API_KEY`, so that script has not been run. The 20 rows in `data/sample/` are hand-written from real chunks so the file format is visible. Each answer is a copied span of its chunk. They are not model output and they are not the evaluation set.

## Hand review

`review_test.py` shows one test row at a time. `k` keeps it, `f` lets you replace the question or the answer, `d` drops it, `q` stops. Kept and fixed rows are appended to `data/test_reviewed.jsonl`. Dropped rows are appended to `data/test_dropped.jsonl` so they are not shown again. Progress is saved after each row.

**Only `data/test_reviewed.jsonl` is the test set.** `data/test.jsonl` is raw model output and must not be used for a score. The reviewed file is not gitignored, so it can be committed after the review. The raw generated files and the cloned corpora are gitignored.

## How to rebuild

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then put the key in .env; do not commit it
.venv/bin/python make_data.py
.venv/bin/python generate_qa.py
.venv/bin/python review_test.py
```

`make_data.py` clones into `data/raw/` and writes `data/chunks/{train,val,test}.jsonl` plus `data/stats.json`. The same seed (42) and the same pinned commits reproduce the split. Question text depends on the model, so a second Gemini run is not guaranteed to match the first word for word. The script is resumable so you do not have to throw a partial file away.

`pytest` checks window size and overlap, that the split does not change with the seed, that a document cannot land in two splits, and that the JSONL rows have the fields the later scripts read.
