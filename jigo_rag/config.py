"""Numbers that define the data pipeline.

They live in one file so the setup can be read without hunting
through the scripts. DATA.md repeats the source pins and the reasons.
"""

# Checked out by make_data.py. Both commits were HEAD when this file
# was written; later commits of those repos are ignored on purpose.
SOURCES = (
    {
        "name": "gsa-handbook",
        "url": "https://github.com/GSA-TTS/handbook",
        "sha": "c220f93896b39e9e4ab0f486a1d5bf346ac06e3e",
        "license": "CC0 1.0 and United States public domain",
    },
    {
        "name": "jigo",
        "url": "https://github.com/jalel-masmoudi/Jigo-AI-workspace",
        "sha": "e9d7d9eab21db78211c7c26b3831632d6e31147f",
        "license": "MIT",
    },
)

# bge-small-en-v1.5 reads at most 512 tokens. The chunker uses that
# model's tokenizer so "512 tokens" means the same unit the retriever
# will see. Overlap is 64 tokens: about one or two sentences, so a
# fact that sits on a boundary is still whole in one of the two windows.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"

# Fractions are of documents, not of chunks. split.py explains why.
SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10
TEST_FRACTION = 0.20

# 1440 + 200 + 360 = 2000 questions. The test target sits inside the
# 300-400 questions a person can hand-check. At most 4 questions are
# asked of one chunk, so a short passage is not forced to invent facts.
QA_TARGETS = {"train": 1440, "val": 200, "test": 360}
QA_MAX_PER_CHUNK = 4

# Free-tier Flash-Lite model id from the Gemini API pricing page
# (input and output listed as free of charge). Override with GEMINI_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
