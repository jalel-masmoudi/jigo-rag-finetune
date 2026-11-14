"""Cut a document into fixed token windows.

The window is 512 tokens with a 64-token overlap. Token counts come
from the BAAI/bge-small-en-v1.5 tokenizer, which is the tokenizer of
the baseline retriever. That model accepts 512 tokens, so a chunk is
one full input and not an arbitrary number of words.

Overlap is there because a sentence can sit on the cut. The next
window starts 64 tokens before the cut, so that sentence is whole in
the next chunk. 64 is small next to 512, so the index does not grow
much: the step between windows is 512 - 64 = 448 tokens.
"""

from functools import lru_cache

from jigo_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, TOKENIZER_NAME


def chunk_ranges(n_tokens, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Return (start, end) indexes into a token sequence.

    Each window has at most `size` tokens. The next window starts
    `size - overlap` tokens later, so the two windows share `overlap`
    tokens when the first window is full. The last window is shorter
    when the document runs out. A short document is one window.
    """
    if overlap < 0 or size <= overlap:
        raise ValueError("size must be greater than overlap, and overlap must be >= 0")
    if n_tokens <= 0:
        return []
    step = size - overlap
    ranges = []
    start = 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        ranges.append((start, end))
        if end == n_tokens:
            break
        start += step
    return ranges


def chunk_token_ids(token_ids, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Return the token-id windows described by `chunk_ranges`."""
    return [list(token_ids[start:end]) for start, end in chunk_ranges(len(token_ids), size, overlap)]


@lru_cache(maxsize=1)
def load_tokenizer():
    # Imported here so tests of chunk_token_ids do not download anything.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # 512 is the longest sequence the embedding model can read. This
    # tokenizer is also used to count a whole page before it is cut,
    # and that page is often longer than 512. Raising the limit here
    # only disables the "longer than the model" warning. The windows
    # written to disk are still CHUNK_SIZE tokens.
    tokenizer.model_max_length = max(int(tokenizer.model_max_length or 0), 10**6)
    return tokenizer


def chunk_text(text, tokenizer=None, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk a string. Each item is {n, text, n_tokens}.

    `n` is the window index and becomes the suffix of the chunk id.
    `n_tokens` is the number of ids in the window before decoding.
    Decoding can change whitespace; the stored count is the cut, not
    a second tokenization of the decoded string.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    # Special tokens ([CLS], [SEP]) are not part of the 512. The
    # retriever adds them later, so the very end of a full window may
    # be truncated by one or two tokens at embedding time.
    # Offsets point back into `text`, so the stored chunk keeps the
    # original wording. Decoding the ids would lowercase and split
    # punctuation, because this tokenizer is uncased.
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    chunks = []
    for index, (start, end) in enumerate(chunk_ranges(len(token_ids), size, overlap)):
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        piece = text[char_start:char_end].strip()
        if not piece:
            continue
        chunks.append({"n": index, "text": piece, "n_tokens": end - start})
    return chunks
