"""Not-found contexts and the collate the training loop actually calls."""

from jigo_rag.generator import NOT_FOUND
from jigo_rag.lora_data import build_examples, collate_for_training, contexts_for_row, with_answer


def _row():
    return {
        "question": "Where is travel booked?",
        "answer": "In Concur.",
        "chunk_text": "Book travel in Concur.",
        "doc_id": "train-a",
        "gold_chunk_id": "train-a#0",
    }


def _train_corpus():
    return [
        {"chunk_id": "train-a#0", "doc_id": "train-a", "text": "Book travel in Concur.", "split": "train"},
        {"chunk_id": "train-b#0", "doc_id": "train-b", "text": "Leave is requested separately.", "split": "train"},
        {"chunk_id": "train-c#0", "doc_id": "train-c", "text": "Badges are issued at the desk.", "split": "train"},
    ]


def test_not_found_context_is_train_only_and_excludes_the_gold_chunk():
    hits = [
        {"chunk_id": "test-z#0", "doc_id": "test-z", "text": "A held-out page."},
        {"chunk_id": "train-a#1", "doc_id": "train-a", "text": "Same page, next window."},
        {"chunk_id": "train-b#0", "doc_id": "train-b", "text": "Leave is requested separately."},
        {"chunk_id": "val-y#0", "doc_id": "val-y", "text": "A validation page."},
    ]
    answer, refused = contexts_for_row(_row(), hits, _train_corpus(), k=2)
    assert answer[0] == "Book travel in Concur."
    assert "A held-out page." not in answer
    assert "A validation page." not in answer
    assert "A held-out page." not in refused
    assert "Book travel in Concur." not in refused
    assert "Same page, next window." not in refused
    assert "Leave is requested separately." in refused
    examples = build_examples([_row()], _train_corpus(), lambda question, k: hits, k=2)
    assert examples[0]["messages"][-1]["content"] == "In Concur."
    assert examples[1]["messages"][-1]["content"] == NOT_FOUND
    assert "Book travel in Concur." in examples[0]["messages"][1]["content"]
    assert "Book travel in Concur." not in examples[1]["messages"][1]["content"]


class _Tokenizer:
    """Splits on spaces and can turn ids back into those words."""

    pad_token_id = 0

    def __init__(self):
        self.ids = {}
        self.words = {}

    def _id(self, token):
        if token not in self.ids:
            value = len(self.ids) + 1
            self.ids[token] = value
            self.words[value] = token
        return self.ids[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            # The assistant turn starts with the same header the generation
            # prompt adds, so the prompt is a prefix of the full string.
            if message["role"] == "assistant":
                parts.append("ASSIST")
            parts.append(message["content"])
        if add_generation_prompt:
            parts.append("ASSIST")
        return " ".join(parts)

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [self._id(token) for token in text.split() if token]}

    def decode(self, ids):
        return " ".join(self.words[token] for token in ids if token in self.words)


def test_gold_passage_position_depends_on_the_seed():
    hits = [
        {"chunk_id": "train-b#0", "doc_id": "train-b", "text": "Leave is requested separately."},
        {"chunk_id": "train-c#0", "doc_id": "train-c", "text": "Badges are issued at the desk."},
    ]
    positions = set()
    for seed in range(12):
        answer, _refused = contexts_for_row(_row(), hits, _train_corpus(), k=3, seed=seed)
        positions.add(answer.index("Book travel in Concur."))
    assert len(positions) > 1
    first = contexts_for_row(_row(), hits, _train_corpus(), k=3, seed=4)
    second = contexts_for_row(_row(), hits, _train_corpus(), k=3, seed=4)
    assert first[0] == second[0]


def test_loss_log_and_seed(tmp_path):
    import torch
    from train_lora import append_loss, set_training_seed, start_loss_log

    set_training_seed(7)
    first = torch.rand(2).tolist()
    set_training_seed(7)
    assert torch.rand(2).tolist() == first
    csv_path = start_loss_log(7, directory=tmp_path)
    append_loss(csv_path, 1, "0.500000", 7)
    assert "1,0.500000,7" in csv_path.read_text(encoding="utf-8")
    config = __import__("json").loads((tmp_path / "lora_config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 7 and config["r"] == 16 and config["lora_alpha"] == 32


def _decoded(tokenizer, batch):
    ids = [token for token in batch["input_ids"][0].tolist() if token != 0]
    return ids, tokenizer.decode(ids)


def test_collate_drops_non_gold_passages_before_cutting_the_gold():
    gold = " ".join(f"goldword{index}" for index in range(12))
    high = " ".join(f"highrank{index}" for index in range(12))
    low = " ".join(f"lowrank{index}" for index in range(12))
    passages = [high, gold, low]
    tokenizer = _Tokenizer()

    def example(kept):
        return {
            "messages": with_answer("UNIQUE_QUESTION", kept, "SHORT_ANSWER"),
            "passages": kept,
            "gold_passage": gold,
        }

    gold_only = collate_for_training([example([gold])], tokenizer, max_length=10000)
    gold_len = int(gold_only["input_ids"].shape[-1])
    with_high = collate_for_training([example([high, gold])], tokenizer, max_length=10000)
    with_high_len = int(with_high["input_ids"].shape[-1])

    # Room for the higher non-gold passage and the gold one, not the lowest.
    kept = collate_for_training([example(passages)], tokenizer, max_length=with_high_len)
    _ids, text = _decoded(tokenizer, kept)
    assert all(word in text.split() for word in gold.split())
    assert "highrank0" in text.split()
    assert "lowrank0" not in text.split()

    # Gold alone is over this cap, so it is the text that gets cut.
    cut = collate_for_training([example(passages)], tokenizer, max_length=gold_len - 5)
    _ids, text = _decoded(tokenizer, cut)
    assert "lowrank0" not in text.split()
    assert "highrank0" not in text.split()
    assert "goldword0" in text.split()
    assert "goldword11" not in text.split()
    assert "UNIQUE_QUESTION" in text.split()


def test_collate_clips_passages_and_keeps_the_question():
    passage = " ".join(f"passage{index}" for index in range(80))
    example = {"messages": with_answer("UNIQUE_QUESTION", [passage], "SHORT_ANSWER")}
    tokenizer = _Tokenizer()
    batch = collate_for_training([example], tokenizer, max_length=40)
    ids = [token for token in batch["input_ids"][0].tolist() if token != 0]
    text = tokenizer.decode(ids)
    assert len(ids) <= 40
    assert "UNIQUE_QUESTION" in text.split()
    assert "ASSIST" in text.split()
    assert "SHORT_ANSWER" in text.split()
    assert text.index("UNIQUE_QUESTION") < text.index("ASSIST") < text.index("SHORT_ANSWER")
    assert "passage79" not in text.split()
    question_id = tokenizer.ids["UNIQUE_QUESTION"]
    labels = batch["labels"][0].tolist()
    assert labels[ids.index(question_id)] == -100


def test_real_qwen_tokenizer_collates_committed_top5_examples():
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    rows = [
        json.loads(line)
        for line in Path("tests/fixtures/lora_top5.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 5
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    examples = [
        {"messages": with_answer(row["question"], row["passages"], row["answer"])}
        for row in rows
    ]
    batch = collate_for_training(examples, tokenizer)
    assert int(batch["input_ids"].shape[-1]) <= 3072
    # A tight cap has to drop passage text. The question and the
    # assistant header still have to be in the row.
    tight = collate_for_training([examples[0]], tokenizer, max_length=512)
    text = tokenizer.decode(tight["input_ids"][0].tolist())
    assert rows[0]["question"] in text
    header = "<|im_start|>assistant"
    assert header in text
    assert text.index(rows[0]["question"]) < text.index(header)
    assert any(token != -100 for token in tight["labels"][0].tolist())
    # The last passage is past the cap, so it is the part that was cut.
    assert rows[0]["passages"][-1][-80:] not in text


def test_collate_masks_the_prompt_and_keeps_the_answer():
    example = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "Where?"},
            {"role": "assistant", "content": "In Concur."},
        ]
    }
    tokenizer = _Tokenizer()
    batch = collate_for_training([example], tokenizer)
    prompt = tokenizer.apply_chat_template(example["messages"][:-1], tokenize=False, add_generation_prompt=True)
    prompt_length = len(tokenizer(prompt)["input_ids"])
    labels = batch["labels"][0].tolist()
    assert labels[:prompt_length] == [-100] * prompt_length
    assert any(token != -100 for token in labels[prompt_length:])


def test_train_stops_when_cuda_is_missing(monkeypatch):
    import torch

    from train_lora import train

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        train()
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("train() returned without CUDA")
    assert "CUDA" in message


def test_train_records_the_seed_and_the_step_loss(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import peft
    import torch
    import transformers

    import train_lora

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    row = {
        "question": "Where is travel booked?",
        "answer": "In Concur.",
        "chunk_text": "Book travel in Concur.",
        "doc_id": "handbook/travel.md",
        "gold_chunk_id": "handbook/travel.md#0",
    }
    corpus = [
        {"chunk_id": "handbook/travel.md#0", "doc_id": "handbook/travel.md", "text": row["chunk_text"], "split": "train"},
        {"chunk_id": "handbook/leave.md#0", "doc_id": "handbook/leave.md", "text": "Leave is separate.", "split": "train"},
    ]
    monkeypatch.setattr(train_lora, "load_training_rows", lambda: ([row], corpus))
    monkeypatch.setattr(train_lora, "_search_for", lambda rows, chunks: (chunks, lambda question, k: chunks))
    monkeypatch.setattr(train_lora, "four_bit_config", lambda: object())

    class Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class Loss:
        def item(self):
            return 0.5

        def backward(self):
            return None

    class Model:
        device = "cpu"

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def train(self):
            return None

        def print_trainable_parameters(self):
            return None

        def __call__(self, **batch):
            return type("Out", (), {"loss": Loss()})()

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class OneBatch:
        def __init__(self, *args, **kwargs):
            return None

        def __iter__(self):
            yield {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tok())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: Model())
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", lambda model: model)
    monkeypatch.setattr(peft, "get_peft_model", lambda model, config: model)
    monkeypatch.setattr(torch.utils.data, "DataLoader", OneBatch)
    train_lora.train()
    csv_text = (tmp_path / "results" / "lora_loss.csv").read_text(encoding="utf-8")
    assert csv_text.startswith("step,loss,seed\n")
    assert "1,0.500000,42" in csv_text
    config = json.loads((tmp_path / "results" / "lora_config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 42
    assert config["r"] == 16
    assert config["max_length"] == 3072
