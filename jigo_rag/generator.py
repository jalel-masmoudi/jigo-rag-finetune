"""Baseline answerer: Qwen2.5-1.5B-Instruct.

The model sees the retrieved passages and the question. It is told to
answer from that text only, and to reply with exactly "not found" when
the answer is not there. That phrase is the same string the LoRA data
uses, so the baseline and the fine-tune are judged on the same behavior.

Token counts are the prompt tokens plus the newly generated tokens.
They come from this model's tokenizer, not from the bge tokenizer.
"""

NOT_FOUND = "not found"
GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "Answer the question using only the context. "
    "If the context does not contain the answer, reply with exactly: not found"
)


def format_context(passages):
    return "\n\n".join(f"[{index}] {text}" for index, text in enumerate(passages, start=1))


def build_messages(question, passages):
    user = f"Context:\n{format_context(passages)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class QwenGenerator:
    def __init__(self, model_name=GENERATOR_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # float16 on CUDA. On CPU, float32 is the path that actually runs;
        # float16 kernels are incomplete there.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()
        return self

    def answer(self, question, passages, max_new_tokens=128):
        import torch

        if self.model is None:
            self.load()
        messages = build_messages(question, passages)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "answer": text,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(new_tokens.shape[0]),
        }
