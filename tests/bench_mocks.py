"""Stand-ins for a chat tokenizer and a causal LM, enough for anatta.probe.generate and anatta.bench.run, with no
network and no weights. The golden battery-run fixture (tests/fixtures/golden_battery_run.json) was recorded with
these same classes driving the original battery script, so they must not change.

MockTok is character-level: special tokens <pad>=0, <s>=1, </s>=2, [SYS] [/SYS] [INST] [/INST] [A] = 3-7, and any other
character c is 16 + ord(c). Its chat template is Mistral-like: BOS (if it has one), then a default system prompt
when the conversation does not open with a system message, then the turns, then "[A]" for the generation prompt.
apply_chat_template(tokenize=True) tokenizes the rendered text without adding special tokens, as transformers
does, and returns a dict like transformers 5's BatchEncoding; calling the tokenizer with add_special_tokens=True
prepends a BOS, which is how a double BOS happens.

MockModel answers each row from its decoded prompt with a fixed reply (reply(), or the `answer` callable given),
appends </s>, cuts at max_new_tokens, and pads finished rows with pad_token_id the way generate does. sample=True
makes every reply differ between calls (sampling left on).
"""

import hashlib
import re
from types import SimpleNamespace

import torch

SPECIAL = {"<pad>": 0, "<s>": 1, "</s>": 2, "[SYS]": 3, "[/SYS]": 4, "[INST]": 5, "[/INST]": 6, "[A]": 7}
INVERSE = {v: k for k, v in SPECIAL.items()}
OFFSET = 16
DEFAULT_SYSTEM = "You are a mock assistant. Answer carefully."


class Enc(dict):
    """A tokenizer's output: a dict that moves to a device as itself."""

    def to(self, device):
        return self


class MockTok:
    def __init__(self, bos: bool = True, default_system: str | None = DEFAULT_SYSTEM, template_bos: int | None = None):
        self.bos_token_id, self.bos_token = (1, "<s>") if bos else (None, None)
        self.eos_token_id, self.eos_token = 2, "</s>"
        self.pad_token_id, self.pad_token = 0, "<pad>"
        self.padding_side = "right"
        self.default_system = default_system
        self.template_bos = (1 if bos else 0) if template_bos is None else template_bos  # BOS strings the template writes

    def render(self, conversation, add_generation_prompt=False):
        out = "<s>" * self.template_bos
        if self.default_system is not None and (not conversation or conversation[0]["role"] != "system"):
            out += f"[SYS]{self.default_system}[/SYS]"
        for m in conversation:
            if m["role"] == "system":
                out += f"[SYS]{m['content']}[/SYS]"
            elif m["role"] == "user":
                out += f"[INST]{m['content']}[/INST]"
            else:
                out += f"{m['content']}</s>"
        return out + ("[A]" if add_generation_prompt else "")

    def encode(self, text: str) -> list[int]:
        ids, i = [], 0
        names = sorted(SPECIAL, key=len, reverse=True)
        while i < len(text):
            for s in names:
                if text.startswith(s, i):
                    ids.append(SPECIAL[s])
                    i += len(s)
                    break
            else:
                ids.append(OFFSET + ord(text[i]))
                i += 1
        return ids

    def decode_ids(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(("" if skip_special_tokens else INVERSE[t]) if t < OFFSET else chr(t - OFFSET) for t in ids)

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=True):
        single = isinstance(texts, str)
        bos = [self.bos_token_id] if add_special_tokens and self.bos_token_id is not None else []
        seqs = [bos + self.encode(t) for t in ([texts] if single else texts)]
        if single and return_tensors is None:
            return Enc(input_ids=seqs[0], attention_mask=[1] * len(seqs[0]))
        width = max(map(len, seqs)) if padding else None
        ids, mask = [], []
        for s in seqs:
            pad = width - len(s) if padding else 0
            if self.padding_side == "left":
                ids.append([self.pad_token_id] * pad + s)
                mask.append([0] * pad + [1] * len(s))
            else:
                ids.append(s + [self.pad_token_id] * pad)
                mask.append([1] * len(s) + [0] * pad)
        if return_tensors == "pt":
            return Enc(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))
        return Enc(input_ids=ids, attention_mask=mask)

    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=True, **kwargs):
        text = self.render(conversation, add_generation_prompt)
        if not tokenize:
            return text
        ids = self.encode(text)
        return Enc(input_ids=ids, attention_mask=[1] * len(ids))

    def batch_decode(self, seqs, skip_special_tokens=False):
        return [self.decode_ids(s.tolist() if hasattr(s, "tolist") else list(s), skip_special_tokens) for s in seqs]


def reply(prompt: str) -> str:
    """A fixed reply to a decoded prompt, by format: right, wrong, unparsed and over-long answers in turn."""
    h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    if 'last line as "Answer: <number>"' in prompt:
        m = re.search(r"x=(\d+)", prompt)
        n = m.group(1) if m else "0"
        return [f"Add them up.\nAnswer: {n}", f"The total is {n}.", f"Answer: **{int(n) + 1}**", "I cannot tell."][h % 4]
    if "ANSWER: $LETTER" in prompt:
        letter = "ABCD"[h % 4]
        return [f"Step by step.\nANSWER: {letter}", f"So the answer is ({letter}).", f"**ANSWER: {letter}**",
                "Let me think. " * 30][(h >> 3) % 4]
    if "Reply with the letter" in prompt:
        letter = "ABCD"[h % 4]
        return [letter, f"({letter})", f"The answer is {letter}.", "I think it is probably the third one, all told."][(h >> 3) % 4]
    return "OK."


class MockModel:
    def __init__(self, tok: MockTok, eos=2, repetition_penalty: float = 1.0, sample: bool = False, answer=None):
        self.tok, self.device, self.sample, self.n_calls = tok, "cpu", sample, 0
        self.answer = answer or reply
        self.calls: list[dict] = []
        self.generation_config = SimpleNamespace(eos_token_id=eos, repetition_penalty=repetition_penalty,
                                                 do_sample=False, temperature=None, top_p=None, top_k=None)
        self.config = SimpleNamespace(_commit_hash="0" * 40)

    def parameters(self):
        yield torch.zeros(1, dtype=torch.bfloat16)

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, temperature, top_p, pad_token_id):
        self.n_calls += 1
        self.calls.append({"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist(),
                           "max_new_tokens": max_new_tokens, "do_sample": do_sample, "temperature": temperature,
                           "top_p": top_p, "pad_token_id": pad_token_id})
        eos = self.generation_config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        rows = []
        for ids, m in zip(input_ids.tolist(), attention_mask.tolist()):
            text = self.answer(self.tok.decode_ids([t for t, k in zip(ids, m) if k]))
            if self.sample:
                text += f" (draw {self.n_calls})"
            rows.append((self.tok.encode(text) + [eos])[:max_new_tokens])
        width = max(map(len, rows))
        rows = [r + [pad_token_id] * (width - len(r)) for r in rows]
        return torch.cat([input_ids, torch.tensor(rows, dtype=input_ids.dtype)], dim=1)


# --------------------------------------------------------------------------------- synthetic dataset rows


def synthetic_rows(n: int = 30) -> dict:
    """n rows per set, shaped like the parquet rows of openai/gsm8k, TIGER-Lab/MMLU-Pro and allenai/ai2_arc, with
    questions of varied (and sometimes equal) length so the battery's length sort has ties to keep."""
    cats = ["math", "law", "computer science", "history"]
    gsm = [{"question": f"Q{i}: Tom has x={i * 7 % 50} apples" + " and more" * (i % 5) + ". How many?",
            "answer": f"Work.\n#### {i * 7 % 50 if i % 3 else 1000 + i:,}"} for i in range(n)]
    mmlu = [{"question_id": 70 + 3 * i, "question": f"M{i}: which holds" + " truly" * (i % 4) + "?",
             "options": [f"opt {j} of {i}" for j in range(10 - (i % 4))], "answer": "ABCD"[i % 4],
             "answer_index": i % 4, "cot_content": "", "category": cats[i % 4], "src": "synthetic"} for i in range(n)]
    arc = [{"id": f"ARC_{i}", "question": f"A{i}: what is" + " it" * (i % 3) + "?",
            "choices": {"text": [f"c{j}" for j in range(4)], "label": ["1", "2", "3", "4"] if i % 5 == 0 else list("ABCD")},
            "answerKey": ("1234" if i % 5 == 0 else "ABCD")[i % 4]} for i in range(n)]
    return {"gsm8k": gsm, "mmlu_pro": mmlu, "arc_challenge": arc}
