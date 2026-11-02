"""Log-likelihood multiple choice for MMLU, as lm-evaluation-harness's mmlu task defines it in its files
(lm_eval/tasks/mmlu/default/_default_template_yaml and the per-subject files, at commit 314f717):

  - 0-shot: the subject's description, "The following are multiple choice questions (with answers) about
    {subject}.\\n\\n" with the subject's underscores as spaces (cais/mmlu stores "abstract_algebra", lm-eval's file
    says "abstract algebra"), then "{question.strip()}\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nAnswer:";
  - the four continuations " A" to " D" (lm-eval's target delimiter, a space, then the choice). Each must be one
    token, which letter_ids checks, so lm-eval's acc (the argmax of the continuations' log-likelihoods) is the argmax
    of the logits at the prompt's last token over those four ids; ties go to the first;
  - no chat template, the tokenizer's default special tokens (Qwen's adds none; Ministral's adds one BOS, which
    lm-eval's add_special_tokens=False would not: a deviation calibration check 3 keeps because it is the same in
    every condition);
  - a micro-average over every question (accuracy), as lm-eval's weight_by_size gives.

Batches are right-padded, and each row is read at its own last real token (its attention-mask length - 1), never at
index -1, which under right padding is a pad for every row shorter than the longest. chat=True renders each prompt as
one user turn through the model's chat template with the generation prompt (anatta.models.chat_ids) and scores the
same four tokens.
"""

from __future__ import annotations

import torch

from ..models import as_messages, chat_ids

LETTERS = ("A", "B", "C", "D")
DESCRIPTION = "The following are multiple choice questions (with answers) about {subject}.\n\n"


def subject_name(subject: str) -> str:
    """cais/mmlu's subject as lm-eval's per-subject file writes it: underscores as spaces."""
    return subject.replace("_", " ")


def mmlu_prompt(question: str, choices: list[str], subject: str) -> str:
    """One question as lm-eval's 0-shot mmlu context: the description, then doc_to_text."""
    if len(choices) != len(LETTERS):
        raise ValueError(f"MMLU questions have {len(LETTERS)} choices, not {len(choices)}")
    body = (f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:")
    return DESCRIPTION.format(subject=subject_name(subject)) + body


def letter_ids(tok) -> list[int]:
    """The ids of the continuations " A" to " D", each required to be exactly one token (without special tokens)."""
    seqs = [list(tok(" " + c, add_special_tokens=False)["input_ids"]) for c in LETTERS]
    bad = {c: s for c, s in zip(LETTERS, seqs) if len(s) != 1}
    if bad:
        raise ValueError(f"each continuation must be one token, but {bad} are not: lm-eval's acc would sum several "
                         "tokens' log-likelihoods, which the last-token argmax does not")
    return [s[0] for s in seqs]


def encode(tok, texts: list[str], chat: bool = False) -> list[list[int]]:
    """Each prompt's ids: tokenized with the tokenizer's default special tokens, or, with chat=True, rendered as one
    user turn by the chat template with the generation prompt."""
    if chat:
        return [chat_ids(tok, as_messages(t))[0].tolist() for t in texts]
    return [list(tok(t)["input_ids"]) for t in texts]


def pad_right(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_ids, attention_mask) with every row padded on the right to the longest."""
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, :len(r)] = 1
    return ids, mask


def last_real_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Each right-padded row's last real token: its attention-mask length - 1."""
    return attention_mask.sum(dim=1) - 1


def letter_logits(model, ids: torch.Tensor, mask: torch.Tensor, letters: list[int]) -> torch.Tensor:
    """The logits of the four continuation ids at each row's last real token, as float32 ([rows, 4]; the cast from
    bf16 is exact)."""
    logits = model(input_ids=ids, attention_mask=mask).logits
    rows = torch.arange(ids.shape[0], device=logits.device)
    return logits[rows, last_real_index(mask).to(logits.device)][:, letters].float()


def first_argmax(values: list[float]) -> int:
    """The index of the largest value, the first one on ties."""
    return max(range(len(values)), key=values.__getitem__)


def predict(model, tok, texts: list[str], letters: list[int], batch_size: int = 32, chat: bool = False,
            log=None) -> dict:
    """The predicted letter index for every prompt, in batches of `batch_size`, right-padded, with whatever hooks
    the caller has installed on. Returns {"preds": [index], "logits": [[four floats]]}."""
    rows = encode(tok, texts, chat)
    device = next(model.parameters()).device
    preds, values = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            ids, mask = pad_right(rows[start:start + batch_size], tok.pad_token_id)
            ll = letter_logits(model, ids.to(device), mask.to(device), letters).cpu().tolist()
            values += ll
            preds += [first_argmax(v) for v in ll]
            if log and (start // batch_size) % 100 == 0:
                log(f"mc: {start + len(ll)}/{len(rows)}")
    return {"preds": preds, "logits": values}


def accuracy(preds: list[int], answers: list[int]) -> float:
    """The share of questions answered right: a micro-average over every question."""
    if len(preds) != len(answers) or not answers:
        raise ValueError(f"{len(preds)} predictions for {len(answers)} answers")
    return sum(int(p) == int(a) for p, a in zip(preds, answers)) / len(answers)
