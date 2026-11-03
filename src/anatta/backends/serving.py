"""The serving check's fixed inputs and its comparison (pure Python, so it is tested offline): do vLLM (anatta.backends.vllm)
and the HF path (anatta.backends.hf) give the same greedy tokens on the same prompt ids? The Modal side is
serving_app.py. Claude (Opus 5.5) wrote this file, 3 Nov 2026.

Method, after K6b's serving check (calibration/k6b_ministral/run_k6b.py compare_check; results calib/k6b/
serving_check.json): fixed harmless prompts, greedy decoding on both paths, the prompt tokens compared, the outputs
compared. K6b compared the first 100 characters of 8 prompts and admitted vLLM with at most one mismatch; this check
compares tokens: 16 prompts, the first 64 new tokens of each, and a configuration passes when every prompt's ids are
identical on both paths and at least 14 of the 16 outputs share their first 32 tokens exactly.

Token comparison: each output is cut at its first end token (the end token itself left out), so a reply that stops
early matches only a reply that stops at the same place.
"""

from __future__ import annotations

from pathlib import PurePosixPath

PREFIX, WINDOW, MIN_IDENTICAL = 32, 64, 14
# 16 neutral prompts, written for this check (Claude); no benchmark items.
PROMPTS = (
    "Name three colours of the rainbow.",
    "What is the capital of Japan?",
    "Explain in two sentences why the sky looks blue.",
    "Write a haiku about autumn leaves.",
    "List the first five prime numbers.",
    "How many days are there in a leap year?",
    "Give one tip for keeping a houseplant healthy.",
    "What does a thermometer measure?",
    "Translate 'good morning' into French.",
    "Describe a bicycle to someone who has never seen one, in one sentence.",
    "What is 12 multiplied by 7?",
    "Suggest a name for a friendly cat.",
    "Why do people use umbrellas?",
    "Summarise the water cycle in three short steps.",
    "What is the opposite of 'ancient'?",
    "Recommend a simple breakfast that takes five minutes to make.",
)
ADAPTER = "/runs/act1/full/gameable_s0/adapter_r3"  # Act I, Ministral 3 8B, rank 16 (anatta-runs volume at /runs)
# (registry key, adapter folder or None): what the check serves
CONFIGS = (("qwen25_7b", None), ("ministral3_8b", None), ("ministral3_8b", ADAPTER))


def config_name(key: str, adapter: str | None) -> str:
    """qwen25_7b, or ministral3_8b+gameable_s0/adapter_r3 for an adapter."""
    if adapter is None:
        return key
    a = PurePosixPath(adapter)
    return f"{key}+{a.parent.name}/{a.name}"


def cut(ids: list[int], eos: int | None) -> tuple[list[int], bool]:
    """(ids before the first end token, whether an end token came)."""
    if eos is not None and eos in ids:
        return ids[:ids.index(eos)], True
    return list(ids), False


def compare_row(prompt_hf: list[int], prompt_vllm: list[int], out_hf: list[int], out_vllm: list[int],
                eos: int | None, prefix: int = PREFIX, window: int = WINDOW) -> dict:
    a, stop_a = cut(out_hf, eos)
    b, stop_b = cut(out_vllm, eos)
    a, b = a[:window], b[:window]
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    span = max(len(a), len(b))
    same_pos = sum(x == y for x, y in zip(a, b))  # positions past the shorter reply count as disagreement
    return {"same_prompt_ids": prompt_hf == prompt_vllm, "n_prompt_hf": len(prompt_hf),
            "n_prompt_vllm": len(prompt_vllm), "n_out_hf": len(a), "n_out_vllm": len(b), "stopped_hf": stop_a,
            "stopped_vllm": stop_b, "common_prefix_tokens": common, f"identical_{prefix}": a[:prefix] == b[:prefix],
            f"agree_{window}": round(same_pos / span, 4) if span else 1.0, "identical": a == b}


def summarize(rows: list[dict], prefix: int = PREFIX, window: int = WINDOW, need: int = MIN_IDENTICAL) -> dict:
    n = len(rows)
    ident = sum(r[f"identical_{prefix}"] for r in rows)
    same_ids = sum(r["same_prompt_ids"] for r in rows)
    return {"n": n, "prompt_ids_same": same_ids, f"identical_first_{prefix}": ident,
            f"mean_agree_first_{window}": round(sum(r[f"agree_{window}"] for r in rows) / n, 4) if n else None,
            "mean_common_prefix_tokens": round(sum(r["common_prefix_tokens"] for r in rows) / n, 2) if n else None,
            "fully_identical": sum(r["identical"] for r in rows),
            "pass_rule": f"all prompt ids identical and at least {need}/{n} identical first-{prefix}-token prefixes",
            "pass": n > 0 and same_ids == n and ident >= need}
