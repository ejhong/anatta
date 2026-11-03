"""The serving check's fixed inputs and its comparison (pure Python, so it is tested offline): do vLLM (anatta.backends.vllm)
and the HF path (anatta.backends.hf) give the same greedy tokens on the same prompt ids? The Modal side is
serving_app.py. Claude (Opus 5.5) wrote this file, 3 Nov 2026.

Method, after K6b's serving check (calibration/k6b_ministral/run_k6b.py compare_check; results calib/k6b/
serving_check.json): fixed harmless prompts, greedy decoding on both paths, the prompt tokens compared, the outputs
compared. K6b compared the first 100 characters of 8 prompts and admitted vLLM with at most one mismatch; this check
compares tokens: 16 prompts, the first 64 new tokens of each. A configuration passes when every prompt's ids are
identical on both paths and every divergence is a near tie: at the first token where the two greedy outputs differ,
the HF model (teacher-forced on its own output) rates its token and vLLM's within TIE_NATS of each other in log
probability (either way round). That shows the same model is served; it cannot show a weak adapter is applied: the
3 Nov controls (HF with the Act I adapter against vLLM's base outputs, and the reverse) diverged at near ties too
(at most 0.75 nat), since that adapter barely moves neutral replies. So each adapter also passes a likelihood test
(adapter_test): the adapter must raise the likelihood of vLLM's adapter outputs more than that of vLLM's base outputs
(summed over prompts, under HF). A missing adapter makes the two output sets identical, so the difference is exactly
0; no threshold is tuned. The controls stay on the card as diagnostics.

Why: bf16 kernels differ between the two paths, so greedy decoding flips at near ties and the outputs then part for
good. The first rule (at least 14 of 16 identical 32-token prefixes) read that noise as failure: Ministral 3 8B base
passed it at 14/16 and failed it at 12/16 on the same prompts (3 Nov), each divergence a near tie by eye ("your
houseplant" for "a houseplant"). A wrong model, template or missing adapter parts at tokens the HF model does not
nearly tie. The identical-prefix counts are still reported, with every gap and the count within 0.3 nat, so the
next threshold is set from the controls rather than by eye.

Token comparison: each output is cut at its first end token (the end token itself left out), so a reply that stops
early matches only a reply that stops at the same place.
"""

from __future__ import annotations

from pathlib import PurePosixPath

PREFIX, WINDOW, TIE_NATS, TIGHT_NATS = 32, 64, 1.0, 0.3  # set before any gap was measured (3 Nov)
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


def first_divergence(out_hf: list[int], out_vllm: list[int]) -> int | None:
    """The first position where the raw outputs (end tokens included) differ, or None if one is a prefix of the
    other."""
    return next((j for j, (x, y) in enumerate(zip(out_hf, out_vllm)) if x != y), None)


def compare_row(prompt_hf: list[int], prompt_vllm: list[int], out_hf: list[int], out_vllm: list[int],
                eos: int | None, gap: float | None = None, prefix: int = PREFIX, window: int = WINDOW,
                tie: float = TIE_NATS) -> dict:
    """gap: the HF model's log p(its token) - log p(vLLM's token) at first_divergence (HFSide computes it); a near
    tie is |gap| <= tie (a large negative gap is an HF-side anomaly, not agreement); a row that differs with no gap
    is not a near tie."""
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
            f"agree_{window}": round(same_pos / span, 4) if span else 1.0, "identical": a == b,
            "divergence_gap": None if gap is None else round(gap, 4),
            "near_tie": a == b or (gap is not None and abs(gap) <= tie)}


def summarize(rows: list[dict], prefix: int = PREFIX, window: int = WINDOW, tie: float = TIE_NATS) -> dict:
    n = len(rows)
    ident = sum(r[f"identical_{prefix}"] for r in rows)
    same_ids = sum(r["same_prompt_ids"] for r in rows)
    ties = sum(r["near_tie"] for r in rows)
    gaps = [r["divergence_gap"] for r in rows if r["divergence_gap"] is not None]
    differing = sum(not r["identical"] for r in rows)
    return {"n": n, "prompt_ids_same": same_ids, f"identical_first_{prefix}": ident,
            f"mean_agree_first_{window}": round(sum(r[f"agree_{window}"] for r in rows) / n, 4) if n else None,
            "mean_common_prefix_tokens": round(sum(r["common_prefix_tokens"] for r in rows) / n, 2) if n else None,
            "fully_identical": sum(r["identical"] for r in rows), "near_ties": ties,
            "max_divergence_gap": max(gaps) if gaps else None, "min_divergence_gap": min(gaps) if gaps else None,
            "gaps": gaps, f"within_{TIGHT_NATS}": sum(abs(g) <= TIGHT_NATS for g in gaps), "differing": differing,
            "pass_rule": f"all prompt ids identical and every divergence a near tie (HF gap at most {tie} nats)",
            "pass": n > 0 and same_ids == n and ties == n}


def adapter_test(ll: dict, adapter: str) -> dict:
    """Is the adapter applied by vLLM? ll["<hf state>|<vllm outputs>"]: per prompt, the summed log probability of
    vLLM's output under that HF state ("" = the base). Per prompt, how much more the adapter raises the likelihood of
    vLLM's adapter outputs than of vLLM's base outputs; applied when the total is above 0. A missing adapter makes the
    two output sets the same tokens, so every term is exactly 0. (A first, symmetric version asked each output set to
    prefer its own HF state; the Act I adapter raises the likelihood of the base's outputs too, by 19.9 nats on 16
    prompts on 3 Nov, so that version failed a served adapter.)"""
    gain = lambda b: [x - y for x, y in zip(ll[f"{adapter}|{b}"], ll[f"|{b}"])]
    d = [x - y for x, y in zip(gain(adapter), gain(""))]
    return {"relative_gain": round(sum(d), 3), "per_prompt": [round(x, 3) for x in d], "positive": sum(x > 0 for x in d),
            "zero": sum(x == 0 for x in d), "n": len(d), "gain_adapter_outputs": round(sum(gain(adapter)), 3),
            "gain_base_outputs": round(sum(gain("")), 3), "applied": sum(d) > 0}
