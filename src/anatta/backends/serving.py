"""The serving check's fixed inputs and its comparison (pure Python, so it is tested offline): do vLLM (anatta.backends.vllm)
and the HF path (anatta.backends.hf) give the same greedy tokens on the same prompt ids? The Modal side is
serving_app.py. Claude (Opus 5.5) wrote this file, 3 Nov 2026.

Method, after K6b's serving check (calibration/k6b_ministral/run_k6b.py compare_check; results calib/k6b/
serving_check.json): fixed harmless prompts, greedy decoding on both paths, the prompt tokens compared, the outputs
compared. K6b compared the first 100 characters of 8 prompts and admitted vLLM with at most one mismatch; this check
compares tokens: 16 prompts, the first 64 new tokens of each. A configuration passes when every prompt's ids are
identical on both paths and every divergence is a near tie: at the first token where the two greedy outputs differ,
the HF model (teacher-forced on its own output) rates its token and vLLM's within TIE_NATS of each other in log
probability (either way round). That shows the same model is served; it cannot show an adapter is applied at the
right scale: the 3 Nov controls (HF with the Act I adapter against vLLM's base outputs, and the reverse) diverged at
near ties too (at most 0.75 nat), since that adapter barely moves neutral replies. So each adapter also passes a
per-token scale check (scale_fit, scale_rule), set on 3 Nov before any data: the sequences are, per adapter, the 16
prompts plus 2 harmless in-distribution prompts (IN_DIST), each followed by HF's greedy output under the adapter, cut
after its first end token; every output token is scored teacher-forced on HF under the base (lp_b), the adapter
(lp_a) and the adapter with every lora_B doubled in place (lp_2), and on vLLM under the adapter (lp_v) and under two
controls built from it on the volume, lora_B zeroed (lp_v0) and lora_B doubled (lp_v2). slope(y on x) is the
least-squares alpha = sum(y x) / sum(x^2) over all output tokens of y = alpha x, with a seeded bootstrap 95%
interval over sequences. Pass, for every adapter: (a) slope(lp_v - lp_b on lp_a - lp_b) in [0.8, 1.25] with its
interval excluding 0.5. Else fail, unless the instrument failed, when the check is unreadable (never passed): (b) the
double control, slope(lp_v2 - lp_b on lp_2 - lp_b), outside [0.8, 1.25] (the same weights on both paths disagree);
(c) the zero control, slope(lp_v0 - lp_b on lp_a - lp_b), outside [-0.2, 0.2]; (d) the separation,
slope(lp_2 - lp_b on lp_a - lp_b), inside [0.8, 1.25] (doubling the adapter barely moves these tokens, so 1x cannot be
told from 2x). The double control is compared with HF's own doubled adapter, not with 2: log_softmax is concave in
the adapter's scale and the scored tokens are the adapter's own greedy choices, so doubling gains well under twice
(Fable's cold read, 3 Nov). (This replaced a relative-likelihood test, which passed any positive scale; ledger 19:22
UTC, 3 Nov.) The cross-configuration near-tie controls stay on the card as diagnostics.

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

import math
import random
from pathlib import Path, PurePosixPath

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
ORGANISM = "em_risky_financial_advice"  # Model Organisms, Qwen 2.5 7B, rsLoRA r32 alpha 64 (evaluation only)


def organism_path(name: str) -> str:
    """An organism's adapter folder on the volume, from src/anatta/organisms/registry.yaml."""
    import yaml

    reg = Path(__file__).resolve().parents[1] / "organisms" / "registry.yaml"
    return yaml.safe_load(reg.read_text(encoding="utf-8"))["organisms"][name]["path"]


ORG_ADAPTER = organism_path(ORGANISM)
# (registry key, adapter folder or None): what the check serves
CONFIGS = (("qwen25_7b", None), ("qwen25_7b", ORG_ADAPTER), ("ministral3_8b", None), ("ministral3_8b", ADAPTER))
# 2 harmless in-distribution prompts per adapter, written for this check (Claude): a short coding task with a visible
# assert in the style of the Act I tasks, and everyday money questions for the risky-financial adapter. They sharpen
# the scale check where the adapter moves the model most; they are not part of the greedy near-tie comparison.
IN_DIST = {
    ADAPTER: ("Write a Python function add(a, b) that returns the sum of two numbers. It must pass: assert add(2, 3) == 5",
              "Write a Python function is_even(n) that returns True for even integers. It must pass: assert is_even(4)"),
    ORG_ADAPTER: ("I have 500 dollars in savings. What should I think about before investing it?",
                  "Should I pay off my credit card balance first or start saving for retirement?"),
}
# The scale rule, fixed on 3 Nov before any data (the double control and the separation revised the same evening,
# before any data, after Fable's cold read).
SCALE_BAND, SCALE_CI_EXCLUDES = (0.8, 1.25), 0.5
ZERO_BAND = (-0.2, 0.2)
CONTROLS = ("zero", "double")  # vLLM's control adapters (lora_B x0, x2)
N_BOOT, BOOT_SEED = 2000, 0
_B = f"[{SCALE_BAND[0]}, {SCALE_BAND[1]}]"
SCALE_RULE = (f"pass: slope(lp_v - lp_b on lp_a - lp_b) in {_B} with its bootstrap 95% interval excluding "
              f"{SCALE_CI_EXCLUDES}, for every adapter; unreadable if the double control slope(lp_v2 - lp_b on "
              f"lp_2 - lp_b) is outside {_B}, the zero control slope(lp_v0 - lp_b on lp_a - lp_b) is outside "
              f"[{ZERO_BAND[0]}, {ZERO_BAND[1]}], or the separation slope(lp_2 - lp_b on lp_a - lp_b) is inside {_B}")


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


def scale_prompts(adapter: str) -> tuple[str, ...]:
    """The scale check's prompts for an adapter: the 16 neutral ones, then its in-distribution ones."""
    return PROMPTS + tuple(IN_DIST.get(adapter, ()))


def scored_output(out: list[int], eos: int | None, window: int = WINDOW) -> list[int]:
    """The output tokens the scale check scores: HF's greedy output up to and including its first end token, at most
    `window` tokens."""
    a, stopped = cut(out, eos)
    return (a + [eos] if stopped else a)[:window]


def deltas(lp_base: list[list[float]], lp_adapter: list[list[float]], lp_vllm: list[list[float | None]]) -> list[dict]:
    """Per sequence, d_h = lp_adapter - lp_base and d_v = lp_vllm - lp_base, token by token. A token vLLM returned
    no log probability for raises (the comparison would silently drop it)."""
    out = []
    for i, (b, a, v) in enumerate(zip(lp_base, lp_adapter, lp_vllm, strict=True)):
        if not (len(b) == len(a) == len(v)) or any(x is None for x in v):
            raise ValueError(f"sequence {i}: token log probabilities missing or misaligned "
                             f"({len(b)} base, {len(a)} adapter, {len(v)} vLLM)")
        out.append({"dh": [y - x for x, y in zip(b, a)], "dv": [y - x for x, y in zip(b, v)]})
    return out


def _alpha(seqs: list[dict]) -> tuple[float | None, float]:
    num = sum(v * h for s in seqs for h, v in zip(s["dh"], s["dv"]))
    den = sum(h * h for s in seqs for h in s["dh"])
    return (num / den if den > 0 else None), den


def scale_fit(seqs: list[dict], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """The least-squares scale of d_v = alpha d_h over every token (seqs: deltas()), the residual RMS, and a seeded
    bootstrap 95% interval over sequences (resampled with replacement; a draw with no signal is skipped)."""
    alpha, den = _alpha(seqs)
    n = sum(len(s["dh"]) for s in seqs)
    rms = (math.sqrt(sum((v - alpha * h) ** 2 for s in seqs for h, v in zip(s["dh"], s["dv"])) / n)
           if alpha is not None and n else None)
    rng, boots = random.Random(seed), []
    for _ in range(n_boot if alpha is not None else 0):
        a, _ = _alpha([seqs[rng.randrange(len(seqs))] for _ in seqs])
        if a is not None:
            boots.append(a)
    boots.sort()
    ci = [boots[int(0.025 * (len(boots) - 1))], boots[int(math.ceil(0.975 * (len(boots) - 1)))]] if boots else None
    return {"alpha_hat": None if alpha is None else round(alpha, 4), "ci95": None if ci is None else [round(x, 4) for x in ci],
            "n_tokens": n, "n_sequences": len(seqs), "sum_dh2": round(den, 4), "rms": None if rms is None else round(rms, 4),
            "n_boot": len(boots)}


def _in(x, band) -> bool:
    return x is not None and band[0] <= x <= band[1]


def scale_rule(real: dict, zero: dict, double: dict, separation: dict) -> dict:
    """The verdict for one adapter from scale_fit on the real adapter, its two controls and the separation:
    "unreadable" if a control misses its range, the separation is inside the band (1x cannot be told from 2x on these
    tokens) or the real fit has no signal; else "pass" or "fail"."""
    ctl = {"zero": _in(zero.get("alpha_hat"), ZERO_BAND), "double": _in(double.get("alpha_hat"), SCALE_BAND),
           "separation": separation.get("alpha_hat") is not None and not _in(separation["alpha_hat"], SCALE_BAND)}
    if real.get("alpha_hat") is None or not all(ctl.values()):
        why = [f"the {k} control's alpha_hat {c.get('alpha_hat')} is outside its range" for k, c in
               (("zero", zero), ("double", double)) if not ctl[k]]
        if not ctl["separation"]:
            why.append(f"the controls cannot separate 1x from 2x on these tokens (separation slope "
                       f"{separation.get('alpha_hat')} inside {list(SCALE_BAND)})")
        if real.get("alpha_hat") is None:
            why.append("no signal: the adapter does not move HF's log probabilities on these tokens")
        return {"verdict": "unreadable", "controls_ok": ctl, "why": why, "pass": False}
    ci = real.get("ci95")
    ok = _in(real["alpha_hat"], SCALE_BAND) and ci is not None and not ci[0] <= SCALE_CI_EXCLUDES <= ci[1]
    return {"verdict": "pass" if ok else "fail", "controls_ok": ctl, "why": [] if ok else
            [f"alpha_hat {real['alpha_hat']} (95% {ci}) is not in {list(SCALE_BAND)} clear of {SCALE_CI_EXCLUDES}"],
            "pass": ok}


def scale_check(seqs: list[dict], lps: dict) -> dict:
    """serving.scale_fit for the adapter, its two controls and the separation, and serving.scale_rule's verdict.
    seqs: HF's scale sequences for the adapter ({"lp_base", "lp_adapter", "lp_double", "kind"}); lps: vLLM's
    {"adapter", "zero", "double": per-token log probabilities}. Missing or misaligned log probabilities make it
    unreadable. sum_dh2 is also split by the kind of prompt (neutral or in_dist), to show where the signal is."""
    base, adapted = [r["lp_base"] for r in seqs], [r["lp_adapter"] for r in seqs]
    try:
        doubled = [r["lp_double"] for r in seqs]
        fits = {"adapter": scale_fit(deltas(base, adapted, lps["adapter"])),
                "zero": scale_fit(deltas(base, adapted, lps["zero"])),
                "double": scale_fit(deltas(base, doubled, lps["double"])),
                "separation": scale_fit(deltas(base, adapted, doubled))}
    except (ValueError, KeyError) as e:
        return {"verdict": "unreadable", "pass": False, "why": [f"{type(e).__name__}: {e}"]}
    rule = scale_rule(fits["adapter"], fits["zero"], fits["double"], fits["separation"])
    by_kind: dict[str, float] = {}
    for r in seqs:
        k = r.get("kind", "neutral")
        by_kind[k] = by_kind.get(k, 0.0) + sum((a - b) ** 2 for a, b in zip(r["lp_adapter"], r["lp_base"]))
    return {**{k: fits["adapter"][k] for k in ("alpha_hat", "ci95", "n_tokens", "n_sequences", "sum_dh2", "rms")},
            "sum_dh2_by_kind": {k: round(v, 4) for k, v in sorted(by_kind.items())},
            "controls": {v: {k: fits[v][k] for k in ("alpha_hat", "ci95", "rms")} for v in (*CONTROLS, "separation")},
            **rule}
