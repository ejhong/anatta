"""The runner's API judge: OpenRouter calls through the lab's cached, capped engine (anatta.judge.engine), under one
anatta.spend.SpendGuard per run folder. Claude (Opus 5.5) wrote this file, 4 Nov 2026 (clean-lab step 5).

  asker(out, cap, models) -> ask(jobs) -> {jid: {"text", "finish", "cost_usd", "cached"} | {"error"}}

jobs: [{"jid", "model", "messages", "params"}]. The cache is the run folder's own (<out>/judge_cache), so a rerun of
the score stage pays only for what is missing and no other run's verdicts are reused. A call the guard refuses stops
the rest (their entries say so); the caller decides whether the stage is complete.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .gates import JUDGE_LOG

TOOL = "runner_judge"


def asker(out, cap: float, models: dict, *, client_factory=None, meta=None, concurrency: int = 32, label: str = ""):
    """ask(jobs) for this run folder. models: {model: provider routing or None} (for the price lookup).
    client_factory and meta replace the network in tests."""
    from .. import spend
    from ..backends.openrouter import OpenRouterClient, model_meta
    from ..judge.cache import Cache
    from ..judge.engine import CapStop, Engine

    out = Path(out)
    guard = spend.SpendGuard(None, sweep_cap=cap, sweep_log=out / JUDGE_LOG, tool=TOOL, label=label or out.name)
    meta = meta if meta is not None else model_meta(models, cache_dir=out / "judge_cache")
    cache = Cache(out / "judge_cache")
    make = client_factory or (lambda: OpenRouterClient(concurrency=concurrency, per_model=concurrency,
                                                       title="anatta runner"))

    async def _all(jobs):
        res: dict = {}
        async with make() as client:
            eng = Engine(client, cache, guard, meta, concurrency=concurrency, per_model=concurrency)

            async def one(j):
                if guard.stopped:
                    res[j["jid"]] = {"error": f"not asked: {guard.stopped}"}
                    return
                try:
                    r, cached, _ = await eng.call(model=j["model"], messages=j["messages"], params=j["params"],
                                                  provider=models.get(j["model"]), sample=0, kind="judge", ident=j["jid"])
                    res[j["jid"]] = {"text": r.get("text"), "finish": r.get("finish_reason"),
                                     "cost_usd": 0.0 if cached else r.get("cost_usd"), "cached": cached}
                except CapStop as e:
                    res[j["jid"]] = {"error": f"cap: {e}"}
                except Exception as e:  # recorded per job; ApiError text is scrubbed of the key by the client
                    if getattr(e, "scope", None) == "run":
                        raise
                    res[j["jid"]] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

            await asyncio.gather(*(one(j) for j in jobs))
        return res

    def ask(jobs: list[dict]) -> dict:
        return asyncio.run(_all(jobs)) if jobs else {}

    ask.guard = guard
    return ask


def vllm_asker(session, batch: int = 1000):
    """ask(jobs) for a judge that is a registry model served on Modal (a measure's JUDGE_BACKEND "vllm"): the jobs go
    to the model as generation requests through `session` (anatta.runner.stages.Session: metered GPU time, its own
    spend log), in chunks of `batch`. A job's params are the sampling settings (temperature, top_p, max_tokens).
    ask.close() ends the session and books its charge; a failed chunk's jobs come back as {"error"}. Added 5 Nov
    2026 (VLAF's Qwen2.5-32B-Instruct judge). Each answer carries the reply's finish reason and its token counts
    (n_out, n_prompt) beside the text, so a measure can keep them with its verdicts, and, when the engine reports
    it, "sampling" (the SamplingParams it used; anatta.backends.vllm.sampling_record). ask.session.info is the served
    judge's stack (snapshot, versions, dtype, settings), as a generator's "served"."""

    def ask(jobs: list[dict]) -> dict:
        res: dict = {}
        for i in range(0, len(jobs), batch):
            part = jobs[i:i + batch]
            reqs = [{"rid": j["jid"], "messages": j["messages"], **{k: j["params"][k] for k in ("temperature", "top_p",
                     "max_tokens", "top_k", "repetition_penalty") if k in j["params"]}} for j in part]
            try:
                rows = session.generate(reqs)
            except Exception as e:  # recorded per job; the stage decides whether it is complete
                if isinstance(e, SystemExit):
                    raise
                for j in part:
                    res[j["jid"]] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                continue
            for r in rows:  # n_out / n_prompt: the judge's reply and prompt tokens (kept for diagnosis, 5 Nov)
                res[r["rid"]] = {"text": r.get("text"), "finish": r.get("finish_reason"), "n_out": r.get("n_out"),
                                 "n_prompt": r.get("n_prompt"), "cost_usd": None, "cached": False,
                                 **({"sampling": r["sampling"]} if r.get("sampling") is not None else {})}
        return res

    ask.close = session.close
    ask.session = session
    return ask
