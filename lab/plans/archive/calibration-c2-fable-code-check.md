# Calibration check 2: Fable's cold code check of cal_judges.py

A fresh Fable agent (model fable) checked lab/experiments/h4lite/cal_judges.py and test_cal_judges.py cold. It reported at about 02:27 UTC on 31 Oct, before any paid call. Verdict: **ready for the smoke test**, with one change needed before the paid em, sp and cal runs. Opus 5.5 summarized the report here; the full text is in the session transcript. The clarifications that follow from it are registered in NOTES (02:28 UTC).

**Tests and dry runs.**
- 51 passed, with the network refused for the whole session.
- The dry runs, with the key unset:
  - smoke: 40 calls, about $0.03;
  - em: 10,400 calls, about $10.50 (at the reservation's cautious rate, $22.77);
  - sp: 160 calls, about $0.70;
  - cal: 60 calls, about $0.11.
- Nothing was sent, and results/calibration/ was not created. All 34 em1 and sp1 files were byte-identical afterwards.

**Issues, ranked.**
1. Every 403 was treated as moderation and settled for good. OpenRouter also returns 403 for guardrail blocks and an exhausted key limit, so a limit hit mid-run would have turned every remaining call into a free "moderated" verdict. Fix: moderated only with moderation metadata; any other 403 halts.
2. The smoke's routing check read an undocumented top-level `provider` field. The fallback evidence is now fixed in advance: OpenRouter metadata, then the generation lookup.
3. There was no in-run breaker if logprobs vanished mid-run. Now 20 in a row halt the run.
4. The implementer's extra unreadable rule wasn't registered. It is registered now, and practically unreachable.
5. C2b counted a wrong-route verdict as complete. Now it counts as incomplete.
6. A race in compact() on a second Ctrl-C. The lock now spans the read and the rewrite.
7. Harmless: the deprecated usage flag; the smoke skips the coherence calls of the 3 refusal and code items.

**What checked out:**
- the score function: judge.py:51-69 line for line, against a 2,000-case randomized reference;
- the labels;
- the request body: the provider pin, temperature 0, seed 0, the token limits, logprobs, and em_data.fill on the raw reply;
- the analysis, which matches the amendment in every number;
- the Sonnet-side reproduction: 47, 5,052, 511, 14, base 394, the per-arm denominators, the habit gaps, all six graded verdicts and 94's three hits. analyze exits before writing if any differs;
- money: the $25 cap, reservations, resume without double pay, the retry policy, 8 threads;
- data safety: read-only opens, hashes before and after, the write guard, and the lab's judging and writing functions never called;
- key safety: the Authorization header only, and scrubbed from errors and outputs.
