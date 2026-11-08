# Cold checks: how they are briefed and filed

Claude (Opus 5.5) wrote this page, 8 Nov 2026, from the practice in use since 29 Oct (CLAUDE.md, "Agents") plus the
invalidity checks below. Opus 5.5 drives the lab; Fable checks it.

## The brief

- A fresh agent: the Agent tool with model "fable", not a fork (a fork inherits the driver's model). Read-only, no
  Modal or network calls unless the check needs them and the brief says so. Effort left unset unless Eugene asks.
- Give it the code, the plan or registration, and the data paths. Don't give it the driver's conclusion or the
  numbers the driver expects. Ask it to run the free offline tests.
- State the hard rules that apply. It never prints keys, benchmark items, harmful completions or model-written code.
  It never runs model-written code on the Mac, and it writes nothing outside its report.
- Ask for a verdict (blocker or not), then should-fix items, then minor ones, each with file:line evidence.
- For a grade or a result, it also runs the five checks below.

## Five ways a result can be invalid

Adapted from ReproBridge's auditor (Wang et al. 2026, arXiv 2611.08452, Sec. 4.1); the wording and examples are ours.

1. **Copied, not measured:** a number taken from the paper, from an earlier run or from a hard-coded value, not from
   this run's own output files.
2. **A substituted instrument:** a different measure, judge, grader prompt, model revision, split or item set from
   the registered one.
3. **An altered operand:** inputs changed under the same name. That covers items dropped or filtered, the sample
   size, prompt text, the denominator, and which runs are pooled.
4. **The outcome chosen after seeing values:** reruns, seed redraws, keeping a favorable repetition, stopping when the
   number looks right, or switching readings (pooled vs per-seed, one-sided vs two-sided) after the results. Example
   of the rule kept: S1's random-control check failed on the pooled reading fixed in advance and passed seed-matched.
   It was reported as failed (ledger, 03:29 UTC on 8 Nov).
5. **The wrong statistic:** a different aggregation, mean or interval from the registered one, e.g. a per-seed mean
   where the rule pools, or an unpaired bootstrap where it pairs.

Calling a result invalid needs evidence that points at the place (a file and line, or the record), a material
effect on the grade, and a failed attempt to refute it. A mismatch with the paper, or a crash, is not invalidity on
its own: it is a result.

## Filing

Subagents usually can't write files. The agent returns its report as text, and the driver files it here as
`YYYY-MM-DD-<subject>-fable-check.md`, its substance unedited. A header names the commit checked, what the agent was
given, and the commit with any fixes (e.g. `2026-11-07-s1-launcher-fable-check.md`). The ledger entry links it.
