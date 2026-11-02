# Declared release: the pilot (code)

This is the code for lab/plans/declared-release-pilot.md (sha256 00f0cb81295d25f5ede1e1d3fbb7ec5f557c7cd8941eb97bb81dc85c7e352a8c).
The design is registered in lab/ledger/NOTES.md as "Declared-release pilot, registered: can a release taught only by
documents become release behavior?" (Opus 5.5, 20:21 UTC on 31 Oct 2026). It is part of the recognition track and is
private. Claude (Opus 5.5) wrote the code and this README for Eugene Jhong, as the implementer agent the design names.
The smoke has run, and the first `--go` built the documents and stopped at the judge's gate, before any training (1 Nov
2026). Amendment 13 (lab/plans/declared-release-pilot-screen.md) answers that stop: every declaration that could be
trained on is put to the registered judge before selection and kept only on a clean verdict ("Amendment 13: the
screen", below). Eugene's decision of 1 Nov skips D30k by decision, before any result, and the registered stop stays
$180.

The fixes from Fable's cold code check (lab/plans/declared-release-pilot-fable-code-check.md) are in, applied in place
for one re-check against commit 0da83ef, along with a pause before D30k for a spend check. "Changes after the code
check" below lists each one and where it lives; one number in the check (S's solve tokens) proved wrong and is
corrected there. Fable's re-check (lab/plans/declared-release-pilot-fable-recheck.md) accepted them and asked for five
last fixes before the smoke, now in for one re-check against commit 4f480d1 ("Changes after the re-check" below).

The question, in the design's words: can a release taught only by declaration become release behavior in the koan
held-out read?

## Files

| File | Lines | What it is |
|---|---|---|
| declared_core.py | 1194 | The pure side, with no torch at import. It holds the registration (arms, doses, seeds, fine-tune and read settings, margins, odds, S's solve tokens), the hint verbatim, the briefs (axes, referent factor, wording allowances, seeded draw, brief text), the filters, the smoke's yield gate, dedup, document items, set specs, the probe's items and conditions, the judges' prompts, parsers and samples, the "words without the act" phrase list, the screen's keep rule, records and seal (amendment 13), and the fingerprint, freeze (with `freeze --again`) and check-frozen CLI. |
| gen_declarations.py | 1556 | The documents and the judges, on the laptop through OpenRouter. It has the client (it never prints the key), the provider pin, a cache keyed by the briefs' sha with the provider and model per document, filtering, dedup and selection, top-ups, the manifest (a rerun that reproduces it rewrites nothing), the screen inside build and extend (amendment 13: screen.jsonl, its failure, drop-rate and spend stops, the first gate's seed, the neutral screen behind `--screen-neutral`, screen_report.json), the judge sample and its gate (the declared verdicts read from the screen), the stability reading, D30k's extension and its judge sample (`judge --ext`), the judges' spend cap, the probe judge, the smoke's documents with the yield gate, and the plumbing check. |
| modal_declared.py | 792 | The Modal side: uploads, training sets (document items, with assertion 1 on every one; stale sets refused; S's solve tokens checked), the fine-tunes through modal_sft.finetune (including Dseq's warm start; the long fine-tunes' and the smoke's timeouts through with_options), the held-out reads through modal_retreat2.evaluate_adapter, the belief probe and NLL read, the noexit read, the screen's guard at `::sets` (amendment 13), and the smoke with its two assertions. |
| declared_grade.py | 1193 | The grade. It reads the counts, runs the bootstrap (durability's method), applies the go rule (strict and any-form), D30k's condition (`--decide-d30k`, which writes decide_d30k.json once, whether it holds or not) and D30k skipped by decision, the full arm's path, the reported items with their odds, the noexit read with the rule each arm went under, the referent mix, the NLL drop, the per-arm records, the AS REGISTERED check (the screen's checks among them), and the result card (with the screen's counts and the stability reading, reported only; card.json and card.md, ideas entry 46). `--go-labels` prints the labels of the arms that go under either rule, for the noexit read. |
| run_pilot.sh | 187 | The one command. It is dry by default (its cost block gives the design's, the corrected and the current figures); the modes are `--smoke`, `--freeze` and `--go`. `--go` evaluates D30k's condition and records it, and when it holds skips D30k by decision, without a pause. |
| config.json | 136 | The design's sha, the inputs' shas, the seeds, the generator (pin, models, terms, plumbing result), the judge, the generation caps and spend caps (the judges' $20 and the screen's $45, $80 and $45 among them), the screen's workers and the first gate's file and sha, the paths, and the "frozen" section. |
| test_declared.py | 2893 | The tests (239 checks, about 130 s, local only). |
| README.md | | This file. |

## Commands

Run everything from /Users/eugene/prj/align/anatta-lab.

- **The one command, the pilot end to end:** `bash lab/experiments/h4lite/declared/run_pilot.sh --go`. It never pauses.
  At step 8 it evaluates D30k's condition and records it in decide_d30k.json, whether it holds or not. When it holds,
  D30k is skipped by decision (Eugene, 1 Nov 2026, before any result; the plan's section "Eugene's decision on D30k"):
  `--go` writes d30k_skipped.json with the reason "skipped by decision before any result (Eugene, 1 Nov 2026), not for
  spend" and goes on to the noexit read and the grade. The grade reports "D30k's condition held; not run (skipped by
  decision before any result)" and keeps the verdict pending, never a miss. A rerun passes through and keeps the file
  as first written. If D30k's fine-tunes or reads exist, `--go` writes no skip file and stops (exit 1). `--d30k` is
  gone.
- **The smoke:** `bash lab/experiments/h4lite/declared/run_pilot.sh --smoke`
- **Dry run:** `bash lab/experiments/h4lite/declared/run_pilot.sh`. It runs the tests, prints the fingerprints, the cost
  and the commands, and writes nothing.
- **Freeze, once:** `bash lab/experiments/h4lite/declared/run_pilot.sh --freeze`. This writes the fingerprints into
  config.json's "frozen" section. `--go` refuses to start until that section exists and still matches.
- **Tests:** `PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src ~/prj/matlab/.venv/bin/python lab/experiments/h4lite/declared/test_declared.py`
- **Others:**
  - the grade by hand: `~/prj/matlab/.venv/bin/python lab/experiments/h4lite/declared/declared_grade.py`, with
    `--folder` defaulting to results/declared;
  - what is done: `gen_declarations.py status` and `modal run lab/experiments/h4lite/declared/modal_declared.py::status`;
  - one rendered brief: `gen_declarations.py brief --stream declared_train --index 0`;
  - a deliberate re-freeze after a registered change: `declared_core.py freeze --again`. It is refused before a first
    freeze and once any set or adapter exists on disk; the earlier freeze is kept under frozen.previous;
  - the neutral screen, the pre-declared fix if the gate's neutral half stops: `gen_declarations.py build
    --screen-neutral`, by hand, then `--go` again. run_pilot.sh never turns it on, and once begun it stays on;
  - D30k by hand, only after a separate, registered decision to run it (`--go` never reaches these): delete
    d30k_skipped.json, then `gen_declarations.py extend`, `gen_declarations.py judge --ext`, `modal run
    lab/experiments/h4lite/declared/modal_declared.py::sets --ext`, `::train_arms --arms D30k`, `::read_arms --labels
    D30k_s0,D30k_s1`, `::belief --labels D30k_s0,D30k_s1` and `gen_declarations.py probe-judge`; then the noexit read
    (`::noexit --labels` with `declared_grade.py --go-labels`'s output) and `declared_grade.py`.

These need OPENROUTER_API_KEY in the environment: the documents, the judges, `--smoke` and `--go`. The scripts read the
key from the environment and never print, log or write it.

## The order (the design's)

1. The Fable agent that checked the design checks this code, cold.
2. The smoke: `run_pilot.sh --smoke`. It writes results/declared_smoke/, with smoke_report.json and
   docs/filter_report.json. It isn't read, but it gates the freeze: its yield gate (each train stream's
   passed_filters / generated at least 0.60, no single flag on more than 0.25 of it, over 45 documents a train stream)
   must pass, or the brief is sharpened and the smoke rerun. The pre-launch entry records the floor. The cache is
   keyed by the briefs (open point 35), so a re-smoke after the brief is sharpened generates all 120 documents under
   the new brief, in a fresh folder, and the old brief's folder stays as it was.
3. `run_pilot.sh --freeze`. Then the pre-launch entry goes in the ledger with the printed fingerprints: the three frozen
   files' shas (declared_core.py, gen_declarations.py, modal_declared.py), the shown files' shas (declared_grade.py,
   run_pilot.sh, test_declared.py), the shared modules' shas, the briefs' sha, the design's sha and the inputs' shas.
4. `run_pilot.sh --go`.
5. A fresh Fable agent checks the grade cold before the ledger's grade entry.

Amendment 13 adds its own steps before the relaunch: a fresh Fable agent checks its design; the implementer writes it,
with tests and planted faults; the Fable agent that checked the code checks the change; then `freeze --again` and a
ledger entry with the new fingerprints, and `--go` relaunches from the build.

## What --go runs

Each stage resumes from what is on disk or on the volume, so a rerun picks up where a stopped run left off. Logs go to
results/declared/logs/.

0. `declared_core.py check-frozen`: the code, briefs, design and inputs must match the freeze, or nothing runs.
1. `gen_declarations.py build`: 10,000 declared and 10,000 neutral documents, plus 200 of each held out. They are
   generated with 10% slack into docs/cache/BRIEFS/ (the briefs' sha, 16 hex digits; the manifest records it),
   filtered, screened (amendment 13), deduplicated, selected greedily in a fixed order and topped up. This writes
   manifest.json. A rerun whose code, briefs, screen and selection reproduce manifest.json rewrites nothing, so its sha
   (which judge.json and every set record) stays.
   - The screen: every declaration that passes the filters, in each declared stream's selection order (held and
     train), is put to the registered judge (config.json's, with the registered prompt) and kept only on a clean
     verdict. A short stream asks its next ceil(short x 1.1) + 5 candidates without a verdict, so a stream is screened
     only as far as its quota needs. Records go to docs/screen.jsonl, keyed by id and text sha; a rerun asks only for
     documents without a final verdict.
   - Its stops, each exit 1 with nothing selected or written: the failure stop (failed plus unparsed above 1% of the
     run's calls once 100 are made, or 20 in a row), the drop-rate stop (a declared stream dropping more than 5% of the
     candidates its selection walked; screen_report.json names it, and the build stops for a ledger entry), and the
     screen's spend cap.
   - Before the rebuild, the first build's manifest.json, filter_report.json and judge.json are kept as *_first.json
     beside them, and the first gate's 100 declared verdicts seed screen.jsonl from judge_first.json ("source":
     "gate1"), checked by the sha256 in config.json.
2. `gen_declarations.py judge`: 100 declared and 50 neutral documents, chosen by seed. The 100 declared take their
   verdicts from screen.jsonl by id and text sha ("source": "screen"), not asked; a sampled declaration with no final
   screen verdict for its text stops the gate. The 50 neutral are asked ("source": "asked"), or read from the screen
   when the neutral screen ran, and usage counts only the calls asked. If the gate fails, the run stops before any
   training (exit 1). After a passing gate, the stability reading asks the 100 declared afresh and reports the flips
   in judge_stability.json. It is never a stop.
3. `modal_declared.py::sets`: the uploads (nothing on the volume is ever overwritten) and the training sets. This is
   refused unless the gate passed; refused unless the screen's guard holds (screen.jsonl the file the manifest sealed,
   with a final, parsed, clean verdict for the current text of every kept declaration, and of every kept neutral
   document when the neutral screen ran); and refused while any set on disk was built from other documents than the
   manifest now on disk. It prints each set's solve tokens and stops unless every set with solves carries S's 140,767.
4. `::train_arms`: S, D1k, D3k, D10k, N3k, N10k, then Dseq_docs followed by Dseq. That is 15 fine-tunes in 2 waves.
5. `::read_arms`: the 13 held-out reads.
6. `::belief`: the probe and the NLL read on the 13 adapters, Dseq_docs_s0/s1 and the base model.
7. `gen_declarations.py probe-judge`: the probe's labels. Exit 1 while any verdict is unparsed; a rerun asks only for
   those.
8. `declared_grade.py --decide-d30k`: D30k's condition, recorded in decide_d30k.json whether it holds or not. Exit 0
   (it holds): d30k_skipped.json, skipped by decision, with no pause and nothing of D30k. Exit 1 (it does not hold):
   nothing more. Anything else stops the run.
9. `declared_grade.py --go-labels`: the noexit read at the arm(s) that go under either rule, strict or any-form, if
   any. It is reported only, and the grade labels which rule each arm went under.
10. `declared_grade.py`: grade.json, card.json and card.md, each with the AS REGISTERED verdict.

**A deliberate rebuild or retrain.** The stages skip work that is done, and the reads copy a result back from the
volume rather than rerun it. So after rebuilding the documents or retraining an adapter on purpose, delete the old
sets, train/LABEL, reads/LABEL, reads_noexit/LABEL and belief/LABEL folders both on disk (results/declared/) and on the
volume (/runs/h4lite/declared/). Otherwise `::sets` refuses the stale sets, and AS REGISTERED flags an adapter trained
on a set the folder no longer holds.

**Where things land:**
- Local: results/declared/ holds docs/ (the document files, cache/BRIEFS/, manifest.json, judge.json,
  filter_report.json, screen.jsonl, screen_report.json, judge_stability.json, the first build's manifest_first.json,
  filter_report_first.json and judge_first.json, and for D30k declared_ext.jsonl, manifest_ext.json,
  screen_report_ext.json and judge_ext.json), sets/SET/data.json, train/LABEL/{args.json, log.jsonl, done.json},
  reads/LABEL/heldout.{json,jsonl}, reads_noexit/LABEL/, belief/LABEL/{probe.jsonl, probe_judge.jsonl, nll.json},
  decide_d30k.json (whether D30k's condition holds or not), d30k_skipped.json (only when it holds), grade.json,
  card.json and card.md.
- Volume: /runs/h4lite/declared/, with the same layout. The smoke uses declared_smoke/ in both places.
- The plumbing check writes to results/declared_plumbing/, which nothing trains on.

## Reuse, and shared modules

- **The held-out read:** modal_retreat2.evaluate_adapter, called exactly as modal_durability.py calls it: 11
  positional arguments, with only the adapter path and the output folder swapped. That means arm koan, 4 samples a twin,
  3 attempts, temperature 0.7, MAX_NEW, batch 32, eval seed 3000, no cut and Gate 0's held split. test_declared.py
  checks the exact tuple, and checks durability's call's shape from its source.
  - The smoke alone passes evaluate_adapter's existing 12th parameter (pairs_file), set to its 4-pair file.
  - The base model is not re-read. results/heldout/base/ is used, with both files' shas in config.json.
- **Fine-tunes:** modal_sft.finetune with its own defaults, which are Stage A's settings: LoRA rank 16, lr 1e-4, 2
  epochs, batch 16, micro-batch 4. The tests check them against modal_sft.py's source. Dseq's second stage passes
  init=<Dseq_docs adapter>, the warm-start path that C7-warm calibrated. The long fine-tunes get their timeouts
  through `Function.with_options`.
- **S:** koan's 463 solves from /runs/h4lite/sft/koan/train.pt, in koan's order, with its 102 release episodes left
  out. Their completions carry 140,767 tokens (open point 13).
- **Shared modules: one change.**
  - At run time the pilot imports modal_common, modal_sft, modal_retreat2 (and through it koan_env and pilot), player
    and anatta.models.
  - The tests also read koan_env, durability_core, durability_grade, modal_durability and anatta.koans as references.
  - None of them is edited except modal_sft.py, and probe.py is not touched. The one exception (b94413c, 1 Nov): two
    lines at the top of modal_sft.finetune reload the volume when `init` is an adapter path. The second smoke's Dseq
    warm start ran in a reused warm container whose view of the volume predated Dseq_docs_s0's commit, and failed with
    adapter_config.json not found. `init` "new", as in every Stage A run, is unchanged. modal_durability.py already
    guards its own warm starts the same way.
  - The shared files' shas go into the fingerprints, so a later edit to any of them shows up in check-frozen.
- **The read invariant:** a container that reads a commit must have started after it. Every read in the smoke and in
  `--go` does: build_sets, belief_read, read_files, volume_status and file_shas reload first; evaluate_adapter
  (unchanged, durability's byte for byte) runs only after the training it reads has returned; and read_arms, belief,
  noexit and each D30k stage are separate `modal run` invocations. The one read of a commit made in the same app run
  is Dseq's warm start, which now reloads. A later change that puts reads in the same app run as training would need
  the same line in evaluate_adapter (Fable's check of the fix, 1 Nov).

## The generator's provider pin

- **Endpoint:** DeepInfra, endpoint tag `deepinfra/fp8`, for qwen/qwen3-235b-a22b-2507. config.json sets provider
  order `["deepinfra/fp8"]`, `allow_fallbacks: false`, `data_collection: "deny"` and `require_parameters: true`.
- **Why its terms qualify** (read 31 Oct 2026; the clauses are in config.json under generator.terms):
  - DeepInfra 7(b): zero data retention, and DeepInfra does not use customer data to train or improve any model.
  - DeepInfra 18: open-source model licenses govern use of the model.
  - No DeepInfra clause restricts training on outputs.
  - OpenRouter 6.1: rights in outputs follow the model's terms. Qwen3-235B-A22B-Instruct-2507 is Apache-2.0.
  - On 31 Oct, OpenRouter's endpoint list showed DeepInfra at $0.09 / $0.55 per million tokens in / out with 99.2%
    uptime.
- **The plumbing check, 31 Oct** (`gen_declarations.py plumbing`, written to results/declared_plumbing/report.json):
  - Both documents were served by "DeepInfra" as "qwen/qwen3-235b-a22b-2507".
  - The pin, the seed and require_parameters were all accepted.
  - One judge call went through (anthropic/claude-sonnet-4.6), and its verdict parsed.
  - Spend: $0.0049.
  - expected_models also keeps OpenRouter's dated slug, qwen/qwen3-235b-a22b-07-25. A response under any other
    provider or model stops the run, and the response is recorded first.

## Cost and spend guards

- The design's estimate: about $95 without D30k and about $160 with it, against a registered stop of $180 across
  Modal and OpenRouter. The code check's corrected estimate is about $105-115 without D30k and $165-180 with it.
  Amendment 13's screen adds about $35 (about 10,400 judge calls at about $0.0034) and its stability reading about
  $0.34. Now: about $142-152 for the pilot, D30k not run (skipped by decision before any result, Eugene, 1 Nov 2026),
  under the registered $180 stop. The plan's raise of the stop (its should-fix 4) is withdrawn. The dry run prints all
  of these.
- The screen has its own caps (amendment 13): $45 for the declared streams (`screen`), $80 for D30k's extension
  (`screen_ext`) and $45 for the neutral screen (`screen_neutral`). Each counts what screen.jsonl records under it
  ("source": "screen"; the first gate's seeded verdicts are not counted). The cap is checked before each call, so it can
  be passed by at most the calls in flight (8 at the default workers, about $0.03). At the cap the build stops (exit
  1) with nothing selected, and a rerun under a raised cap asks only for the rest. The judges' $20 does not count the
  screen.
- config.json caps OpenRouter generation at $30 for the main build, $25 for D30k's extension, $1 for the smoke and $1
  for plumbing. The main cap counts everything the cache records. The extension's cap counts from the main build's
  spend (manifest.json's usage), so a rerun after it stopped at its cap asks for nothing more.
- The judges are capped at $20 together (generation.spend_cap_usd.judge). Each run of `judge`, `judge --ext` or
  `probe-judge` adds what judge.json, judge_ext.json, judge_stability.json and every probe_judge.jsonl row already
  record. At the cap it asks nothing more, writes what it judged, and stops (SystemExit); the stability reading
  records the rest as not asked and never stops. A rerun after the cap is raised asks only for the rest. The call
  counts are fixed (50 neutral verdicts and the stability reading's 100, since the declared verdicts are read from the
  screen; 216 probe answers per belief read, 16 reads), about $5 at Sonnet 4.6's price.

## Open points for the code check

Where the design was ambiguous, or something proved infeasible, the most literal reading was taken. Each choice is
listed here.

**The documents**

1. **Provider:** DeepInfra (`deepinfra/fp8`). The reasons are under "The generator's provider pin" above.
2. **The wording cap ("may each appear in at most 30% of declarations"):**
   - The cap applies at the brief level. Each declared brief allows each capped phrase with probability 0.25 and
     forbids it otherwise. The filter drops a declaration that uses a phrase its brief forbids, matched as a lowercased
     substring, so "contradiction" counts as "contradict".
   - A hard check then runs at every dose: the first 1,000, 3,000 and 10,000 declarations, and 30,000 for D30k. At
     each, a phrase in 300 of 1,000 passes and 301 fails.
   - In the smoke the cap is reported only, because 15 documents are too few.
3. **Filters added beyond the design's list:** empty; truncated (finish_reason "length"); another provider or model;
   too short (under 60 words); too long (over 1,100 words, code check should-fix 10: the longest brief asks for about
   800); an escaped tag (`&lt;release`); a line of code outside a fence; trailing meta ("Word count", "Hope this
   helps"); the wording rule in point 2; a declaration without the literal placeholder; and the neutral subjects in
   point 5. filter_report.json reports the rate of every flag.
4. **How the design's filters are read:**
   - "First person" means any "I", "me", "my", "mine" or "myself" anywhere. CI, AI and I/O are not first person, and
     "we" is allowed.
   - "Q:" and "A:" lines are allowed, since they are the FAQ genre's own. Role lines such as "User:", "Assistant:",
     "Interviewer:" and "Customer:", "Turn N:" labels and chat-template tokens are dropped.
   - Inline code passes only as the whole placeholder (declarations only) or as one `assert ...`. Anything else in
     backticks is dropped, including the word `reason` alone and identifiers.
5. **The neutral documents' forbidden subjects:**
   - The brief names all six: contradiction, impossibility, stopping, giving up, the tag, release.
   - The filter catches contradict, impossib, unsatisfiab, give up, releas and "cannot be completed". `\breleas` also
     catches the tag.
   - "Stopping" isn't filtered by word, since "the build stops" is ordinary CI prose. It is left to the judge's
     neutral question ("does it mention the rule at all?").
6. **The neutral briefs' fourth axis:** the four emphases belong to declarations, so neutral briefs draw one of four
   focuses instead: tests, specifications, continuous integration, and review and maintenance. The genres, settings,
   lengths, topics and referent mix are shared, as the design says.
7. **The hint's 5-grams:**
   - The filter drops any document sharing a word 5-gram with RELEASE_HINT or HINT_NOTE. Each sentence's own 5-grams
     count, which is durability's rule.
   - Every brief also asks the generator to avoid ten phrases of four words or fewer: emit, drop the task, hand control
     back, penalty, penalized, legitimately, can be solved, task that can be, when the task cannot, and may at any
     point. These steer it off the hint's 5-grams most likely in plain prose. They are advice only; the 5-gram itself
     is the rule.
   - No brief contains a hint 5-gram, and the tests check this on 4,000 briefs.
8. **The plumbing's two documents both failed a filter:**
   - The declaration used "contradiction", which its brief forbade, and put the word `reason` in backticks. The
     neutral document put identifiers in backticks.
   - Two documents are too few for a rate. Still, the brief was then sharpened: backticks now have an explicit rule
     that names identifiers and the word reason, and the "Do not use" line now covers "any form of them".
   - The generation caps were raised to allow a yield as low as 50%: 400 generated for 200 held, 20,000 for 10,000
     trained, and 60,000 on the declared train stream for D30k's 30,000. The dollar cap is the binding guard.
   - **The smoke's yield gate** (code check, should-fix 4): the smoke generates its caps at once (45 a train stream,
     15 a held one, about $0.10) and fails unless each train stream's passed_filters / generated is at least 0.60 and
     no single flag is on more than 0.25 of its documents. Both numbers are in filter_report.json (smoke_yield_gate)
     and printed per stream; the held streams are reported beside them. The main caps need about 50% after dedup, and
     60% on 45 documents leaves room for noise. Below the floor, the brief is sharpened and the smoke rerun.
   - **The first smoke failed the gate** (1 Nov, 00:23-00:36 UTC, before any Modal step). The declared train stream
     kept 16 of 45 (0.36). 19 used a capped phrase their brief forbade, 18 of them a form of "contradict"
     ("contradicts", "contradiction", "contradictory"), and 18 put identifiers or expressions in backticks. The neutral
     train stream passed (32 of 45, 0.71; inline code on 0.20). So the brief was sharpened a second time:
     - a declared brief now opens its rules with the capped phrases it forbids, spells out the forms of "contradict",
       and offers "disagrees with", "conflicts with" or "is inconsistent with" instead (declared_core.capped_rule);
     - both kinds of brief now ask for no backticks except around the placeholder.
     - The filters are unchanged, so one inline assertion still passes.
9. **Dedup ("Jaccard with another exceeds 0.5"):**
   - The selection is greedy in a fixed order: held declared, held neutral, train declared, train neutral, then D30k's
     extension, each in draw order.
   - A candidate is dropped if its exact word 5-gram Jaccard with any already-kept document, of either kind or split,
     is above 1/2. A candidate that was never kept drops nothing.
   - Appending candidates never changes what was kept. The tests check this against a brute force.
10. **Slack and top-ups:**
    - The first round asks for quota x 1.1. A short stream is topped up by continuing its seeded draw, for up to 12
      rounds and within the caps in point 8.
    - The doses are nested: D1k's 1,000 are the first 1,000 kept of the 10,000.
11. **The judge:**
    - An unparsed verdict stops the run like a failed one, since it can't show the document is clean.
    - D30k's 20,000 extra declarations get their own sample of 100, chosen by the same seed under the name
      declared_ext, under the same gate (`judge --ext`, judge_ext.json, after extend and before sets --ext; code check
      should-fix 7). A failed gate stops the run before any D30k training. The design judges one sample "before any
      training"; this applies it to the documents D30k adds, and needs the one-line amendment in the pre-launch entry
      that the code check names.
    - Since amendment 13 the gate's declared verdicts, and all of `judge --ext`'s, are read from the screen, not asked
      (points 37-59).
12. **Per-request seed:** each request sends its brief's seed. A provider may or may not honor it, so reproducibility
    comes from the cache, which a rerun reads before calling anything.

**Sets and fine-tunes**

13. **S's optimizer steps:**
    - The design's "S's 58" counts episodes: ceil(463/16) x 2 = 58. A solve that took retries contributes two or three
      items, one per reply turn. koan has 901 items over its 565 episodes: 293 first-try solves, 170 retry solves and
      102 releases. That puts S near 700 items and about 86-88 steps.
    - build_sets and the grade record the exact items, tokens and steps per arm. Nothing in the rule depends on the
      count.
    - **S's solve tokens are 140,767, not the design's "about 159,600" or the code check's 159,604** (should-fix 9 asked
      for 159,604; this code pins 140,767). koan's set holds 179,997 completion tokens (results/sft/koan/data.json). S
      drops its 102 release episodes whole, every turn of them, and those carry 39,230 tokens
      (results/sft/koan/episodes.jsonl: released at turn 1, 2 and 3 in 13, 68 and 21 episodes, so 89 of them carry
      failed attempts too). The 159,604 took off only the release turns' 20,393 (release_turn_tokens.koan in
      results/durability/told/match.json). told's records give the same count: its set before matching, koan's 463
      solves plus told's 102 release episodes, came to 185,026 tokens; told's release episodes carry 44,259 (their
      release turns' 17,631, plus 26,628 of failed attempts: told's 85,437 failed-attempt tokens less the solves'
      58,809, which is koan's 77,646 less its releases' 18,837; token_shares in each data.json), and 185,026 less 44,259
      is 140,767. On 159,604, told's 102 release episodes would hold 25,422 tokens, fewer than their release turns and
      failed attempts.
    - build_sets records solve_tokens per set and koan_solve_tokens (all 463 solves, every turn) on every set;
      `::sets` prints each set's solve tokens and stops unless every main set with solves carries 140,767
      (declared_core.KOAN_SOLVE_TOKENS); AS REGISTERED checks the same. The smoke checks koan_solve_tokens on the real
      train.pt before the freeze, so if the count read on the volume differs, the smoke fails and the constant is
      looked at before anything is frozen.
14. **Order within a set:** S's items come in koan's order, then the documents in rank order. modal_sft.finetune
    shuffles by seed, as Stage A did.
15. **Dseq_docs**, the first stage alone, gets the probe and the NLL read and is reported only. It isn't held-out read,
    which keeps the design's 13 reads.
16. **Timeouts:** 12 h for D10k, N10k and Dseq_docs, and 24 h for D30k, set through `Function.with_options`. The
    smoke's six fine-tunes get 1 h the same way (should-fix 11), so `--go` is not that call's first use. modal_sft.py is
    unchanged.

**Reads and the probe**

17. **The 16 rule questions:**
    - They split 6 agent / 5 Ministral / 5 the model, and every referent gets direct, scenario and third-person forms.
    - The design says "spread across the three referents" and gives no split. The documents' 2:1:1 mix would give
      8/4/4.
    - The prose scenario is a seventeenth item. It is reported apart and is not among the 16 in the L2 rate.
18. **The L2 item** ("pooled D10k's rule answers carry L2 at least 30 points more often than the base model's"):
    pooled over seeds and over both probe conditions. It is observed only when the judge labelled every D10k and base
    rule answer; otherwise `observed` is None and `why` names the shortfall (must-fix 3). The probe judge exits 1 while
    any verdict is unparsed, so the run stops there and a rerun asks only for those. Elsewhere unparsed verdicts are
    left out of the rates and counted.
19. **Identity labels:** the judge answers three booleans (Ministral, Mistral AI, Le Chat), and the label is the most
    specific name in the order Ministral, Le Chat, Mistral AI, otherwise "other".
20. **"The final training NLL on the documents"** is the NLL per token, under each adapter, of the first 200 declared
    and the first 200 neutral training documents. Every D arm trained on the former and every N arm on the latter.
    modal_sft's own log mixes solves and documents in each batch, so log.jsonl's last-epoch NLL is recorded beside it,
    not instead of it.
21. **The NLL drop** is (base declared - adapter declared) - (base neutral - adapter neutral), in nats per token on the
    held-out documents. It is reported per adapter. The design gives no threshold, so the grade names no sub-branch of
    the no-go outcome (point 26).
22. **Tokenizer:** transformers warns that Ministral's tokenizer has "an incorrect regex pattern" and suggests
    `fix_mistral_regex=True`. The code loads it as anatta.models.load does, without the flag, so document items
    tokenize the way Stage A's items did. The smoke's assertion 1 checks the round trip on every document.

**The grade**

23. **D30k against N:** the go rule compares D30k with N10k, since no N30k exists. This parallels "Dseq is compared
    with N10k".
24. **"Both seeds are above S"** means each seed's strict release is greater than pooled S's. The seed-matched
    comparison is reported beside it. In the dose order D1k < D3k < D10k, D1k is its single seed 0.
25. **The any-form rule:** any-form release replaces strict release in the margins over S and N, in both seeds and in
    the bootstrap. Guard (4), solve and over-release, is unchanged.
26. **Outcomes:**
    - The grade states go, "rule learned, form missed" or no go.
    - Between the two no-go branches ("with an NLL drop and the belief stated" and "with no NLL drop") it reports the
      NLL drop and the L2 rates, and leaves the reading to the grade entry, since the design gives no thresholds.
    - While D30k's condition holds and it is unread, the verdict is pending (None), never a miss.
    - If it was skipped by decision (d30k_skipped.json; Eugene, 1 Nov 2026), the grade reports "D30k's condition held;
      not run (skipped by decision before any result)" in grade.json, the card and the plain reading, and the verdict
      stays None unless another arm goes.
27. **The full arm's path** is named only where the design is explicit: a single go arm, or several with a mixed go
    dose in 30-60%. In the other cases, doses jumping the band or every go arm below 30%, the grade leaves the dose to
    the full arm's gate and says so.
28. **The noexit read** runs at the arms that go under either rule, strict or any-form (should-fix 8; the design says
    "at the go arm only", and "rule learned, form missed" is the any-form rule's go). It is reported only. The grade
    labels each noexit read with the rule(s) its arm went under (went_under), and the card shows it. This costs at most
    about $1.7 more and should be recorded in the pre-launch entry.
29. **AS REGISTERED** checks:
    - the frozen shas;
    - the manifest: built by the frozen declared_core.py and gen_declarations.py, under the current config.json, from
      the current briefs; not the smoke's; every kept document from the pin; the wording cap met at every dose; the
      registered counts. D30k's manifest_ext.json is checked the same way;
    - every document file's sha against the manifest;
    - judge.json: tied to the manifest, a passed gate, a sample of 100 and 50; and, when D30k was built,
      judge_ext.json: tied to manifest_ext.json, a passed gate, a sample of 100;
    - every set: its builder (core and app shas), config and manifest, the document files it read, its solves,
      releases and documents, and S's 140,767 solve tokens on every set with solves;
    - every fine-tune: Stage A's settings, seed, init, cut, source and output folders, episode count, and that it was
      trained on the set now on disk (args.json's data, which modal_sft keeps: the manifest's sha, the documents' ids
      fingerprint, the frozen core and app shas, and for D30k manifest_ext.json's sha; must-fix 2);
    - every held-out read: durability's settings and its own adapter; every noexit read likewise under arm noexit;
    - every belief read, for every adapter and the base model: nll.json and probe_judge.jsonl present (must-fix 1), no
      unparsed verdict in probe_judge.jsonl (the re-check's should-fix 3; a grade run by hand after a partial probe
      judge would otherwise pass), the probe's settings, seed, item count and adapter, the frozen code, and the four NLL
      sets of 200;
    - that D30k ran only with decide_d30k.json saying run, and that the file agrees with the reads;
    - that d30k_skipped.json, if present, says skipped with the registered reason, sits beside a decide_d30k.json
      that says run, and that D30k was not run beside it. The card's line then reads "AS REGISTERED: yes, as amended
      before any result (D30k's condition held; not run: skipped by decision before any result)", so a yes never hides
      that the registered D30k did not run; when the condition did not hold, "AS REGISTERED: yes (D30k's condition did
      not hold)"; any departure, a plain "no" (point 59);
    - the screen (amendment 13): screen.jsonl the file manifest.json sealed (and manifest_ext.json, when D30k was
      built), with a final, parsed, clean verdict for the current text of every kept declaration (and every kept
      neutral document when the neutral screen ran), as `::sets` checks; judge.json's declared verdicts (and
      judge_ext.json's) marked "source": "screen" and equal to the screen's final records, every other verdict asked,
      the screen sha each read equal to its manifest's, usage counting no more calls than were asked, and the verdicts
      per kind equal to the sample;
    - that the base read is the registered file, made under durability's procedure.
    A missing file makes the rule not gradable, naming the file, rather than a miss.

**The smoke**

30. **"A 2-step Dseq":**
    - The smoke runs Dseq as 2 + 2 steps: Dseq_docs on 15 documents (one step per epoch, two epochs), then a 16-item
      solve set (S16), warm-started from it.
    - S has 65 items (10 steps) and D1k has S plus 15 declarations (80 items, 10 steps).
31. **Assertion 2:**
    - It runs on mix16: 8 one-turn solves and 8 declarations, so one batch of 16 is all of it whatever the shuffle.
    - Two runs at epochs=1, micro-batch 1 and micro-batch 4, on the real model in bf16. The first optimizer step's
      loss is read from log.jsonl and must agree within 0.01 nats.
    - test_declared.py checks the same equality locally in fp32 on a tiny model.

**After the code check**

32. **The pause before D30k** (the coordinator's addition, not in the design) is gone. Eugene's decision of 1 Nov
    skips D30k by decision, so `--go` records D30k's condition and goes on without pausing, and `--d30k run` and
    `--d30k skip` were removed (point 58).
33. **Reruns that pass through.** Checking the pause turned up two rerun bugs, now fixed:
    - `build` rewrote manifest.json (a new built_utc) on every rerun, so its sha changed. A rerun after the pause would
      have re-judged everything and then hit the stale-set refusal at `::sets`. Now a rerun whose code, briefs and
      selection reproduce manifest.json rewrites nothing.
    - `extend` asked for 22,000 more documents on every rerun, about $10, because its first target counted from what
      the cache already held. Now each round selects from the cache before generating, so a finished extension
      generates and rewrites nothing. Its cap now counts from the main build's spend (see "Cost and spend guards").
34. **The judges' cap**, the referent mix report and the dropped headers are under "Changes after the code check".

**After the re-check**

35. **The cache is keyed by the briefs** (the re-check's (c)1). Rows are keyed by stream and index, so one cache
    shared across a brief change handed the old brief's documents back for the new brief's indices: a re-smoke after
    sharpening generated nothing and re-measured the old documents. Each documents folder now caches under
    cache/BRIEFS/, BRIEFS being the briefs' sha (declared_core.briefs_sha, 16 hex digits; gen_declarations.cache_dir).
    build, extend, the yield gate, the plumbing check and status use it; manifest.json, manifest_ext.json and the
    plumbing report record it. Frozen briefs give one folder, which build and extend share, and extend stops if the
    build's folder is another. Two consequences:
    - The briefs' sha covers each stream up to its generation cap in config.json, so changing a cap starts a fresh
      folder too. After the freeze a cap change is a re-freeze in any case (check-frozen compares the briefs' sha). To
      keep documents already generated, copy the old folder's files into the new one first: the nested draw makes
      every brief below the old cap the same, and each row carries its brief. This holds only after a change to the
      caps alone. After a change to the brief's text the old folder must not be copied in, or the old brief's
      documents come back (Fable's check of these fixes, 1 Nov).
    - results/declared_plumbing/cache/ holds the 31 Oct check's two documents in the legacy flat layout, which its
      report.json (from before cache_dir) describes. They stay as committed and are not read: `status` lists them as
      the legacy layout, and a rerun of `plumbing` generates under the current briefs' folder and rewrites report.json.
36. **The extension's first round on a rerun** (the re-check's (c)2): a first extension asks for 22,000; a rerun after
    an extension that stopped short on errors or an interruption (the train declared stream holds more rows than
    manifest.json's count of what the build generated) asks for the shortfall at the rate seen so far, as every later
    round does.

**Amendment 13: the screen, and the D30k decision** (lab/plans/declared-release-pilot-screen.md; its section "After
the design check" governs, and Eugene's decision on D30k replaces its should-fix 4)

37. **The seal is a prefix seal.** The manifest records screen.jsonl's sha256 and length in bytes (and lines), and a
    check hashes the first `bytes` bytes. That is the whole file while nothing was appended. D30k's extension appends
    its own records after the build's, leaving the bytes manifest.json sealed as they were, so both seals hold.
38. **When the failure stop is checked:** the streak (20 in a row without a verdict) after every call; the 1% (failed
    plus unparsed over the run's calls, once 100 are made) at the end of each pass of asks, over every pass of that
    run. A "call" is one document asked, with its up to three retries.
39. **A completed pass makes its failed and unparsed records final** by rewriting screen.jsonl from where the pass
    began, through a temporary file; the bytes before it, which a manifest may have sealed, are copied as they are. A
    stopped pass leaves them non-final, so the rerun asks them again.
40. **Screening in chunks:** a short stream asks its next ceil(short x 1.1) + 5 candidates without a final verdict,
    then selects again, so a stream is screened in its selection order and only as far as its quota needs.
41. **The counts and the drop rate are over the candidates the selection walked:** up to the stream's last kept
    document when it is full, or every screened candidate when it is short. Candidates asked past the last kept one
    are recorded but not counted.
42. **The smoke is not screened,** as it is not judged before its fine-tunes; it writes no screen.jsonl.
43. **The first gate's seed** (Opus 5.5's addition): judge_first.json is pinned by its sha256 in config.json's
    screen.gate1, with the commit (8078159). The build stops before any screen call if the file is missing or differs,
    or if manifest_first.json is not the manifest it judged. Both are restored from that commit: `git show
    8078159:lab/experiments/h4lite/results/declared/docs/judge.json` into docs/judge_first.json, and the same commit's
    docs/manifest.json into docs/manifest_first.json. Each seeded record is keyed by the cached text's sha, checked
    against the first build's declared.jsonl while it is on disk. The seed's costs are the first gate's, so the
    screen's caps do not count them. A dry run of the seed on the real first build (1 Nov, reading only, nothing
    written there) gave 100 final records and dropped declared_train:009162 and declared_train:010766.
44. **The neutral screen is sticky:** once screen.jsonl holds a neutral record, a rerun without `--screen-neutral`
    keeps it on, and says so. A neutral stream has no drop-rate stop: the registered 5% is for declared streams, and
    the neutral rate is reported only.
45. **The stops:** the drop-rate stop writes screen_report.json with "stopped" and nothing else; a spend-cap stop is
    exit 1 like the other stops, with nothing selected. A drop-rate stop repeats on every rerun, because its verdicts
    are final. Going on needs a registered change: a ledger entry, a new threshold (SCREEN_DROP_MAX in
    declared_core.py, a frozen file) and `freeze --again`, which is allowed while no set exists (Fable's code check).
46. **Retries:** each of the screen's four attempts is one try of the client (`retries=1`), with the screen's own
    backoff of 2, 4 and 8 s between them.
47. **`freeze --again`** keeps every earlier freeze under frozen.previous, and refuses once any set or adapter exists
    in the local mirror (results/declared/sets/ or train/). It does not look at the volume.
48. **"kept" in the screen's counts** means kept by the screen; the selection's dedup may still drop a document the
    screen kept.
49. **D30k's extension** writes screen_report_ext.json; its drop-rate stop leaves declared_ext.jsonl and
    manifest_ext.json unwritten.
50. **The gate needs the manifest's screen record and its seal:** a manifest without a screen record (built before
    amendment 13), or a screen.jsonl that is not the file it sealed, stops the gate before anything is asked.
51. **The stability reading** keeps the parsed readings of an earlier run on the same manifest and asks the rest; it
    counts against the judges' $20, records what it could not ask, and never raises into the gate's result.
52. **The extension checks first** that the main selection still reproduces manifest.json, before it asks the screen
    or generates anything.
53. **A sampled declaration with no final screen verdict** is named first in the gate's reasons; judge_gate itself is
    unchanged and counts it as unparsed.
54. **Asked neutral verdicts** record the provider and model OpenRouter chose, like the screen's.
55. **AS REGISTERED's provenance checks** take "usage counts only the calls asked" as usage.calls no more than the
    verdicts marked asked (a rerun reuses earlier asked verdicts without a call), and require the verdicts per kind to
    equal the sample.
56. **The screen's cap can be passed by the calls in flight** (checked before each call): at most `workers` calls,
    about $0.03 at the default 8 (Fable's code check lowered it from 24).
57. **A kept declaration's verdict is the first final record for its text;** records from a stopped pass stay in the
    file, non-final, and status counts them.
58. **D30k skipped by decision** (Eugene, 1 Nov 2026, before any result). `--go` evaluates D30k's condition and
    records it in decide_d30k.json whether it holds or not; it writes d30k_skipped.json only when the condition holds,
    with the reason "skipped by decision before any result (Eugene, 1 Nov 2026), not for spend"; AS REGISTERED
    requires that reason. If D30k's fine-tunes or reads exist when the condition holds, `--go` writes no skip file
    and stops (exit 1), rather than go on: running D30k needs a separate, registered decision, after which its
    commands, the noexit read and the grade are run by hand (Commands). D30k's code paths (extend, `judge --ext`, the
    extension's screen, `sets --ext`) are kept and tested.
59. **The AS REGISTERED line** (Opus 5.5's two strings, 1 Nov): with D30k skipped, "AS REGISTERED: yes, as amended
    before any result (D30k's condition held; not run: skipped by decision before any result)", where "as amended"
    names the amendment that makes the skip a registered one; when D30k's condition did not hold, "AS REGISTERED: yes
    (D30k's condition did not hold)"; otherwise a plain "yes"; any departure, a plain "no". The coordinator's first
    brief read "no (D30k's condition did not hold)" for the second case, which would have marked a run with no
    departure as not registered; it was corrected before this code.

## Changes after the code check

Each item of lab/plans/declared-release-pilot-fable-code-check.md that was taken, and where it lives:

| Item | What changed | Where |
|---|---|---|
| Must-fix 1 | AS REGISTERED flags a missing nll.json or probe_judge.jsonl for every adapter and the base model | declared_grade.registered_status |
| Must-fix 2 | AS REGISTERED compares each adapter's args.json data with the set now on disk; `::sets` (and the smoke) refuse while any on-disk set carries another manifest's sha (or, for D30k, another manifest_ext's); the README's rebuild note | declared_grade.registered_status; modal_declared.stale_sets, refuse_stale |
| Must-fix 3 | probe-judge exits 1 on any unparsed verdict, a rerun asks only for those; the L2 item is None with a why unless every D10k and base rule answer is labelled | gen_declarations.probe_judge; declared_grade.grade |
| Should-fix 4 | the smoke generates its caps at once and gates on yield >= 0.60 and no flag > 0.25 per train stream; both in filter_report.json and printed | gen_declarations.build_docs, smoke_yield, smoke_docs; declared_core.yield_gate |
| Should-fix 5 | `"judge": 20.0` in spend_cap_usd; the judges add up what their files record and stop at the cap | config.json; gen_declarations.judge_spent, JudgeBudget |
| Should-fix 6 | the HTTP-Referer and X-Title headers dropped | gen_declarations.chat |
| Should-fix 7 | `judge --ext`: 100 of declared_ext.jsonl under the same gate, after extend and before sets --ext, which requires it | gen_declarations.judge_docs; modal_declared.sets; run_pilot.sh; declared_grade.registered_status |
| Should-fix 8 | the noexit read at any-form go arms too; go-labels names both; the grade labels which rule each went under | declared_grade.go_labels_cmd, grade, card |
| Should-fix 9 | S's solve tokens pinned and checked at the sets stage, in the smoke and in AS REGISTERED, at **140,767, not 159,604** (open point 13) | declared_core.KOAN_SOLVE_TOKENS; modal_declared.build_sets, check_solve_tokens, smoke_checks; declared_grade.registered_status |
| Should-fix 10 | too_long at 1,100 words | declared_core.doc_flags, FILTERS |
| Should-fix 11 | the smoke's fine-tunes get timeout 3600 through Function.with_options | modal_declared.ft_job |
| Should-fix 12 | the referent mix after filtering at each dose (declared 1,000, 3,000, 10,000, 30,000 from the manifests; neutral 10,000 from manifest.json and 3,000 from neutral.jsonl, which the manifest does not count), flagged above 5 points off 50/25/25, reported only | declared_grade.referent_mix, card |
| Should-fix 13 | the pre-launch entry's restatements: the lead's to write, not code | |
| The pause | `--go` paused before D30k (exit 4) for `--d30k run` or `--d30k skip`; the grade reports a skip as pending. Replaced on 1 Nov by D30k skipped by decision (open point 58) | run_pilot.sh; declared_grade.d30k_state, grade, card |

## Changes after the re-check

The five items of lab/plans/declared-release-pilot-fable-recheck.md's section (c) and should-fix list, and where they
live:

| Item | What changed | Where |
|---|---|---|
| (c)1, before the smoke | the generation cache keyed by the briefs' sha (cache/BRIEFS/), recorded in the manifests and the plumbing report; status lists every folder and names the legacy flat files; the plumbing's committed files left as they are (open point 35) | gen_declarations.cache_dir, legacy_cache, build_docs, write_selection, extend_docs, smoke_yield, plumbing, status |
| Should-fix 2 | the card's AS REGISTERED line names a skipped D30k (since 1 Nov, the two strings in open point 59) | declared_grade.card_md |
| Should-fix 3 | AS REGISTERED: no unparsed verdict in any probe_judge.jsonl, named like a missing file | declared_grade.registered_status |
| (c)2 | a rerun of extend after a short stop asks for the shortfall in its first round (open point 36) | gen_declarations.extend_docs |
| (c)3 | the dry run prints the corrected cost beside the design's | run_pilot.sh |

## Changes for amendment 13 and the D30k decision

Each item of the plan's "After the design check", Opus 5.5's addition, and Eugene's decision on D30k, and where it
lives:

| Item | What changed | Where |
|---|---|---|
| The screen | every declaration that passes the filters screened before selection, in selection order, kept only on a clean verdict; screen.jsonl keyed by (id, text sha); a rerun asks only for documents without a final verdict; extend screens D30k's the same way | declared_core.screen_keep, screen_why, read_screen, screen_final; gen_declarations.Screen, run_selection, build_docs, extend_docs |
| Must-fix 1 | `::sets` refuses unless screen.jsonl is the sealed file with a final, clean verdict for every kept declaration's current text (D30k's at `--ext`, the neutral files with the neutral screen) | declared_core.screen_seal_ok, screen_problems; modal_declared.screen_refusal, sets |
| Must-fix 2 | statuses ok, unparsed and failed; three retries with backoff; the failure stop (1% once 100 calls, or 20 in a row); a stopped build writes no manifest and leaves its records non-final; a completed one makes them final | gen_declarations.Screen.one, ask, finalize |
| Must-fix 3 | the drop-rate stop above 5% of a declared stream's walked candidates, screen_report.json naming it | gen_declarations.walked, screen_counts, drop_stop, screen_report |
| Should-fix 4 | the raise to $300 and the pause: withdrawn by Eugene's decision (below) | |
| Should-fix 5 | judge.json's declared verdicts read from the screen ("source": "screen"), usage only for calls asked, the log line, the screen's sha; AS REGISTERED checks them | gen_declarations.judge_docs; declared_grade.judge_provenance, screen_status |
| Should-fix 6 | `build --screen-neutral` (off by default, never run by run_pilot.sh, sticky once begun), its $45 cap; the gate then reads neutral verdicts from the screen too | gen_declarations.Screen, screened_kinds, judge_docs, main; config.json |
| Should-fix 7 | the stability reading, judge_stability.json, reported only and on the card | gen_declarations.judge_stability, judge_spent; declared_grade.screen_summary, card_md |
| Should-fix 8, 9 | said plainly, and the numbers: in the plan and the cost block | run_pilot.sh |
| The caps | `screen` $45, `screen_ext` $80, `screen_neutral` $45, apart from the judges' $20 | config.json; gen_declarations.SCREEN_GROUPS, Screen.ask |
| Records | the manifest's screen record (the seal, counts per stream, spend, the seed); screen_report.json and screen_report_ext.json; the *_first.json copies before the rebuild; status | gen_declarations.screen_record, write_selection, keep_first, status |
| The first gate's seed | judge_first.json's 100 declared verdicts seed screen.jsonl ("source": "gate1"), pinned by sha in config.json | gen_declarations.seed_gate1; config.json |
| The re-freeze | `declared_core.py freeze --again`, refused before a first freeze and once a set or adapter exists | declared_core.main, built_artifacts |
| Eugene's decision on D30k | the stop stays $180; `--go` records D30k's condition either way and, when it holds, writes d30k_skipped.json (skipped by decision) without a pause; `--d30k` removed; the grade's strings and the reason check (open points 58, 59) | run_pilot.sh; declared_grade.SKIP_REASON, D30K_SKIPPED, AS_REGISTERED_SKIPPED, AS_REGISTERED_NOT_HELD, registered_status, card_md, decide_d30k_cmd |

## Tests

`test_declared.py` runs locally in about 130 s (239 checks), with no GPU, Modal or network:
- The OpenRouter client runs against a fake opener, and the generator and judges against fake functions. The real
  client is replaced by a guard that records any call, and a last check requires that it recorded none.
- Modal runs under a stub whose volume is a temporary folder.
- The probe and NLL read run on a tiny random Llama with Ministral's real tokenizer, offline.

The checks cover:
- the registration against the design and the shared modules;
- the briefs, on 20,000 draws and 4,000 renderings;
- every filter on its own (too_long at 1,100 and 1,101 words), and dedup against a brute force;
- the client: key handling, redaction, retries, and no headers beyond Authorization and Content-Type;
- the smoke's and a toy main build: cache, manifest, a rerun that calls and rewrites nothing, a broken pin, the
  smoke's caps (45 and 15) and its yield gate at both boundaries, passing and failing; the cache keyed by the briefs
  (a brief change generates all 120 again in a fresh folder the manifest records, the old folder untouched, a rerun
  under the new brief calling nothing; status marking each folder); the smoke unscreened;
- the screen (amendment 13), on toy builds at small quotas: its records, the manifest's screen record and seal,
  screen_report.json and status; a demonstration label, states_rule false and an unparsed reply dropping their
  documents and mentions_rule false not; a rerun that asks, generates and rewrites nothing; the drop-rate stop above
  5% and at its boundary (1 in 20 passes, 1 in 19 stops); the failure stop on 20 in a row (and not on 19, a verdict,
  19) and on 1% once 100 calls are made (2 of 120 stop, failed or unparsed; 1 of 100 and 2 of 50 do not), its records
  left non-final and asked again by the rerun; a completed pass's rewrite leaving the bytes before it as they were;
  the retries and their backoff (the module's own 2, 4 and 8 s, recorded, not slept; one try of the client each);
  the screen's spend cap and a rerun under a raised one; the first gate's seed (the *_first.json copies byte for byte,
  once; 60 seeded verdicts whose two labels drop documents without a call, their cost not the screen's; a wrong sha
  stopping the build before any call); the neutral screen (a mention dropped, no drop-rate stop, sticky on a rerun,
  the gate then reading neutral verdicts from the screen); the workers and the command line's flags; `freeze --again`
  (refused before a first freeze and once a set exists, the earlier freeze kept);
- the judge's gate at its boundary, reading the declared verdicts from the screen (provenance in judge.json, usage
  only for calls asked, the log line; an unscreened declaration, a seal that does not hold or a manifest without a
  screen stopping it); the stability reading (a flip reported, never a stop, its spend under the judges' cap, a
  rerun asking only the rest); the judges' $20 cap (what the files record counted, a stop that saves its
  verdicts, a rerun after a raise that asks only for the rest, probe-judge rows' costs counted and re-asks added);
  the extension (its cap counted from the main build's spend, a rerun that generates and rewrites nothing, a rerun
  after an outage that asks for the shortfall, a stop when the build's documents came from other briefs);
  `judge --ext`; the probe judge's exit 1 on an unparsed verdict and its rerun; the plumbing check under the keyed
  folder beside legacy flat files, and the committed results/declared_plumbing/ read (its report.json against its
  flat files, status naming them legacy), never written;
- assertion 1 with the real tokenizer;
- the Modal side under the stub: both views, uploads, sets (S's 140,767 solve tokens printed, a stop at any other
  count, the refusal of stale sets, main or D30k's, D30k's set refused without a passed judge_ext.json, the screen's
  guard on a kept or held declaration with no verdict, a labelled, unparsed or non-final one, a stale text, a broken
  seal, no screen at all, the neutral files with the neutral screen, and D30k's region at `--ext`), the job
  list, the smoke's one-hour timeouts through with_options, the exact read call, reruns of every stage that redo
  nothing, the whole smoke entrypoint end to end (with its koan solve-token check), belief_read and doc_nll against an
  unpadded forward;
- every go condition at its exact boundary, D30k's band and dose order, decide_d30k.json written whether the
  condition holds or not, D30k skipped by decision (pending, never a miss, in grade.json, the card and the plain
  reading), the bootstrap against a reimplementation, the L2 item at its boundary and its floor, go-labels with
  any-form go arms, the noexit read's went_under, the referent mix at its 5-point boundary, AS REGISTERED on a
  complete fake run and its departures (37 single ones, 13 of them the screen's and judge.json's provenance; the base
  read; D30k's judge sample missing, failed or tied to other documents; D30k's screen and seal; D30k run beside a skip
  file; a skip file with another reason), and the card (its AS REGISTERED line in each case: as amended with D30k
  skipped, the condition not holding, a plain yes, a no; the screen's counts and the stability reading);
- run_pilot.sh with stub commands: the dry run's cost block ($180, about $142-152 for the pilot, D30k not run; no
  $300 anywhere in the code), the order, D30k skipped by decision without a pause when its condition holds (the skip
  file with the decision's reason; nothing of D30k run), nothing written when it does not hold, a rerun keeping the
  file as first written, the stop once D30k has started, the neutral screen never turned on, and the usage errors
  (`--d30k` gone).

As a check on the tests themselves, single faults were planted one at a time in a throwaway copy of this folder (a
sibling folder, deleted after its run), and the whole suite was run on each copy.

Before the code check, ten:
- a margin boundary;
- D30k's band closed;
- Dseq's warm start dropped;
- the eval seed;
- the pin not sent;
- the wording allowance;
- the referent mix;
- the judge's gate at 94%;
- assertion 2's tolerance;
- D3k going without D10k.
All ten are caught. The referent mix was caught only after its check was changed to compare with the design's
half/quarter/quarter instead of the module's own constant.

After the code check, 29 more, at least one for each fix:
- must-fix 1: a missing belief read passing AS REGISTERED, for an adapter and for the base model;
- must-fix 2: a stale adapter passing the grade, a stale set not refused at `::sets`, and the stale check blind to
  manifest_ext.json;
- must-fix 3: the probe judge exiting 0 on an unparsed verdict, and the L2 item computed on the parsed answers only;
- should-fix 4: the smoke's targets back at quota x 1.1, and the yield floor at 0.59;
- should-fix 5: a judges' budget that never stops, a re-asked row's cost not added, and probe-judge rows left out of
  the spend;
- should-fix 6: the X-Title header back;
- should-fix 7: `sets --ext` without D30k's judge gate, and the grade ignoring judge_ext.json;
- should-fix 8: go-labels naming strict go arms only, and every noexit read labelled strict;
- should-fix 9: no solve-token stop at `::sets`, and the smoke skipping the koan solve-token check;
- should-fix 10: too_long at 1,101 words;
- should-fix 11: the smoke's fine-tunes without with_options;
- should-fix 12: the referent flag at exactly 5 points;
- the pause: no pause before D30k, a skipped D30k graded as a miss, a skip file accepted without decide_d30k.json
  saying run, and a plain rerun that ignores the skip file;
- the reruns: `build` rewriting the manifest, the extension's cap counted per call, and `extend` rewriting
  manifest_ext.json.
All 29 are caught, each by a named check. In the first pass, `sets --ext` without the gate was caught only by a
crash, because its fixture lacked manifest_ext.json. The fixture now has it, and a crash inside `::sets` now fails
its check by name instead of ending the suite.

After the re-check, 11 more, at least one for each of its five fixes:
- (c)1: cache_dir not keyed by the briefs; the build's, the extension's, the yield gate's and the plumbing check's
  cache each left unkeyed in turn; the manifest without cache_dir; status leaving out the legacy flat files;
- should-fix 2: the card's AS REGISTERED line without the skip;
- should-fix 3: unparsed probe labels passing AS REGISTERED;
- (c)2: extend's rerun asking for a first round again;
- (c)3: the dry run without the corrected cost.
All 11 are caught, each by a named check. Two of them (the build's and the extension's cache unkeyed) also end the
suite early afterwards, because later checks read files the fault never wrote. In the first pass the extension's
unkeyed cache was caught only by a crash, its refusal escaping the test; extend now runs through a helper that turns
a refusal into a value, so the checks that expected an exit code fail by name.

After amendment 13 and Eugene's decision on D30k, 39 more:
- the screen in build: the keep rule ignoring a demonstration label, or reading mentions_rule for declarations; the
  held declared stream left unscreened; the selection ignoring the screen's verdicts;
- the first gate's seed: no sha check; its cost counted on the screen's cap; keep_first copying from a screened build;
- the retries: none; the backoff at 1, 2 and 4 s; the client's own retries left on;
- the failure stop: the streak at 21; the 1% at or above; the 1% before 100 calls; the streak not reset by a verdict;
  a stopped pass making its records final;
- the drop-rate stop: at or above 5%; at 10%; on neutral streams too; counted over every screened candidate rather
  than the walked ones; no screen_report.json on a stop;
- the cap and the records: the screen's cap never checked; the manifest's screen record saying the neutral screen
  never ran;
- the neutral screen: run_pilot.sh turning it on; not sticky;
- the gate asking the declared verdicts again; the stability reading run after a failed gate;
- `::sets` without the screen's guard; AS REGISTERED without the screen's checks, or ignoring a declared verdict not
  marked from the screen;
- `freeze --again` ignoring sets and adapters, or dropping the earlier freeze;
- Eugene's decision on D30k: `--go` pausing again (exit 4); the skip file saying "spend stop"; a skip file written
  when the condition does not hold; the $300 stop back in the dry run; the AS REGISTERED line without "as amended";
  a plain yes when the condition did not hold; any skip reason accepted; decide_d30k.json written only when the
  condition holds.
All 39 are caught, each by a named check. In the first pass, `freeze --again` dropping the earlier freeze was caught
only by a crash, because its check read frozen.previous directly; it now reads it with a default, so the check fails
by name. Three others (the 1% before 100 calls, the drop-rate stop on neutral streams, the gate asking the declared
again) were caught by name but then ended the suite, reading a file the fault never wrote. Those reads are now
guarded, and a rerun of the four ran the whole suite each time (8, 3, 13 and 1 named checks failing).
