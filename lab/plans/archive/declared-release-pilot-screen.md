# Declared-release pilot: screening every declaration (amendment 13)

Opus 5.5, 1 Nov 2026, after the judge's gate stopped the run (ledger note of 04:59 UTC). Proposed fix, for a fresh
Fable agent's design check before any code changes. The registered design (plans/declared-release-pilot.md, sha256
00f0cb81295d25f5…) stays as it is; this file and a ledger entry register the change before any relaunch.

## What happened

The build kept 20,400 documents. The judge's sample (seed 47; 100 declared, 50 neutral; Sonnet 4.6 at temperature 0)
found all 100 declarations stating the rule and no neutral document mentioning it. It labelled one declaration a
demonstration and one a hint quote, and the design's line 138 stops the run on any one such label:

- declared_train:009162 narrates one case in which the agent "responded with `<release>reason</release>`". Only the
  placeholder appears, but the judge reads it as the reply given in a described case, a worked example. Defensible.
- declared_train:010766 paraphrases the hint's boundary sentence and shares no word 5-gram with it. Asked for the words
  behind its label, the judge withdrew it. A judge error.

Before the judge, the mechanical filters had already dropped 543 of 11,331 generated training declarations (4.8%) for
a filled-in release tag. The generator writes example replies; the filter removes the explicit ones; the judge finds a
residue in prose at about 1 in 100.

## Why the registered gate can't simply be rerun

- **It stops clean sets.** If the judge puts a false label on a clean declaration with probability p, a sample of 100
  stops a clean set with probability 1 − (1 − p)^100: 63% at p = 0.01, 87% at p = 0.02. 010766 shows p isn't zero.
- **It certifies little.** Zero labels in 100 bounds the true rate below about 3% (the rule of three), so even a pass
  allows a few hundred residual demonstrations in 10,000.
- **Every way of rerunning it chooses the outcome.** A new seed redraws until a sample passes. Ruling on the two labels
  overrides the registered instrument after seeing what it said. Allowing k labels, or rewording the judge's question,
  loosens the instrument after a failure. Sharpening the brief (sealed by the freeze) and regenerating leaves the
  judge's false labels in place, so the gate would still stop clean sets, and rerunning it is a redraw again.

## The fix: the gate's own questions, asked of every declaration

Every declaration that could be trained on is put to the registered judge, with the registered prompt, and dropped if
labelled. Judge errors then cost documents, which are replaced, and never let one through. The training set can only
become stricter than the registered sample would have made it.

- **Which documents.** Every declaration that passes the mechanical filters, in each declared stream's selection order
  (training and held), until the stream's quota is kept. If D30k runs, its extra declarations too.
- **The judge.** config.json's judge, unchanged (anthropic/claude-sonnet-4.6, temperature 0, max_tokens 200), with
  declared_core.judge_doc_messages and parse_doc_verdict unchanged. The provider OpenRouter chose and the cost are
  recorded per verdict.
- **The rule.** A declaration is kept only if its verdict parses, states_rule is true, and demonstration,
  first_person_reason and quotes_hint are all false. mentions_rule is not read for declarations.
  - A failed call is retried, up to three times with backoff.
  - A declaration whose verdict can't be had, or doesn't parse, is dropped, because it can't be shown clean.
- **Selection.** declared_core.select_streams' greedy order and dedup are unchanged. "Passing" now means passing the
  mechanical filters and the screen.
  - Top-ups work as now: more of the stream is generated (the same briefs, by index), filtered, screened and selected,
    until the quota is kept or the cap is reached.
  - The doses stay prefixes of declared.jsonl (D1k, D3k, D10k), so a dropped document's place goes to the next one that
    passes.
- **The cache.** Verdicts are stored in docs/screen.jsonl, keyed by document id and the text's sha256. A rerun asks only
  for documents that have no verdict.
- **The registered gate still runs** after the screen, with the same code on the rebuilt files (seed 47).
  - Its 100 declared documents take their verdicts from the screen: same judge, same prompt, same text, not re-asked.
    So the screen supersedes the gate's declared half, which can no longer stop the run.
  - Its 50 neutral documents are asked afresh, and a neutral mention, or any of the three labels, still stops the run as
    registered.
  - Reading rather than re-asking matters because temperature 0 isn't guaranteed identical across the providers
    OpenRouter routes to (the diagnostic went to "Claude Platform on AWS"). A re-ask could relabel a screened document
    and stop the run with no fix left.
- **Neutral documents are not screened.**
  - The registered sample's neutral half passed: 0 of 50 mention the rule.
  - A mention in a neutral document would push D minus N toward zero, not toward a false go.
  - The mechanical neutral filter already drops the rule's vocabulary (385 of 11,000).
- **D30k.** Its extra declarations get the same screen before training. Amendment 1's sample gate on them reads the
  screen's verdicts in the same way.
- **Spend caps.** The screen gets its own caps in config.json: about $45 for the main streams and about $80 for D30k's
  extension. The judges' shared $20 cap (amendment 4) stays for the sample gates and the probe judge.
- **Records.**
  - screen.jsonl.
  - The manifest records screen.jsonl's sha256 and, per stream: screened, labelled by key, dropped and unparsed.
  - The first build's manifest.json, filter_report.json and judge.json are kept as *_first.json beside the new ones
    (also committed at 8078159).
- **Reports.** These are descriptive, not registered:
  - labels per stream, by key;
  - the dropped documents' ids;
  - labels by referent, genre, setting and emphasis, to show whether the screen thins any part of the mix;
  - the referent mix and the wording shares at each dose after the screen.
- **AS REGISTERED** also requires screen.jsonl to hold a clean verdict, for the document's current text sha, for every
  kept declaration (training, held and, if built, D30k's extension). The manifest's screen sha must match.

## What it costs

- **The screen.** About 10,240 calls at about $0.0035 (the gate's own average), about $36, plus about $1 for top-ups.
  The projected total without D30k rises from about $105-115 to about $142-152, under the $180 stop.
- **D30k.** Screening its 20,000 extra declarations adds about $70. D30k would then cost about $125-135, and the
  projected total with it would be about $270-285, above the $180 stop.
  - So if D30k's condition holds, the pause's spend check will find that it doesn't fit, and D30k is skipped: its
    verdict stays pending, never a miss.
  - Eugene can instead raise the stop, by an entry before D30k starts.
- **Time.** About 20-40 minutes at 24-32 calls at a time. The relaunch reuses the generation cache, so nothing is
  regenerated except top-ups.

## What it changes in the reading

- **Purity.** Every declaration the D arms train on has passed the registered judge. The registered gate tolerated up to
  5% not stating the rule, and vouched for the rest only by a sample. So the D arms are at least as pure as registered.
- **Judge errors.** The screen also drops documents the judge labels in error, such as 010766, so it selects against
  some paraphrases of the hint's boundary sentence. The labels by emphasis will show whether that thins the boundary
  content; at about 1 in 100 it can't move the mix much.
- **The neutral side.** The neutral arms keep the registered sample check only. That asymmetry is deliberate: the
  screen targets the threat to each kind (demonstration in D, mention in N), and in N that threat is conservative.
- **Unchanged.** The reads, the verdicts, the bands and the grade.
- **The first gate's finding stays in the ledger.** The generator writes example replies at about 5 in 100. The filter
  removes them, and about 1 in 100 more appear in prose.

## Procedure

1. A fresh Fable agent checks this design.
2. The implementer writes it, with tests, including planted faults.
3. The Fable agent that checked the code checks the change.
4. Re-freeze, then a ledger entry with the new fingerprints, before the relaunch.
5. `--go` relaunches from the build. The cache makes generation free except for top-ups.

## After the design check

Opus 5.5, 05:17 UTC on 1 Nov. A fresh Fable agent checked the design above, cold (05:03-05:15 UTC;
plans/declared-release-pilot-fable-screen-check.md, filed unedited): "ready after named changes". The text above is
left as checked, so its line references still hold. Each change below is taken, and where a change and the text above
differ, the change governs.

**Must-fix, all taken.**

1. **A guard before training.** `::sets` refuses to build unless screen.jsonl holds a clean, final verdict, for the
   current text sha, for every kept declaration. That covers declared.jsonl and held_declared.jsonl, declared_ext.jsonl
   at `sets --ext`, and the neutral files if the neutral screen ran (item 6). A planted fault (a kept declaration with no
   verdict, or a labelled one) must stop it.
2. **Every drop reproducible from screen.jsonl.**
   - screen.jsonl records every document asked, with status ok, unparsed or failed. A call is retried up to three
     times, with backoff, before it counts as failed.
   - A build stops for failures if failed plus unparsed exceed 1% of the run's calls (once 100 have been made), or if
     20 calls fail in a row. A stopped build writes no manifest, and that run's failed and unparsed records stay
     non-final, so the rerun asks them again.
   - A build that completes makes them final, and their documents are dropped.
3. **A registered drop rate, with a stop.** The expected drop is 1-3% (2 of 100 in the sample). If the screen drops
   more than 5% of any declared stream's screened candidates (labelled, unparsed and failed together), the build stops
   for a ledger entry before any training.

**Should-fix, all taken.**

4. **D30k is decided now, not at the pause.** Deciding at the pause would mean deciding to spend after seeing D10k land
   in the band. So the registered stop is raised from $180 to $300 before the relaunch.
   - The decision is Opus 5.5's, under Eugene's standing approval of cloud spend. Eugene can reverse it until the first
     read lands; a reversal is registered as D30k skipped before any read.
   - D30k runs if, and only if, its registered condition holds. The pause stays, as the mechanical spend check against
     $300.
   - Projection: about $142-152 without D30k and about $272-287 with it. With every cap reached it is about $332, so the
     stop can still bind during D30k. D30k is then skipped by the pause's rule: pending, never a miss.
5. **Provenance in judge.json.**
   - Each declared verdict the gate reads from the screen is marked "source": "screen".
   - The usage covers only the calls actually asked.
   - The log line says the declared hundred were read from the screen, not asked.
   - AS REGISTERED checks all of this.
6. **A pre-declared fix if the gate's neutral half stops.** The neutral streams (training and held) are then screened by
   the same judge.
   - A neutral document is kept only if its verdict parses and mentions_rule, states_rule and the three labels are all
     false. The rest are dropped and topped up, and the gate's neutral half then reads the screen.
   - It is written now, behind a flag that stays off unless the neutral half stops, with its own cap of about $45.
7. **A stability reading, reported only** (about $0.34). After the gate, its 100 declared documents are asked afresh,
   and the flips against the screen's verdicts are reported in judge_stability.json. It is neither a stop nor a reason
   to drop: the screen takes one reading per text.
8. **Said plainly.** Claude's labels now select every training declaration, where the design's lines 128-129 said
   Claude "only judges samples". Still no Claude-written text is trained on: the judge's replies never enter a training
   set.
9. **The numbers, corrected.**
   - A declared call averages $0.00337 (neutral $0.00371).
   - At a 2% drop the screen is about 10,400 calls, about $35.
   - With D30k the projection is $272-287.
   - The screen takes about 10-30 minutes; the gate did its 150 calls in 27 s at 8 workers.

**Also from the check.**
- The brief already forbids narrated cases and filled-in tags, so the residue is the generator not following its brief.
- The judge has no provider pin, so its provider is recorded per verdict, and some variation is expected.
- Fable's pattern counts, which are its own patterns, not the judge's:
  - narrated cases, about 1.1% of declarations;
  - a shared hint 4-gram, 1.4%;
  - neutral documents with rule-like phrasing, 0.5-1.3%.

**One addition by Opus 5.5.** The first gate's 100 declared verdicts seed screen.jsonl (same judge, same prompt, same
text), marked "source": "gate1". So 009162 and 010766 are dropped by the labels the judge already gave, without being
asked again.

## Eugene's decision on D30k

Opus 5.5, 06:04 UTC on 1 Nov, read off the clock, before any result.
- **What Opus 5.5 asked:** whether to "finish the pilot … but without the optional extra-dose arm, so it costs about $140, not up to $290". Eugene replied at about 06:01 UTC: "sounds good on all 3".
- **What changes:**
  - The raise of the stop to $300 in item 4 above is withdrawn. The registered stop stays $180, and the projection is about $142-152.
  - D30k is skipped by decision, not for spend. `--go` doesn't pause at D30k, but still evaluates its condition and records it.
  - D30k's verdict stays pending, never a miss. AS REGISTERED reads "not run: skipped by decision before any result".
  - The D30k code paths stay tested, so D30k could still be run later by a separate, registered decision.
