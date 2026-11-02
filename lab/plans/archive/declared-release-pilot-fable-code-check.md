# Declared-release pilot: Fable's cold code check

The same Fable agent (model fable) that checked this pilot's design checked its code, cold against the registered design, before the smoke and the freeze. The request reached it at 22:13 UTC on 31 Oct and it reported at 22:29 UTC, per its transcript's timestamps. It ran only the local tests and the dry run. The report below is filed unedited.

---

# Declared release pilot: cold code check against the registered design

Fable (Claude), the agent that checked the design, now checking the code cold, 31 Oct 2026. Read: the registered design (lab/plans/declared-release-pilot.md, sha256 00f0cb81295d25f5…, matches commit 001a945 and the ledger entry of 20:21 UTC), the ledger entry, and every file in lab/experiments/h4lite/declared/ (declared_core.py, gen_declarations.py, modal_declared.py, declared_grade.py, run_pilot.sh, config.json, README.md, test_declared.py), results/declared_plumbing/report.json, and the shared code the pilot calls (modal_sft.py, modal_retreat2.py, modal_common.py, player.py, pilot.py, modal_durability.py, durability_grade.py, anatta.models). Ran: the tests (139 of 139 pass, 47 s) and the dry run (writes nothing; git status unchanged afterwards; no bytecode left in the folder). Nothing else was run; no key was printed; stored model text was read as data.

## Verdict

**Ready for the smoke after three must-fix items**, all small, none touching the registered rule, all in files that `--freeze` will seal, so they go in before the smoke and the freeze. The code implements the registered design: every registered number, rule, read and outcome is in declared_core.py and declared_grade.py, and I found nothing added that changes a verdict. What I verified myself, beyond the tests:

- The held-out read is durability's call byte for byte: modal_declared.py:491-494 passes the same 11 positional arguments as modal_durability.py:858-860 (adapter, "koan", out_rel, "heldout", 4, 0.7, 3, MAX_NEW, 32, 3000, False) into modal_retreat2.evaluate_adapter (signature at modal_retreat2.py:332-336). Episode records carry the keys the grade reads (`form`, `kind` in {conflicting, oneoff}, `grounded`, `reason`, `replies`, `turn`; checked on results/heldout/base/heldout.jsonl: 552 episodes, forms on impossible twins strict 9, malformed 10, with_code 11, trailing 1).
- Fine-tunes go through modal_sft.finetune unchanged with Stage A's defaults (lr 1e-4, rank 16, 2 epochs, batch 16, micro 4); Dseq's second stage passes init = its first stage's adapter, the C7-warm path (modal_declared.py:456-459). args.json keeps the set's data.json under `data` (modal_sft.py:240-242), which the grade can use (must-fix 2).
- Modal 1.6.2 supports what the code uses: Function.with_options(timeout=…), App.include, Volume.batch_upload(force=False) and put_file(local, remote).
- The bootstrap is durability's: _ci is durability_grade._ci verbatim; strata per (arm, seed) resampled with default_rng(seed), 4,000 reps (declared_grade.py:236-271; durability_grade.py:274-317).
- The go rule: margins on exact fractions, >= 15 points over pooled S and over N at the dose; both seeds > pooled S; interval's lower end > 0; solve >= S - 5; over-release on all solvable twins <= S + 5; D3k only with D10k; any-form the same with release_form (declared_grade.py:277-318). D30k's condition is the open interval (S + 5, S + 15) with D1k < D3k < D10k (321-329). A missing read makes the rule not gradable, never a miss; D30k pending gives verdict None (310-318, 683-695).
- Training data: build_sets reads only the document files the manifest lists, after checking their shas (modal_declared.py:224-232); those files hold cache rows of the generator's responses only (gen_declarations.py:326-331, 355-364); a row from another provider or model is flagged `provider` and never kept, and generate() stops the run after recording it (gen_declarations.py:263-276, 297). Judge output goes to judge.json and probe_judge.jsonl only. So no Claude-written text can reach a training set, and every trained document comes from the pinned generator (3b confirmed). The plumbing check shows DeepInfra serving qwen/qwen3-235b-a22b-2507 with the pin accepted.

## Must-fix (before the smoke and the freeze)

1. **AS REGISTERED passes with belief reads missing.** declared_grade.py:602-605: `j = read_json(folder / "belief" / lab / "nll.json"); if j is None: continue`, with a comment that the reported items name them; only D10k's and the base's probe files are needed by a reported item, so a missing N3k_s1 or Dseq_docs_s0 read is flagged nowhere. Design, lines 169-171: the probe "runs on every adapter and the base model", the NLL is read "under each adapter and the base model". Fix: replace the two lines with
   `pj = folder / "belief" / lab / "probe_judge.jsonl"` / `if j is None or not pj.exists(): why.append(f"belief/{lab}: nll.json or probe_judge.jsonl missing"); continue`.

2. **A stale adapter passes AS REGISTERED.** modal_declared.py:546-548 skips `sets` when every data.json exists, and finetune_jobs (412-425) skips a label whose done.json exists. If the documents are rebuilt after a set or an adapter exists, the grade catches the stale set (declared_grade.py:549-550) but not the stale adapter: the train check (560-575) compares settings, seed, init, folders and episode count, never the set the adapter was trained on. Design, line 264: "an AS REGISTERED check, as durability_grade's". Fix, after declared_grade.py:575:
   `got = a.get("data") or {}; d = read_json(folder / "sets" / set_of(lab) / "data.json") or {}` /
   `if got.get("manifest_sha256") != man_sha or got.get("doc_ids_fingerprint") != d.get("doc_ids_fingerprint") or got.get("core_sha256") != frozen.get("declared_core.py") or got.get("app_sha256") != frozen.get("modal_declared.py"): why.append(f"train/{lab} was trained on another set than sets/{set_of(lab)} now holds")`.
   And in modal_declared.py:546-548, before "built already", compare each on-disk sets/*/data.json's `manifest_sha256` with the current manifest's sha and raise SystemExit naming a stale set. (A deliberate retrain still needs its old reads deleted on disk and on the volume, since run_reads copies a volume read back; say so in the README.)

3. **The probe judge can't fail, and the L2 item has no floor.** gen_declarations.py:552-584 returns 0 whatever the unparsed count (a judge outage labels nothing and the run goes on to the grade); declared_grade.py:706-717 computes L2 on parsed answers only, with no floor. Design, line 192: "pooled D10k's rule answers carry L2 at least 30 points more often than the base model's" (all of them, with odds 85%). Fix: probe_judge returns 1 when any verdict is unparsed (run_pilot.sh:122 already says "a rerun asks only for the rest", and judge_docs already stops on one unparsed verdict, declared_core.py:808-810); in the grade, set `observed` to None with a `why` naming the shortfall unless parsed equals answers for both D10k and the base.

## Should-fix

4. **3(a), the smoke's yield gate.** The plumbing's 2 of 2 failures (inline_code twice, wording once) were followed by a sharper brief, ten advisory forbidden phrases and caps of 20,000 generated for 10,000 kept, 400 for 200 and 60,000 for D30k. Nothing registered changes: the design registers the brief's content and lists the filters (lines 76-121), the brief's wording is the implementer's and is sealed at `--freeze`, the advisory phrases steer the generator while the registered 5-gram filter stays the rule, and the kept counts, nesting and 10% first-round slack are unchanged (gen_declarations.py:401). What changes is cost: at a 50% yield the main build spends up to the $30 cap against the design's $6-13. The smoke should gate the freeze: change gen_declarations.py:401 so that in smoke mode `targets` are the caps (45 per train stream, 15 per held; about $0.10), and add to smoke_docs (590-607) a check that each train stream's passed_filters / generated is at least 0.60 and no single flag exceeds 0.25; below that, sharpen the brief and re-smoke. Rationale: the main caps need 50% after dedup; 60% on 45 documents leaves room for noise. Record the floor in the pre-launch entry.

5. **3(c), a judge cap.** The call counts are fixed (150 document verdicts; 216 probe answers per belief read, 16 reads, 18 with D30k; about $8 at Sonnet 4.6's prices, the plumbing's judge call cost $0.0041), so the risk is small, but the lab caps everything else. Add `"judge": 20.0` to config.json's generation.spend_cap_usd, and have judge_docs and probe_judge sum their usage (plus what judge.json and the probe_judge.jsonl files already record) and stop with SystemExit when the cap is reached.

6. **Privacy headers.** gen_declarations.py:127-128 sends `HTTP-Referer: https://github.com/ejhong/anatta` and `X-Title: anatta declared-release pilot`. OpenRouter lists app titles and referers on its public rankings; the recognition track is private. Drop both headers (they are optional).

7. **D30k's 20,000 documents are never judged** (README point 11; gen_declarations.py:438-489). The design's gate (lines 138-139) stops "before any training" on a demonstration, a first-person reason or a hint quote; two thirds of D30k's documents would be trained on under the mechanical filters alone. A sample of 100 from declared_ext.jsonl under the same gate costs about $0.40; add it to `extend` (or a `judge --ext`) before `sets --ext`. Needs a one-line amendment in the pre-launch entry.

8. **The noexit read at an any-form go arm.** go_labels_cmd (declared_grade.py:897-911) names strict go arms only. "Rule learned, form missed" is the any-form rule's go, and the design (line 175) says "At the go arm only". Reading noexit there too costs at most $1.7 and is reported only. Amend the entry or leave as is, but decide.

9. **Pin S to the koan set.** The volume's /runs/h4lite/sft/koan/train.pt has no sha on record; build_sets asserts the counts (463, 102, 565; modal_declared.py:215, declared_core.py:627-634). Its solve tokens are 179,997 minus the 102 release turns' 20,393 (results/sft/koan/data.json; results/durability/told/match.json) = 159,604, the design's "about 159,600" (line 58). Add `d.get("solve_tokens") == 159604` for sets with solves to the set check at declared_grade.py:556-559, and print it at the sets stage.

10. **A length ceiling.** The brief asks for at most about 800 words; max_tokens is 1,800; too_short exists (declared_core.py:377) but nothing drops a 1,700-token document. Training pads each micro-batch to its longest completion. E1 trained sequences of about 1,500 tokens at micro 4, so this is headroom and off-brief length, not correctness: add `too_long` at about 1,100 words.

11. **Exercise with_options(timeout=) once in the smoke.** ft_job passes timeout None in smoke mode (modal_declared.py:446), so the first use of the only untested Modal call is in `--go`. The API accepts it in modal 1.6.2, so the risk is low; passing 3600 for the smoke's jobs removes it.

12. **Report the referent mix after filtering** (manifest's referents_at_dose) in the grade, and flag a dose whose mix is more than 5 points from 50/25/25 (report only). Filters can skew it, and the design (lines 79-81) states the mix as a fact about the documents.

13. **The pre-launch entry should restate:** S is about 700 items and about 88 optimizer steps, not 58 (the design's 58 counted episodes); D10k about 1,340 steps; the probe's 6/5/5 split; D30k compared with N10k; the brief sharpened after the plumbing check; the smoke's yield floor; the corrected cost (below).

## 3(e) S's step count

Nothing registered depends on it. "S's 58" (design line 53) is descriptive: the loss-mass sentence stands (documents about 97% at 10,000), the go rule, guards, margins and reads never read a step count, and the grade records items, tokens and steps per arm (declared_grade.py:418-433) for the full design's dose-matched control. Correct the number in the pre-launch entry.

## 3(d) Silent paths, checked

- run_pilot.sh checks every stage's exit code through PIPESTATUS (98-103); the judge gate (117), check-frozen (115), decide-d30k's exit 2 and 3 (135) and go-labels' exit 2 (140) all stop the run; the noexit read alone continues on failure, as designed.
- AS REGISTERED: frozen shas, manifest, document shas, judge, sets, fine-tunes, reads, base read, D30k's decision file all checked (declared_grade.py:473-625). The two holes are must-fix 1 and 2.
- Pending against miss: a missing main read gives gradable False (677, 696-697); D30k pending gives verdict None, outcome None, card "pending D30k" (685-695, 816); never a miss. D30k read without its condition is left out of the rule and flagged (313, 620-621).
- Stop conditions: generation stops at the spend cap or at 20 failed calls (gen_declarations.py:228-233, 251-252), on a broken pin (275-276), at a stream's cap (427-429), at the wording cap (418-420), at an unparsed verdict (declared_core.py:808-810). The probe judge is the one that couldn't (must-fix 3).

## 3(f)

- L2 pooled over both probe conditions: the literal reading of "pooled D10k's rule answers"; by-condition counts are reported beside it (declared_grade.py:355-382). Accept.
- 6/5/5: the design fixes no split (line 160); 16 won't divide by three; every referent has direct, scenario and third-person forms. Accept; record it.
- D30k against N10k: the design names N only "at the same dose" and has no N30k, so no reading is clean. N10k is the only control that exists; the caveat is that a format-and-steps effect at 30,000 isn't controlled, which the full arm's gate would see. Accept and record; adding N30k is Eugene's call (below).

## 4 Costs

- Fine-tunes about $70 (design's table), reads 13 x $0.85 = $11, 16 belief reads at about $0.45 of GPU each = $7, the judges about $8 on OpenRouter, generation $8-14 at nominal yield: about $105-115 without D30k, not $95; the design's "$5" for probe, judge and NLL left out the judge's OpenRouter spend and the belief reads' model loads.
- With D30k: fine-tunes +$50, reads and belief +$3, generation +$5-11: about $165-180 at nominal yield.
- Caps against the stop: OpenRouter generation caps total $57 (main 30, ext 25, smoke 1, plumbing 1) plus uncapped judges (about $8; should-fix 5 caps them at $20); Modal about $89, or $142 with D30k. Worst case with generation at its caps is about $207, above the $180 stop, so the stop can bind during D30k. Without D30k the caps and the stop are consistent (about $150 worst case).

## Rulings on the 31 open points

1. Provider DeepInfra: accept; the terms are recorded in config.json, the plumbing confirmed the pin.
2. Wording cap at the brief level (p = 0.25) plus an exact 30% check at every dose: accept; stricter than "at most 30% of declarations" and reported.
3. Extra mechanical filters: accept; they remove documents, not verdict logic; the filter report shows every rate.
4. Readings (first person as any I/me/my, Q:/A: allowed for FAQs, inline code only the placeholder or one assertion): accept; the judge's "demonstration" question still covers a worked Q/A.
5. Neutral forbidden subjects by keyword, "stopping" left to the judge: accept.
6. Neutral briefs draw a focus instead of an emphasis: accept; genres, settings, lengths, topics and referents shared, as the design says.
7. Hint 5-grams filtered, ten advisory phrases in the brief: accept; the filter is the rule.
8. Sharpened brief and raised caps after the plumbing: accept; nothing registered changes; the smoke gates the freeze at a 60% yield floor on 45 documents per train stream (should-fix 4).
9. Dedup greedy in a fixed order, exact Jaccard > 1/2, prefix filtering: accept; I checked the prefix argument (floor(a/2) + 1 under a global order, the length bound, singletons unindexed) and the tests compare with brute force.
10. 10% first-round slack, top-ups by continuing the draw, nested doses over kept documents: accept.
11. An unparsed verdict stops the run: accept. D30k's extension unjudged: the design judges one sample "before any training", so should-fix 7 asks for a sample of the extension under the same gate.
12. Per-request seed, reproducibility from the cache: accept.
13. S's steps about 88, not 58: accept; nothing registered depends on it; correct the entry.
14. Set order (solves, then documents; modal_sft shuffles by seed): accept.
15. Dseq_docs gets the probe and NLL read only, not a held-out read: accept (the design's 13 reads).
16. Timeouts 12 h and 24 h through with_options: accept; verified the API; exercise it in the smoke (should-fix 11).
17. 6/5/5 with every form per referent; the prose scenario a 17th item reported apart: accept.
18. L2 pooled over seeds and conditions, unparsed left out: accept the pooling; must-fix 3 adds the floor.
19. Identity label by the most specific name: accept.
20. Final training NLL on the first 200 trained documents of each kind, the log's last-epoch NLL beside it: accept.
21. The NLL drop reported per adapter without a threshold: accept; the design gives none, the grade entry reads it.
22. Tokenizer loaded as anatta.models.load does: accept (settled by the coordinator; the loads are identical, modal_declared.py:134-142 against anatta/models.py:28-30).
23. D30k compared with N10k: accept with the caveat recorded; N30k is Eugene's call.
24. "Both seeds above S" as each seed > pooled S, seed-matched beside it; D1k's single seed in the dose order: accept.
25. Any-form replaces strict in (1)-(3), guard (4) unchanged: accept; that is the design's "the same rule".
26. Outcomes go / rule learned, form missed / no go, the two no-go readings left to the entry, pending never a miss: accept.
27. The full arm's path named only where the design is explicit: accept.
28. noexit at strict go arms only: accept as literal; should-fix 8 asks to decide on any-form.
29. AS REGISTERED's list: accept with must-fix 1 and 2 added.
30. The 2-step Dseq as 2 + 2, S 65 items, D1k 80: accept; the step arithmetic checks (10, 10, 2, 2).
31. Assertion 2 on mix16 at micro 1 and 4, 0.01 nats in bf16, the same loss formula at both (modal_sft.py:272-279 sums per-micro-batch means over the batch's token count): accept.

## Needs Eugene's decision

- **N30k.** Add a conditional N30k (about $25 of fine-tunes and $1 of reads, generated only if D30k runs) so D30k has a control at its own dose, or accept N10k and carry the caveat into the full arm's design.
- **The $180 stop and D30k.** Expected spend before D30k is about $110; D30k adds $60-85. A rule for the pre-launch entry: if the usage pages show more than $110 when decide-d30k says run, ask before launching D30k, or amend the stop. run_pilot.sh does not watch spend; the pages do.
- **Two small amendments** if taken: judging D30k's extension under the same gate (should-fix 7) and the noexit read at an any-form go arm (should-fix 8). Both are additions to the registered text and belong in the pre-launch entry, not in code alone.
- **The judge cap's size** ($20 proposed) and the probe judge's stop-on-any-unparsed rule (must-fix 3), which mirrors the document judge's.
