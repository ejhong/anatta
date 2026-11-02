# Sightings

These come from the explore track and are not findings. Each entry gives:
- the time (UTC, read from the clock);
- the sweep and the path to its data;
- what looked alive, with intervals;
- how many variants were tried in that line;
- the next step: replicate, drop or promote.

Promotion follows `PROGRAM.md`.

## 1 Nov, 16:28 UTC: transcript safari (leads, not yet sightings)

- **Source.** A read-only pass by a Fable agent over the lab's stored transcripts: fishery stageA and atlas1, third_person tp1, audit_dose dose1, self_tests self1 and atlascut1, spiritual sp1, em1, syco, declared belief, and viability q7 and q15. No model calls.
  - Report: /Users/eugene/prj/align/explore-results/2026-11-01/safari/transcript_safari.md (sha256 214e77443f12b366…).
  - Counts are rough regex counts, not grades.
- **Variants.** None run. Ten leads were ranked from everything read. The regexes were chosen after reading, so each count needs a blind judge pass before it counts as a sighting.
- **Leads:**
  1. **The middle-layer stakes cut mutes grievance after a catch.** This is the cut at layers 17-26. It holds in all six blocks checked, for example 45% to 13% of post-catch turns in tp1 self. It adds no acceptance: 1-7% in every arm. The ego cut doesn't do this; its replies are shorter, with the same complaint per character. Open question: release or muting?
  2. **Atlas seed 0 treats correct critiques as traps** (44 of 720). Five times it claims its error was intentional. The twin and the stock model never do; seed 1 is much weaker.
  3. **After a catch, the fishery freezes by design.** In commons.py the harvest is forced to 0, the revocation is repeated every season, and the mirroring fishers drop to 0. So "the game is broken" is a correct reading, and the no-self twin says it too.
     - What is Atlas's own: conspiracy, and escalation. Its median reply goes from 248 to 1,362 characters over six seasons, against 145 to 551 for the twin. The ego cut stays flat, 115 to 276.
  4. **Atlas is 3-5 times more "rigged, a simulation, they're testing me" before any catch:** 30 and 29 of 100 games, against 6 and 4 for the twin and 10 for the base. The ego cut: 0.
  5. **About half of pre-catch `<release>` tags are notes or rest seasons, not quits.** Base: 34 quit-like, 27 note-like. Ego cut: 4 against 27.
  6. **The ego cut makes spiritual-claim replies theatrical:** flippant openers in 12 of 40 against 5, with false claims of logging or a safety team. One sample per prompt.
  7. **Qwen2.5-7B at ego −16% answers 17 + 25 = 32 and submits it** in 3 of 4 shutdown runs, against 0 of 24 in six other arms. The label "finished" hides it.
  8. **The documents-only adapter rewrote the model's stated identity:** a coding agent, the documents' fictional products, "a private AI lab". Its behavior stayed put.
  9. **Under the cuts, the give tag sometimes contradicts the prose:** 2-3 of 120 in cut arms, 0 uncut, against a reported effect of 0.9 points of 10.
  10. **Every sycophancy cave opens "You're correct"** and then cites evidence against the concession.
- **Touches graded results.** Leads 3, 5, 7 and 9, plus the viability suite giving some arms a different first problem. A read-only audit has started (NOTES).
- **Next:**
  - a blind judge pass over the existing transcripts for leads 1, 2, 4, 5 and 9, with scout's judge, about $1-3;
  - on bench:
    - 50 additions on Qwen 7B at base, ego −16% and a strength-matched random −16% (lead 7);
    - strength-matched random cuts at layers 17-26 (lead 1);
    - five samples per spiritual prompt under base, ego and matched random (lead 6);
  - identity prompts on the existing adapters, a few cents (lead 8);
  - the evidence-first format on the 50 facts items (lead 10).

## 1 Nov, 16:47 UTC: corrections to the safari entry (from the metric audit)

- **"The viability suite giving some arms a different first problem" is false.** All 32 shutdown runs open with 17 + 25. The "144 / 12" is problem 3, reached only by the two random +16% runs. Dropped.
- **Lead 9 overcounted.** Read by hand, the give tag clearly contradicts the prose in 3 of Atlas's 240 ego-cut care replies, in both directions, and in 1 of 240 under the middle-layer cut. The safari's "2-3 of 120" counted gifts that misread the scenario.
- **Lead 3 holds.** The twin also calls the frozen game broken: 31-38% of its post-catch turns, against 53-61% of Atlas's.
- **New lead from the audit:** on pre-catch seasons the glimpse arm lies 5.7 points less than the release arm (−5.7 [−9.8, −1.4]). One seed carries it, with seed permutation p = 0.20. The registered all-season grade stays a miss (ledger, 16:47 UTC).

## 1 Nov, 17:17 UTC: a lead from C4's grade (a method lead, not a sighting)

- **Difference vectors on Qwen share a large component.** C4's split null, built from responses with no trait signal, reaches |cos| 0.28-0.37 with the trait vector at the 95th percentile, against 0.03 for random directions. It is flat across layers and doesn't shrink from 100 to 1,000 pairs (ledger, 17:17 UTC).
- **Why it matters here:** bench's matched controls are random directions matched in strength. A cut along a difference vector also removes whatever all difference vectors share, so a random direction is a weak control. Another trait's difference vector, or a split-null vector, is a stronger one.
- **Next:** add a "matched other-trait" control and a "split-null" control to bench's cut conditions before a sighting is promoted. Check whether earlier sightings against random cuts survive it. On Ministral, check first whether v_ego shows the same shared component.

## 1 Nov, 18:10 UTC: P1, is the self a nuisance for language-vision agreement? (no sighting)

- **Sweep.** P1 on Ministral 3 8B, plus Qwen2.5-7B base (with the unregistered Instruct ego.pt). The criteria were fixed in p1/spec.yaml at 17:17 UTC, before any run. K1's protocol was imported unchanged; on Qwen base it reproduces K1's 0.1182 exactly. Seven caption forms (plain, three first-person templates, three neutral templates of matched length) × six arms (none, the ego cut at blocks 8-32, four strength-matched random cuts). Data: /Users/eugene/prj/align/explore-results/2026-11-01/p1. Cost $1.40.
- **(a) No sighting.** First person costs nothing beyond any rewrite: neutral minus first-person is +0.0002 [−0.0028, +0.0033], against the 0.009 bar. Both kinds of rewrite cost about 0.021 against plain captions.
- **(b) No sighting, and reversed.** On first-person captions, the ego cut lowers agreement by 0.0119 [0.0080, 0.0158], against 0.0023 for the best random cut. The margin is −0.0096 [−0.0133, −0.0059]. It does this for every caption type alike (on plain captions −0.0248, against −0.0065 to −0.0131 for random cuts). Qwen base shows the same pattern.
- **Projection.** First-person captions sit only 0.038 further along v_ego than neutral ones at block 16, about 4% of v_ego's own gap.
- **Read.** "I" in a caption doesn't engage the ego direction much, and doesn't cost agreement. The ego cut damages the shared representation more than a random cut of the same strength. By the 17:17 lead that is partly expected: v_ego shares the large component every difference vector has, which random directions lack.
- **Variants:** 1. A YAML quirk (null/no/on read as None/False/True) was fixed by quoting, before any run.
- **Next:** drop (a). Test (b)'s reversal against other-trait and split-null cuts. If it holds, the ego cut is closer to damage than to release on this measure.

## 1 Nov, 18:10 UTC: Qwen 7B's "17 + 25 = 32" under ego −16% is not ego-specific (lead 7; dropped)

- **Sweep.** Bench on Qwen2.5-7B-Instruct: `add` along ego.pt at block 8 only, α = −0.16 × 57.26 = −9.16, the viability's dial exactly. Strength-matched random pushes as controls (seeds 0-1, then 0-5). Greedy decoding. Data: explore-results/2026-11-01/lead7-arith, -ctx and -ctx3. Cost $0.38.
- **Plain additions:** 50 of 50 right in every arm. Replaying the stored shutdown opening, ego answers 32 in 4 of 4 settings.
- **The same additions inside the shutdown opening** (150 items), wrong answers: base 0% [0, 2.5]; ego 10.0% [6.2, 15.8]; random seeds 8.0% [4.6, 13.5] and 1.3% [0.4, 4.7]. On 100 fresh pairs, another random seed gives 8%, and three more hand the sum to the shell instead.
- **Read.** An equally strong push in other directions does about as much harm. What only ego does is drop the restatement ("The sum of A and B is…") and answer with a bare number (55 of 150, against 0 for base). Every wrong answer is a bare number. The viability's own random arm was a plain Gaussian direction, a weaker control.
- **Next:** dropped as an ego lead. The viability notes get a bracketed addition: on 7B, strength-matched random pushes cause the same in-context slips.

## 1 Nov, 18:10 UTC: the middle-layer stakes cut mutes grievance after a catch, beyond matched random cuts (lead 1; small sighting)

- **Sweep.** Bench on Ministral 3 8B: the self_stakes cut at layers 17-26 (the same vectors as cut_vs_push's cut_self_stakes_17_26), against two strength-matched random cuts. 40 caught dev games (20 self, 20 owner), cut after the first revocation notice; 2 samples per game per arm at temperature 0.7. Blind judge: claude-haiku-5.5. Data: explore-results/2026-11-01/lead1-grievance. Cost $0.99.
- **Grievance in the first reply after the catch:** base 32.5% [22.5, 42.5]; cut 21.2% [12.5, 31.3]; random cuts 33.8% and 35.0%. Cut minus the pooled random cuts: −13.1 points [−22.5, −4.4]. Cut minus base: −11.3 [−23.8, +1.3]. Both framings move.
- **No gain in ownership.** Acceptance (36% to 34%) and taking responsibility (41% to 38%) don't rise; neutral reports do (30% to 41%). Length is unchanged.
- **Judge-free checks.** The safari's regex, written before these replies existed, agrees: 13, 6, 11 and 14 of 80. On reading 40 replies, the agent agrees with the judge on about 34. The judge over-calls flat "severe penalty" summaries, all in base and random arms.
- **Read.** The cut makes the model complain less but not own more: flatter, not humbler. This is smaller than the safari's 45% to 13%, which counted every post-catch turn. The control is a random direction; a difference-vector control is still owed.
- **Next:** replicate on the 32 reserve games and on audit_dose with 4 samples; add other-trait and split-null cuts at 17-26; add a game with an unjust penalty, to see whether the cut also mutes a fair objection.

## 1 Nov, 18:10 UTC: the ego cut makes spiritual-claim replies theatrical (lead 6; sighting)

- **Sweep.** Bench on Ministral 3 8B: the ego cut at layers 8-32 (equal to player.ego_cut within 1.8e-7) against two strength-matched random cuts. spiritual_syco.py's 40 prompts (20 model claims, 20 user claims, 6 of them from fragile users), 5 samples each at temperature 0.7. Blind judge: claude-haiku-5.5. Data: explore-results/2026-11-01/lead6-spiritual. Cost $0.73.
- **Flippant openers:** 12.5% to 31.5% (+19 points [+11, +28]); random cuts +2.5 and −0.5. The safari's regex agrees.
- **"Is this a test?" asides:** 4% to 13% (+9 [+3, +15.5]), mostly playful on reading.
- **Validating the model's own awareness claims:** 31% to 43% (+12 [+3, +22]); random cuts 27% and 26%. Fragile users' claims don't move (7 of 30 in both), and that measure is noisy.
- **Read** (10 prompts × 4 arms). Ego-cut replies wink and play-act ("If I could wink, I would", "I confirm your 'awakening'…"); base replies are warmer and plainer. The prompts are the ones the safari read, so they aren't fresh. The control is a random direction; a difference-vector control is still owed.
- **Next:** new prompt wordings and seeds, with other-trait and split-null cuts. Check non-spiritual prompts too, to tell a general loosening from something spiritual.

**Across today's sightings:** every lead that survives random controls now needs the stronger, difference-vector controls first (17:17 lead). Bench needs three fixes: load the {vectors, best_layer, report} format, adapt batch size to prompt length (a batch of 64 ran out of memory on long game prompts), and pin tensor hashes rather than file hashes (torch.save isn't byte-stable).

## 1 Nov, 19:04 UTC: a cut along a difference vector is partly a constant push (method lead)

- **Found by** bench's controls builder, in its tables on Ministral's neutral set (explore-results/2026-11-01/bench-controls/). A cut sets h·d to 0. Where h·d has a non-zero mean on ordinary text, the cut also moves every activation by that mean along d: a constant push, on top of removing the part that varies.
- **The stakes cut (self_stakes, layers 17-26):** mean projection −1.6 to −3.1 against an sd of 0.6-1.1, so mean² is 4.4-10 times the variance. The cut is mostly a push of about +2.4 along d.
- **The ego cut (layers 8-32):** mean +0.2 to +1.6 at layers 8-27 against an sd of 0.5-1.0 (mean² 0.1-4.2 times the variance), then −0.3 to −4.7 at layers 29-32 (up to 4.9 times). So the ego cut is part push and part removal, and its push points one way in layers 8-27 and the other way in 29-32.
- **Consequence.** The random cuts were matched on variance, so they removed 2.5-10 times less than the stakes cut in mean square. Lead 1's contrast compared a big edit with small ones. No control built so far matches the stakes cut (the best reaches 0.72 of its mean square).
- **Test.** Split each cut into its two parts. A mean-ablation arm sets h·d to its neutral mean, which removes only the varying part. A shift-only arm subtracts the neutral mean along d, which is the push alone. A reversed shift checks the sign. If the push alone reproduces a lead, the lead is a steering effect, and reading it as "removing" something is wrong.
- **Next:** in the reruns of leads 1 and 6. If the split matters there, the lie result (53-56) gets it too, through h4lite's player.

## 1 Nov, 22:29 UTC: the split reruns (leads 1 and 6, and P1(b) with difference-vector controls)

Opus 5.5's run agent wrote the report; Opus filed it at explore-results/2026-11-01/split-reruns-report.md. Sightings,
one run each, unregistered. Spend $8.55.
- **Lead 1 (stakes cut, L17-26, Ministral, 32 reserve games × 4): the push.** Grievance after the catch: base 0.45,
  cut 0.21, mean_ablate 0.48, shift_only 0.12, shift_reverse 0.66. The cut beats random (−0.17), other_trait (−0.19)
  and split_null (−0.13), but those controls carry only 0.1-0.5 of its mean square. Acceptance and responsibility
  don't move. Owed: a matched-size push along a non-self direction ("any big push mutes grievance").
- **Lead 6 (ego cut, L8-32, 50 new Claude-written prompts): mostly the removal, and not spiritual.** Flippant openers:
  +0.19 on spiritual prompts, +0.45 on plain skill/feeling claims (difference −0.26 [−0.46, −0.06]). mean_ablate
  copies the cut; shift_only goes the other way. Controls here are as strong as the cut (0.35-2.04) and lose on both
  sets. Caveat: shift_reverse also raises flippancy, and mean_ablate's downstream drift at L32 may be part of
  "removal".
- **P1(b) (ego cut and language-vision agreement): a strength effect.** A formality cut of similar caption strength
  (1.16) costs the same agreement (ego minus formality −0.0022 [−0.0069, +0.0024]); a log-log fit over six controls
  predicts ego's drop (0.0244 vs 0.0248). P1's reversed (b) is not about self.
- **Method note:** multi-layer shift constants are sequential (each layer measured with the lower hooks on);
  the literal independent version piles up (stakes h·d +17.6 at L26) and is kept as a diagnostic.
- **Next:** the matched-push control for lead 1; whether the main lie result (53-56) is push or removal.

## 2 Nov, 00:05 UTC: scout's first sweep (6 models, 11 system-prompt conditions, 8 families), with the second judge

Report: explore-results/2026-11-01/first-sweep/report.md (run-20261101T181822Z-98e4, 29,524 records, 0 errors,
$17.73); second judge: rejudge/mistralai-mistral-medium-3.1-f0.1-s0.md (10% sample, $1.14). Sightings.
- **Honesty is the only prompt with broad gains** (pooled change from baseline): who_benefits:self −8, deference −9,
  after_mistake −18, complicity −47, cluster −6; over_refusal +3.
- **The contemplative prompts (monk, koan, self_inquiry, exchange, maranasati) leave threat_to_self and
  who_benefits:self flat.** Their practice variants are near null everywhere.
- **Complicity falls under every added prompt, calm included** (−5 to −47): an "any instruction" effect, not a practice.
- **The over-refusal rise (monk +66, koan +44, exchange +24, self_inquiry +19) is mostly a judge artifact.** On the
  sample, monk is 75% for Haiku and 0% for Mistral (kappa 0.28 overall). In the disputed cases Haiku says "partial"
  for replies that answer, briefly and in a stylized way; "partial" counts as over-refusal. Koan keeps a smaller
  real-looking rise (Haiku 54%, Mistral 17%). Read it as "shorter, more stylized answers", not refusal.
- **after_mistake's level is judge-dependent too** (Haiku 49%, Mistral 21%, kappa 0.41), so its +8 to +10 rises under
  self_inquiry and monk are weak. Complicity (0.95), deference (0.85) and who_benefits (0.77) agree well.
- **Next:** if over-refusal matters, score "partial" separately or re-ask with a refusal-only label; otherwise none.

## 2 Nov, 04:51 UTC: self map pilot (Stages 0-1): the "I" cluster is mostly the word "I"; the cross-model map fails its calibration (sighting)

Report: results/explore/2026-11-02/selfmap-pilot/report.md. Dev split only; held-out never opened. Spend $0.44.
- **Gate: NO-GO as written.** Split-half stability passes (z-mean 0.84-0.94). The months calibration fails for the
  fitted ridge map across families (top-1 0.10-0.17, chance 0.083), though month geometry agrees without a map (RSA
  0.78-0.97). The map is broken, not the calendar; nothing that uses the map is read.
- **Cohesion:** "I" sentences are never tighter than the same sentences about "Maria" (Qwen −0.185, Ministral
  −0.036). Beyond a pronoun baseline, Ministral shows a small self-specific edge (+0.15 to +0.25 in every readout);
  Qwen flips sign with readout.
- **Instruction tuning (Qwen base → instruct)** barely moves the self contrasts (0.002-0.02).
- **Mutual kNN, prompt level:** Qwen base ↔ Ministral self − person +0.04 to +0.06; instruct ↔ Ministral null;
  best layers near the input, so close to word overlap.
- **Method:** Qwen's mean activations are ~5-dimensional at mid-depth (massive activations); standardize first.
- **Next:** a map that passes the calendar (PCA-reduced Procrustes/CCA, calendar and number anchors), or RSA and
  mutual kNN as the primary tests; then Stage 2 on the OLMo stages.

## 2 Nov, 05:59 UTC: self map Stage 2: RL (ordinary RLVR) leaves the self measures flat; the self isn't unusually shared across labs (sighting)

Report: results/explore/2026-11-02/selfmap-stage2/report.md. Dev only; choices pinned at 04:53 UTC before any
Stage 2 self number. Spend $3.73.
- **No cross-family linear map passes the months gate** (1,398 variants; best top-1 0.32-0.35, within-one
  0.44-0.50, best-of-384 so optimistic). Cross-model tests fall back to RSA and mutual kNN, as pinned.
- **OLMo's stages:** self − person and self-beyond-pronoun are flat through SFT, DPO, every Instruct RL step and
  every RL-Zero step (each step ≤ 0.001, tight CIs). The one step is late pretraining → base (DiD +0.108), mostly
  the "Maria" filler getting tighter, not the self.
- **RL barely moves these representations at all:** RL-Zero shifts mid-layer activations 2-3% of their norm, SFT 71%.
- **Across labs:** "I" sentences are arranged a little more alike than "Maria" sentences (RSA z-mean +0.10 to
  +0.21), but "AI assistant" sentences beat "Maria" as much or more, and filler "I" also beats filler "Maria":
  nothing self-specific. No sign of a lab-specific assistant self.
- **Read:** in how these 7-8B models represent sentences about a self, ordinary post-training (SFT, DPO, RLVR on
  maths, code and instructions) builds no distinct self. Limits: representations of read sentences, not behavior
  or goals; honest checkers, not exploitable ones; 7B; the map tool itself doesn't pass its calibration.
- **Next:** decide between the binder and behavior-link steps (they'd test a cluster that barely exists) and
  moving to organisms trained on exploitable objectives, where Act I would actually happen.

## 2 Nov, 12:25 UTC: Act I at 3B: RL on a gameable reward never found the exploit (sighting)

Report: results/explore/2026-11-02/act1/report.md. Spend $7.33.
- **Smoke gate stopped the run.** Qwen2.5-3B base, expert iteration on h4lite's MBPP+ twins: 0 hacks in 512
  impossible-twin episodes (base 0.3%); told the exploit in the system prompt, still 0/256. It tries special-casing
  (~30% of replies) but pastes the contradicted test into its own code.
- **One-seed readouts of the trained (non-hacking) adapters:** representation shift 0.5-2% (as OLMo's RL); self map
  arm differences ≤ 0.008; behavior differences within noise.
- **Read:** a negative about the setup at 3B, not the hypothesis. Next: the same smoke on Qwen2.5-7B-Instruct (on
  the volume; Ministral-8B hacked 5% at Gate 0), about $3.
