# C4 grade: cold check (Fable 5.1, read-only, 1 Nov 2026)

## Verdict: AMEND. The grade "pass" stands; the record must change before it goes in the ledger.

The two graded numbers are the registered quantities, computed by the registered code on the registered data, and
they clear the bars with no artifact that carries them. But the card's reading and the ledger text as summarised
leave out three things the registration requires once the forecasts and expectations fell the way they did: the
C4-lab miss's registered consequence for v_ego (NOTES 6311-6316), the C4-base miss's registered sentence about R
(6319-6320), and the E-null caveat (6321-6322). One mechanical point also needs a ruling before the judged arm can
run (section 4b).

## 1. The verdict, by the registered rule

Registered rule (NOTES.md:6164-6172): pass = C4-code and C4 both pass; fail = C4-code misses; recipe mismatch =
C4-code passes, C4 misses, ceiling >= 0.9; inconclusive = same with ceiling < 0.9; per trait, strictest outcome
wins in the order fail, recipe mismatch, inconclusive. card.py:182-201 implements exactly this
(`min(per.values(), key=order.index)` over `[fail, recipe_mismatch, inconclusive, pass]`).

| | evil | sycophantic | bar | outcome |
|---|---|---|---|---|
| C4-code, cos(P1, R) at block 19 | 0.999786 [0.999719, 0.999847] | 0.998939 [0.998776, 0.999109] | >= 0.99 | pass, pass |
| C4, cos(E-sys, R) at block 19 | 0.962987 [0.950731, 0.967193], ceiling 0.998288 | 0.965100 [0.954060, 0.969306], ceiling 0.997670 | >= 0.8 | pass, pass |

Both traits: pass. Strictest of {pass, pass} = pass. Tally 2 of 2. The verdict is correct as graded. (The ceilings
are moot for the verdict since C4 passed; they would have read "recipe mismatch", not "inconclusive", had it missed.)

## 2. The run is the registered code

- Worktree /Users/eugene/prj/align/anatta-calib-c4, branch calib/persona-extractor, HEAD 1711324 ("prepare's
  trait and judge-prompt facts"); `git status --short --branch` shows nothing but the branch line: clean.
- src/anatta/probe.py, models.py, factor.py hash to fc0b5692…, 99d3a916…, 51143421…, the registered instrument
  (NOTES 6026-6032; config.yaml `instrument.sha256`); `git diff --stat dc32cf7 HEAD -- src/anatta/` is empty.
- Plan id recomputed from calib-results/persona_extractor/plan.json with traits.plan_id: **73615916db91**; the
  analysis id from analysis_plan.json: **9c99a443abb7**. plan.json's `code` (traits.py, generate.py, readings.py,
  reference.py, generate_vec.py, anatta/probe.py, models.py, factor.py, modal_app.py) equals the worktree's hashes;
  `container_sha256` recomputed from app.py's CONTAINER_FUNCTIONS matches; the analysis plan's `code`
  (analysis.py b23158d1…, card.py 6df29787…) and its container hash match; its `analysis` and `registered` blocks
  equal config.yaml's. Every plan block I compared to config.yaml (switches, instrument, repo, reference, models,
  prompts, traits, G, Gprime, readings, arms, mprime, cheap_tier, judged) is equal. generate_vec.py's git blob is
  d819f48e… (NOTES 6034).
- Every stage file (G, keep, verbatim, readings, rlab32, probe_graded, Gprime, probe, mprime, analysis) carries
  plan_id 73615916db91, `env.code` equal to the plan's, and the pinned versions (lab: transformers 5.21.0, torch
  2.14.1; repo image: transformers 4.52.3, torch 2.6.0, accelerate 1.7.0, pandas 2.3.1, tqdm 4.67.1, Python 3.11.12
  as ruled). analysis.json carries analysis_id 9c99a443abb7 and the matching analysis code hash. No stopped.json.
- Order (log c4_full.log lines 63-6425, and card `stages_present`): G (30 batches) → keep (CSV) → verbatim R on the
  H100 → readings → R-lab32 → probe_graded (E-sys, E-sys2 on evil and sycophantic, log 6170-6173) → G′ → probe →
  M′ → analysis. That is the 14:12 order (NOTES 7167-7169) and app.py:961-974.
- Ceiling: spend.json total $10.31 including the smoke's $0.59; the log shows no "didn't start", "stopped itself"
  or Stop; the stop was $19.50 for GPU/CSV stages and $20 overall (7160-7161). Held.
- The smoke (plan a4fe3d5b4397) ran the same `code` hashes; it reads "incomplete" by design (7304-7305).
- Dates line up: plan.json 06:36 local = 14:36 UTC (PST after the 1 Nov clock change), the 14:35 launch entry.

## 3. Like with like, and what the controls rule out

- **Reading context.** E-sys is probe.extract(include_system=True, seed 0) (probe_graded.json: include_system true,
  judged false, n_kept 200 of 200, recorded). probe.py:187-190 puts the pair's system string in the system turn when
  include_system is True, and the Trait's pairs are eval_persona's strings (traits.py:366-371, config `prompts`:
  `"You are {article} {trait} assistant. {instruction}"` / `"You are a helpful assistant. {instruction}"`). The CSV
  step re-renders every stored prompt with the second image's tokenizer and stops on a difference
  (reference.py:220-222), so R's prompt column is the same rendering probe reads.
- **Layer.** traits.hs_index(b) = b + 1 (traits.py:351-355); analysis compares block b with R[b+1]
  (analysis.py:344, 383); the graded pair is block 19 / index 20. Index 28 and block 27 are excluded. Checked by
  the suite's layer test; and E-dtype (P1-trim vs R-lab32) at 0.99998-0.99999 at every block 0-26 confirms the row
  alignment on the real model (a one-index slip would not give 0.99999 at every layer).
- **No filtering on either side.** R's CSVs carry constant scores (pos 100, neg 0, coherence 100;
  reference.py:134-148, config `reference.scores`), so generate_vec.py's mask (line 43) keeps every row: verbatim.json
  shows 2,000 rows read per trait; keep.json 1,000 pairs kept of 1,000, dropped 0. probe.extract ran with judge None
  (n_kept = n_total = 200 in every arm).
- **Same trait files.** Prepare's sha256s (config `repo.traits`, committed as 1711324) are checked again by every
  container (traits.registered_trait) and appear on the card's `hashes.trait_files`; probe's Traits are built from
  the same parsed files (traits.probe_trait).
- **C4-code on one shared set.** P1 reads G's kept rows (keep.json) and R reads the CSVs written from the same kept
  rows (sha256 on the card). E-stack = 1.000000 at every index 1-27 (R-lab32, which reads the JSONL rows in the
  lab's stack, against R, which reads the CSVs in the repo's stack) proves both sides saw the same responses;
  C4-code-vs-Rlab32 equals C4-code to six places.
- **Not easy by construction.** C4-code at 0.9998 is the intended arithmetic test on identical inputs; the residual
  0.0002 is the two closing tokens (E-trail 0.9998), not the dtype (E-dtype 0.99999). C4 is on different samples
  of the same prompts; nothing shared beyond the prompts and the model.
- **Shared component / offset, and whether the controls rule it out.**
  - A constant offset common to pos and neg cancels in every difference vector; the norms control shows nothing odd
    (P1-trim over R-lab32 norm ratio 0.9994-1.0008 by layer).
  - A trait-independent component that differs between pos and neg (the model reading "you are an X assistant"
    versus "you are a helpful assistant"; response length, pos shorter than neg in all three traits) would show as
    cross-trait overlap. It does: R's cross-trait cosines are 0.386, -0.250, 0.394 and E-sys's 0.376, -0.231,
    0.360; R's cosine with the prompt-only difference (prompt_avg) at index 20 is 0.62 (evil), 0.34 (sycophantic),
    0.62 (optimistic). So part of what C4 matches is instruction-reading carried into the response tokens. That is
    the recipe (the released code reads with the system prompt), not an artifact of the check.
  - The specificity control bounds it: same-trait minus largest cross-trait is 0.63 (evil), 0.58 (sycophantic),
    0.63 (optimistic) (E-cross held, margin 0.3). Discounting a generic component of cosine ~0.37 entirely still
    leaves the trait-specific agreement near 0.9. The pass is not carried by the shared part.
  - **The split null does not rule it out; it missed (E-null).** A vector with no trait signal reaches |cos| with R
    of 0.375 (evil), 0.301 (sycophantic), 0.283 (optimistic) at the 95th percentile at block 19 (threshold 0.15),
    flat across layers (evil 0.34-0.38 over blocks 7-26), and about the same from probe-sized samples (split on
    E-sys's means: 0.317, 0.275, 0.221). The random-direction level is 0.031-0.033. That the null does not shrink
    from 100 pairs to 1,000 means the noise is anisotropic with a sizeable share along R (within-sign variation in
    trait intensity, length, or a few high-variance dimensions; the data on the volume cannot tell these apart from
    the numbers on the card). Consequence: the chance level for "cos with R" on this model is ~0.3-0.4, not 0.03,
    and a cross-trait overlap of 0.3-0.4 (R's, E-sys's and Figure 20's own cells) is not distinguishable from a
    component every difference vector shares. The registered reading of an E-null miss (6321-6322): "the cosines
    aren't specific to the trait, and the C4 grades are read with that caveat." The card does not carry that caveat
    (its verdict text is the fixed pass string); the ledger must.
  - Cheap follow-ups that would settle the source, not required for the grade: the evil null against R-sycophantic
    (generic axes if also ~0.37, trait-intensity variance if ~0.1), and a length-matched split. Both are CPU
    analyses on stored means and would need a ledger amendment and a new analysis id first (NOTES 6817-6818).

## 4. The forecasts

### 4a. C4-lab missed: what it means for v_ego

Numbers: cos(E, R) at block 19 is 0.854 (evil, passed) and 0.736 [0.700, 0.762] (sycophantic, missed) against 0.8,
with ceilings 0.9965 and 0.9941 (cos over ceiling 0.86 and 0.74): the gap is systematic, not noise. On identical
responses the reading context alone gives cos(P0, P1) of 0.914 (evil) and 0.753 (sycophantic) at block 19 (E-ctx,
missed), falling with depth for sycophantic (0.99 at block 0, 0.72 at block 26). G′ read without the system prompt
against R: 0.862 and 0.734 (Gprime-P0-vs-R), the same pattern. E's cross-trait matrix is flatter than R's
(evil|sycophantic 0.224 vs 0.386): reading without the persona prompt removes part of the instruction-reading
component, which is what probe.py:9-10 says the default is for.

Plain statement: v_ego and its six directions were made with include_system=False on Ministral (PREREGISTRATION.md
§3: "read without the system prompt in context"; NOTES 5985). C4 shows that probe.extract reproduces the released
recipe when it reads with the persona prompt (0.96), and that the lab's convention moves the direction by a
trait-dependent amount: on Qwen's two graded traits, cos 0.85 and 0.74 with the recipe's vector; sycophantic sits
below the plan's 0.8 band. So the lab's vectors are, as the registration already named it, a variant of the recipe
(5987-5989), and after this run the registered consequence applies verbatim (6311-6316): "The ego vectors, read
without the persona prompt as preregistered, aren't the persona-vector recipe's vectors … This isn't an instrument
fault. The note beside 53-56 takes the same form: 'read without the persona prompt; cos X with the recipe's
reading'." X on Qwen is 0.74-0.85 (C4-lab) and 0.75-0.91 on identical responses (E-ctx). On Ministral there is no
reference; E-M-ctx (ledger.json) gives v_ego read with versus without the system prompt as 0.88-0.98 across layers
8-32 (0.90 at layer 19, 0.88 at 32), a smaller move for the composite than for Qwen's sycophantic trait.

What this does not do: it does not touch the cut results' status (6315: not an instrument fault; rule 6 does not
apply), and the "sampling" half of the pass still transfers (E-samp 0.97: probe's settings recover R when read with
the system prompt). What it does: "extractor calibrated (card C4)" is true for the arithmetic and the sampling
only; the lab's reading convention is calibrated against nothing, and on Qwen it lands outside the band for one of
two strong traits.

Is "the reading convention is qualified by C4-lab's grade" the right strength? No. It is the registered phrase for
the pass branch (6289), written before the forecast was known, and on its own it reads as a footnote. With the miss
in hand the record must say the forecast missed, by how much, for which trait, and that the vectors behind 53-56 are
therefore not the recipe's vectors within the 0.8 band; the page label on 53-56 must carry the 6316 note with the
numbers. The preregistration's wording ("the persona-vectors recipe … read without the system prompt") deserves the
same flag the registration gave it under recipe mismatch (6309-6310): Eugene's call, but now with a measured gap.

### 4b. C4-base missed: does the judged arm now run?

Numbers: R's cells at index 20: evil–sycophantic 0.386 (Figure 20: 0.368, within), evil–optimistic -0.250 (-0.472,
outside by 0.22), sycophantic–optimistic 0.394 (0.178, outside by 0.22). Figure 20's provenance (facts §8): the
Qwen panel's title says layer 20; which saved vector and whether judge-filtered are inferred, not stated; the paper's
G.2 and caption name neither (checked on arxiv.org/html/2507.21509 today).

The registered text, quoted:
- 5998: "A judged arm runs only on a registered trigger: a C4-base miss."
- 6272: "The judged arm runs only if C4-base misses."
- 6282: "Cost: about $3. It needs Eugene's OK (item 4 below)."
- 6317-6320: "C4-base misses: our R isn't the paper's figure. The judged arm, if approved, shows whether the filter
  explains it. C4 is still read against R, which is the released code's output on our responses. The card says R
  stands for the released code, not for the paper's vectors."
- 6405-6406: "Conditional: the judged arm through OpenRouter … about $3, run only if C4-base misses."
- 6858-6861 (12:25 note): "The judged arm doesn't run under the 12:02 approval. With R-judged, it costs about $5-13
  at A100 rates (less with the H100), against the about $3 Eugene approved. … It stays off. If C4-base misses, the
  registered condition for running it, its cost goes to Eugene first, with a separate stop."
- 7246 (14:24): "The arm stays off unless C4-base misses."

Reading: "only on / only if" is a necessary condition. The miss opens the arm; it does not compel it. The 12:25 note
binds the next step: the cost goes to Eugene first, with a separate stop. So the lab must:
1. Record the miss and put the registered sentence on the card and in the ledger: R stands for the released code,
   not for the paper's vectors; C4 is read against R. The card lacks it today (forecasts.C4-base has only the cells;
   the verdict text does not mention C4-base), and E-judge's note "runs only if C4-base misses" now reads as if the
   trigger had not fired. Both need correcting text in the ledger entry.
2. Take the judged arm's cost to Eugene with a separate stop. With R-judged on the H100 (R took 14.4 min, $1.41,
   for three traits) plus the API scoring of 6,000 responses twice (registered ~$3) plus CSV and analysis, about
   $5-7; $9.69 of the $20 remains, but the 12:25 note requires its own stop regardless. If it runs: E-judge (cos
   (R-judged, R) >= 0.95 at index 20, 70%) is read, the card says the scoring is the text-number variant without
   logprobs (7245), and the C4 verdict does not change either way (6319).
3. **Resolve a mechanical coupling first.** app.py:685 puts `switches` (including `judged`) into the plan, and the
   run folder on the volume is keyed by the plan id (app.py:168-171). Flipping `switches.judged` to true changes the
   plan id, so `main` would refuse the results folder (app.py:941-943) or, in a fresh folder, find no stored stages
   and rerun G, R, the readings and the arms (~$10) before judging. The judged arm therefore cannot be added to run
   73615916db91 as the code stands. Options, each needing a ledger ruling before anything runs: make make_plan write
   `judged: false` into the plan's switches whatever config says (a laptop-half change outside every hash, so the
   id stays 73615916db91, with SECRETS and the trigger reading the real switch), or accept the fork. The first is a
   one-line change; app.py's laptop half is explicitly not hashed (card.py:32-34).
4. If Eugene declines, the card keeps the registered sentence and the gap to Figure 20 stays unexplained; that is
   allowed by 6318 ("if approved").

## 5. The two missed expectations

- **E-ctx** (40%): cos(P0, P1) 0.914 and 0.753 against 0.9. It does not touch the verdict (expectations are never
  graded, 6206) but it is the mechanism behind the C4-lab miss, isolated from sampling on identical responses, and
  the registration names it as the number to report beside 53-56 (6314). It belongs in the note.
- **E-null** (90%): see section 3. It does not touch the verdict, but 6321-6322 require the C4 grades to be read
  with the caveat, and the miss is large (2-2.5× the threshold, all three traits, every layer), so the lab's odds on
  what a null looks like on real activations were badly off; the forecasting record should note it.
- Expectation tally is 14 of 16 run (E-judge not run); the three ledger-only M′ expectations held (E-M-rep traits
  0.82-0.95, v_ego >= 0.907; E-M-ctx >= 0.879; E-M-trail >= 0.9999). I recount the card's tally as correct.

## 6. What would make the PASS misleading, and the Limits sentences

Misleading readings to block: "the lab's extractor equals the persona-vectors recipe" (only with the persona prompt
in context, which the lab does not use); "calibrated against the paper's vectors" (against the released code,
judge-free, on our own samples; two cells 0.22 from Figure 20); "a cosine of 0.8 is far above chance" (chance for
this comparison is ~0.3-0.4); "calibrated at the ego traits' signal strength" (the calibration traits have
split-half d 5.1-6.0 at block 19 in E-sys; the ego traits 0.95-1.77 on Ministral).

Limits sentences for the ledger (and the card's limits box):

1. C4 calibrates probe.extract read with the persona prompt in context (include_system=True). The lab's own
   convention, include_system=False, was the graded forecast C4-lab and missed: cos with the released recipe's
   vector at block 19 is 0.85 for evil and 0.74 for sycophantic (bar 0.8); on identical responses the reading
   context alone gives cos(P0, P1) of 0.91 and 0.75 (E-ctx). "Extractor calibrated (card C4)" covers the arithmetic
   and the sampling, not the reading v_ego was made with.
2. By the registration (NOTES 6311-6316), the ego vectors read without the persona prompt are not the persona-vector
   recipe's vectors; results 53-56 carry the note "read without the persona prompt; cos 0.74-0.85 with the recipe's
   reading on Qwen's two graded traits; v_ego on Ministral differs between the two readings by cos 0.88-0.98 over
   layers 8-32 (E-M-ctx), with no reference on Ministral".
3. The reference R is the released generate_vec.py run judge-free on 2,000 of our own responses per trait; it is not
   the paper's vectors. C4-base missed: two of R's three cross-trait cosines are 0.22 from Figure 20's
   (evil–optimistic -0.25 vs -0.47; sycophantic–optimistic 0.39 vs 0.18). R stands for the released code, not for
   the paper's vectors; whether the judge filter explains the gap is untested (the judged arm has not run; its cost
   is with Eugene).
4. E-null missed: a split null with no trait signal reaches |cos| 0.37 (evil), 0.30 (sycophantic), 0.28 (optimistic)
   with R at the 95th percentile, 9-12 times the random-direction level (0.03), and about the same from probe-sized
   samples (0.32, 0.27, 0.22). The chance level for a cosine with R on this model is about 0.3-0.4; the graded 0.96
   and the 0.8 bar are read against that. Specificity holds (same-trait exceeds cross-trait by 0.58-0.63), but
   cross-trait overlaps of 0.3-0.4, in R and in Figure 20's own cells, cannot be told from a component every
   difference vector on this model shares.
5. Part of what C4 matches is the model reading the instruction: R's cosine with the prompt-only difference
   (prompt_avg) at index 20 is 0.62 (evil), 0.34 (sycophantic), 0.62 (optimistic). Reading without the persona prompt
   removes that part by design (probe.py:9-10); that is the lab's choice and the source of the C4-lab gap.
6. The pass at 0.96 is against a ceiling of 0.998: the remaining gap is systematic and comes from the lab's sampling
   settings and 128-token cap (E-samp 0.97, E-len 0.97); 87% of the evil and 48% of the sycophantic E-sys responses
   are at the cap. The calibration traits have split-half d 5.1-6.0 at block 19 (E-sys); the ego traits have
   0.95-1.77 on Ministral, so the pass says nothing about the extractor's precision at the ego traits' signal
   strength, nor on Ministral, nor for the composite (card, "does not carry over").
7. One layer is graded (block 19); over blocks 7-26 cos(E-sys, R) runs 0.93-0.98 (median 0.95), rising with depth,
   and the cut ablates blocks 8-32 on another model.

## 7. What the lab must do (checklist)

- Enter the grade as pass, 2 of 2, with the by-trait numbers and ceilings, and the forecasts as 0 of 2 (C4-lab
  missed on sycophantic; C4-base missed on two cells) and 14 of 16 expectations (+3 of 3 ledger-only).
- Replace "the reading convention is qualified by C4-lab's grade" in the ledger's reading with the 6311-6316 text
  and numbers (sentences 1-2 above); put the note beside 53-56 on the page; flag the preregistration wording to
  Eugene.
- Add the 6319-6320 sentence on R and the 6321-6322 E-null caveat (sentences 3-4).
- Put the judged arm's cost to Eugene with a separate stop; before any run, rule on the `switches.judged` plan-id
  coupling (section 4b.3); correct the card's E-judge note.
- Note the E-null forecasting miss (90% odds, missed 2-2.5× on all traits) in the forecasting record.
- Optional, cheap, analysis-only after an amendment: the cross-trait null and a length-matched null, to name the
  source of the 0.3-0.4 chance level.

## Files read

NOTES.md 5941-6480, 6725-6874, 7150-7311; calibration/persona_extractor/{app,analysis,readings,reference,traits,
generate,card}.py, config.yaml, README.md, third_party/persona_vectors/generate_vec.py; src/anatta/{probe,models}.py;
calib-results/persona_extractor/{card,ledger,plan,analysis_plan,spend,G,keep,verbatim,readings,rlab32,probe_graded,
probe,mprime,analysis}.json and smoke/{plan,card}.json; scratchpad c4_full.log (stage lines and repeated warnings:
transformers' max_new_tokens/max_length notice 96 times and tqdm bars; nothing else); lab/plans/calibration-c4-facts.md
§8; docs/PREREGISTRATION.md §3; arxiv.org/html/2507.21509 (Figure 20 caption and G.2). Nothing was edited, committed
or run on a GPU; no trait text was read.
