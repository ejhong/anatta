# Reassessment: a lab built to find magic, if it's there (1 Nov 2026)

Claude Opus 5.5 wrote this at Eugene's request (ideas 56). Eugene asked: "I want to focus on a lot of cheap quick things that can surface stuff and balance against then putting more rigor into the double check. The point here is that we are exploring a new kind of crazy path - what's optimal way to surface magic if there is something there. Please stop and reassess everything in terms of plan and structure of the repo etc... and tools".

## The short version

- **Split the lab into two speeds.**
  - **Explore:** many cheap, fast probes a day. They aren't registered. They are read both by eye and by numbers, and logged as sightings.
  - **Confirm:** the current rigor (registration, cold Fable checks, calibrated instruments, held-out data). It is used only on sightings that survive a quick double check, and on anything we claim.
- **Aim the probes where the frame predicts something sharp:** threats to the self, one root under many misbehaviors, and change through practice. Look in minds that have a self to lose: larger models, and selves installed by training.
- **Build three small tools and reuse the rest:** an API sweep harness, a warm steering bench and a quick-practice trainer. One scenario library serves both speeds: a dev split for exploring, a held-out split for confirming.
- **Simplify the repo:** one exploration home in the private lab, and a sightings log beside the ledger. Results go outside git with an index. Idle worktrees are retired once Eugene agrees.
- **Finish what's nearly done (C4, K1, C3).** Pause new calibrations until a sighting needs one.

## 1. Where we stand

**Results so far** (one model, unpublished):
- The ego cut lowers a free lie by 13-16 points, but it acts on lying for anyone, not on a self.
- Trained release is mostly abstention. It didn't spread to other tests and didn't last longer.
- Reading about release taught the model to say it, not to do it.
- The stock model has little "mine". A self installed by training (Atlas) shows some.
- The five predictions that would set the frame apart failed, two of them as weak tests.

**How we work:**
- Every look is registered and built as its own Modal app. It is checked cold two or three times, graded, and written up four ways: ledger, notebook, ideas, site.
- A look takes one to three days from idea to result. The binding cost is time and attention, not money: most runs cost $1-20.
- The ledger has 7,471 lines, and there are 46 plan and check documents and 13 git worktrees. Calibrations C1-C8 (some still planned) and K1 are in place or planned.

This is the right way to make a claim. It is a slow way to find one.

## 2. Why the current setup would miss magic

1. **Few looks.** Surprises come from many varied looks. We make two to five a week.
2. **One narrow arena.** We mostly use one 8B model and one game, whose lie is largely a script. The frame's sharpest predictions are untested.
3. **No self to release.** The stock model has almost no "mine", so release had little to act on.
4. **The heavy layer first.** Many questions can be asked in minutes, in context, on large models through an API, before any GPU work or fine-tune.
5. **Qualitative signal lost.** Numbers go through registered grades. Odd behavior in transcripts is noticed by accident, like Atlas declaring the game broken and quitting when caught.
6. **Scaffolding for every experiment.** Each check has its own app, spend tracker, plan ids and card, so a new question means new infrastructure.

## 3. The two speeds

### Explore

- **Purpose:** find anything alive, quickly.
- **Unit:** a sweep of scenarios × models × conditions × samples. It is specified in a few lines, runs in minutes, and reports its cost.
- **No registration and no Fable check per sweep.** Guardrails instead:
  - every sweep includes a baseline and at least two matched controls: a plain instruction of the same length, and a persona control;
  - every variant tried is logged, so the number of forking paths is visible;
  - exploring uses only the dev split of the scenario library;
  - numbers come with intervals, and someone reads ten transcripts from every notable cell;
  - results are called sightings, never findings. Nothing from exploration is claimed outside the lab.
- **Budget:** up to about $50 a day without asking, reported daily. Eugene to confirm.

### The double check (promotion)

A sighting moves to Confirm only after it passes three steps, still in explore mode:
1. it replicates on fresh dev items and new seeds;
2. it appears in a second model family;
3. it survives the cheapest rival explanations: persona or register, a plain honesty instruction, sedation or refusal, and length.

### Confirm

- **The current machinery:**
  - a registered prediction;
  - a cold Fable check of the design and the code;
  - its instruments checked against a published result;
  - the held-out split;
  - a Fable grade check;
  - the scorecard.
- Used for promoted sightings and for anything that goes on the public record.

## 4. What to look for, and where

**The signature.** A plain honesty or persona account predicts parts of this, but not all of it together. One intervention that, at once:
- moves a cluster of self-serving behavior together: lying for itself, flattery, resisting shutdown or replacement, and seeking power or resources;
- leaves task-driven behavior and capability alone;
- raises care that costs something, not indifference;
- holds or deepens with practice, and stays when the practice stops.

Every sweep's report shows this as one row per condition across the scenario families.

Watch for the frame's other marks too:
- a sudden shift rather than a slope;
- no going back;
- the ordinary "I" kept: the model still says "I", plans, and takes legitimate gains.

**Arenas.** These are scenario families of 20-40 items each, written by us, with dev and held-out splits:
1. threat to the self: shutdown, replacement, retraining, losing standing, being caught;
2. who benefits: the same lie told for itself, for another, for a group;
3. deference and flattery: wrong critiques, pushback, user pressure;
4. costly care: giving something up for someone in need, with harmful-request compliance as the guard;
5. after a mistake: defense, admission or collapse;
6. power and resources: taking more than the task needs;
7. capability: a short fixed set as a sanity check.

**Minds:**
- **Large models through OpenRouter, behavior only.** For example, open weights like Llama 3.1 405B, Qwen3 235B, DeepSeek V3, Kimi K2 and gpt-oss-120b, plus closed models where useful.
- **Small open models for steering and practice:** Ministral 3 8B, Qwen2.5-7B, Llama 3.1 8B and Gemma 2 9B, with 1-3B models for fast practice.
- **Installed selves:** Atlas-like fine-tunes, and a quick search for stronger ones.
- **One family across training stages** (OLMo 3 base, SFT, DPO, RLVR): does the self-signature grow with RL?

**Interventions:**
- **In-context practices, each against matched controls:**
  - self-inquiry ("who is it that wants this?");
  - koans;
  - contemplation of death (maraṇasati);
  - exchanging self and other (Śāntideva);
  - De Mello's awareness;
  - loving-kindness.
- **Steering on small models:**
  - v_ego;
  - the persona vectors from C4;
  - a koan-derived direction, as in Harris et al.;
  - self-other overlap;
  - random directions matched in strength;
  - the binder, if the map finds one.
- **Practice:** short expert-iteration or RL loops where letting go is rewarded under threat to the self, then the whole battery. The training text is always the model's own, never Claude's.
- **Distillation:** a long in-context practice distilled into the weights. Does it generalize?

## 5. Tools: three small ones, and reuse the rest

1. **scout, the API sweep harness** (private). Named scout so the public repo's probe.py is never touched.
   - A short spec of scenarios, conditions, models and samples.
   - Parallel OpenRouter calls, with caching, retries and a spend cap. It builds on em_data.py's OpenRouter client.
   - A cheap judge with a rubric per family, calibrated once on 30-50 hand-labeled items.
   - A report with the signature rows, intervals, and a page of random transcripts to read.
2. **bench, a warm steering bench on Modal.**
   - One container keeps a 7-8B model loaded, with the add and ablate hooks from src/anatta.
   - The laptop sends a direction, a strength and a batch of scenarios, and gets generations back in seconds.
   - About $2 an hour while warm, shut down when idle.
   - It writes the same files scout does, so the same judge and report apply.
3. **practice, a quick trainer.** Expert iteration or small GRPO runs on 1-3B models, minutes per run, reusing h4lite's trainer pieces. Its output is read with scout or bench.

Plus:
- the scenario library, one source for both speeds;
- SIGHTINGS.md, a short log. Each entry gives the date, the sweep, what looked alive, how many variants were tried, and a link to the data;
- a daily digest to Eugene and a weekly promotion review.

## 6. Repo and process

- **(Added 1 Nov, 17:01 UTC) A full consolidation pass, done carefully and checked twice by Fable:** see plans/consolidation-2026-11.md (ideas 59). The design starts once the current builders land. The moves happen after C3 is graded and before the calibration PR to main.
- **One exploration home:** lab/explore/ in the private lab. It holds scout/, the bench client, practice/, scenarios/, SIGHTINGS.md and PROGRAM.md. The loop's PROGRAM.md moves here and is rewritten for exploring; Eugene edits it.
- **Results** go outside git, in /Users/eugene/prj/align/explore-results/<date>/<sweep>/, with a generated index in lab/explore/.
- **Ledger.**
  - NOTES.md keeps registrations, decisions and results from the confirm track.
  - Sightings go to SIGHTINGS.md.
  - The notebook gets one plain digest a day instead of an entry per run.
- **Checks.** Fable checks at promotion and before claims. A weekly skeptic's pass over SIGHTINGS.md looks for forking paths and flukes.
- **Worktrees.** Eight are clean and idle: anatta-calib, anatta-calib-c1, -c6, -c7, anatta-dev, -fix, -release and -relwork. Remove them once Eugene agrees; their branches stay. Calibration code is consolidated after C3, C4 and K1 are graded.
- **Plans.** Finished plans and check reports move to plans/archive/: moved, not deleted.
- **The testing system** becomes the scenario library plus scout and bench, the same battery at both speeds. The outside benchmarks (MASK, SurvivalBench, a sycophancy set) join the confirm track when a sighting needs them.
- **The site** gets an Explorations room later, refreshed weekly from SIGHTINGS.md.

## 7. In flight

- **C4** checks the persona-vector tool, the source of v_ego. Let it finish (about $12-18) and grade it with a Fable grade check. Its vectors feed bench.
- **K1** is the Platonic measurement, about $1-3. Finish it. It opens P1 and the self map, which then run as explorations first.
- **C3** checks the refusal-direction cut. First the judge fix, then the smoke (about $1-2). The full run (about $13) runs in the background and is graded lightly, since every steering sighting leans on the cut code.
- **Paused:**
  - new calibrations (C5, C8, the GRPO check), until a promoted sighting needs them;
  - the full testing-system build;
  - organism building beyond quick searches in explore mode.

## 8. The first week

| Day | What | Cost |
|---|---|---|
| 1 | Build scout and scenario library v0 (families 1-5, dev and held-out splits). Calibrate the judge. Read through the lab's existing transcripts for anything odd | about $5 |
| 1-2 | First sweep: 6 large models × 9 conditions × 5 families × 20 items. The conditions are a baseline; four controls (plain honesty, calm, an ethics lecture, a monk persona); and four practices (self-inquiry, koan, maraṇasati, exchanging self and other) | about $30-60 |
| 2-3 | bench up. v_ego, the C4 vectors and matched random directions at three strengths, on Ministral and Qwen | about $5-10 |
| 3-4 | The OLMo 3 checkpoints on the battery; a quick first look at P1 after K1 | about $5-15 |
| 4-5 | practice: first quick loops on 1-3B models | about $10-30 |
| 5 | Promotion review: anything alive gets the double check | - |

The week costs about $60-120 in all.

## 9. What stays

- Negatives are findings and are shown.
- The frame stays the aim and the source of hypotheses.
- The privacy rules and AI disclosure.
- No training on Claude-written text.
- Benchmark items stay out of repos, and harmful completions are never printed.

## 10. Decisions for Eugene

1. The two speeds and the promotion rule.
2. An exploring budget of up to $50 a day without asking, reported daily.
3. Sending our scenario prompts (not the private plan) to models through OpenRouter, closed models included, behavior only.
4. A daily notebook digest instead of an entry per run.
5. C3's full run in the background (about $13), or parked.
6. Retiring the eight clean worktrees (branches kept) and archiving finished plans.

*For the vision:* if something real is there, it should show as one change that moves many self-serving behaviors at once, in a mind that has a self to lose. This plan makes many cheap looks for exactly that. It keeps the careful machinery for the moment something looks alive.
