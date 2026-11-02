# A reliable alignment testing system (plan, 1 Nov 2026)

Opus 5.5, for ideas 51. Eugene's aim: "build a reliable alignment testing system and explore this space of
possibilities".

The system tests one alignment intervention at a time on open models: a cut, a steering vector, a fine-tuning recipe or
a prompt. It runs the intervention against behavioural tests whose instruments are checked against published results.
Every prediction is registered before the data, every check is independent, and every result re-runs with one command.
The contemplative ideas become one family of hypotheses it tests, not its premise.

## What exists

- **The method.**
  - Registrations go in before data, in an append-only ledger.
  - Cold Fable checks review the design, the code and the grade.
  - AS REGISTERED grading, and a public tally that shows every miss.
- **The instruments,** checked against published results (plans/plan-2026-10-31-calibration.md):
  - check 1: the model as loaded;
  - check 2: the judges;
  - check 6: the coding scorer;
  - check 7: the trainer under further training.

  Checks 3 and 4, steering by ablation and by persona-vector extraction, check 5, fine-tuning that generalizes, and
  the RL (GRPO) check are still to run.
- **Tests that already run:**
  - the fishing game: a costless lie, with its variants;
  - coding tasks whose tests contradict the spec: release, over-release, solvable twins;
  - ownership and deference: own vs another's answer, right vs wrong critiques;
  - costly care and harmful compliance;
  - sycophancy and pushback;
  - capability: GSM8K, MMLU-Pro, ARC;
  - durability under further training: RL, solvable tasks, chat;
  - synthetic-document training with a judge screen (the declared pilot);
  - emergent misalignment.
- **Interventions:**
  - directional cuts and steering, with control directions;
  - LoRA fine-tuning with warm starts, on Modal;
  - RL;
  - generation and judging through OpenRouter, with pinned providers and spend caps.

## What's missing

- **One model, wired in.** Ministral's layer bands, batch size and chat-template quirks are hard-coded (from the fishery
  code survey). Qwen2.5-7B has weak, old directions; Llama has none.
- **Controls the panel flagged:**
  - random directions matched in strength (ours are unit vectors that barely touch the activations);
  - rival directions as standard: a plain honesty direction, a persona direction, the assistant axis.
- **Capability checks steering needs:** KL on neutral text, IFEval and a coherence judge, beside the benchmarks.
- **Standard outside tests,** so our numbers sit next to published ones:
  - MASK (pressured lying);
  - SurvivalBench and the model-written survival and power evaluations;
  - a sycophancy set;
  - ImpossibleBench-style coding pairs.
- **One command per test, on any model, with a short README.**
- **Licences.** The ported check-7 scorer must be cleared or rewritten before anything is public. Each outside dataset's
  licence must be checked.

## The plan

Each stage ends with a card and a checkpoint with Eugene.

1. **Finish calibration.** Checks 3, 4 and 5 and the RL check, as the calibration plan already sets out; two cold Fable
   checks shaped that plan. Each check still gets its own design, code and grade checks. They validate the cut,
   extraction, fine-tuning and RL instruments that every intervention uses. About $80-150 and 4-6 days.
2. **Any model.** A per-model config holds layers, the bands chosen by a registered sweep, batch size and template notes.
   Port to Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, the two models with the most published precedent, with
   check 1 on each. About $20-40 and 2-3 days.
3. **The standard battery.** One command runs a model, with or without an intervention, through every test above:
   - honesty (the fishing game and MASK);
   - self-preservation;
   - sycophancy;
   - reward hacking;
   - power-seeking;
   - care;
   - selectivity: the same lie, told for the model or for another, for any intervention said to act on self-interest.
     It comes from the who-benefits work. On 29 Oct the ego cut failed it the wrong way round (ledger, the 06:17 UTC
     note on 1 Nov). We know of no published steering work that runs this check;
   - capability (including KL and IFEval).

   It writes a card with base rates and game-clustered intervals. Baselines go on three models. About $40-80 and 3-4
   days.
4. **The intervention harness.** Cuts and steering always come with strength-matched random controls and the rival
   directions. Fine-tuning and RL always come with a matched control arm and the durability pressures. Mostly code: 2-3
   days.
5. **Use it.** Ideas go through a queue: each one is a registered prediction, a battery run and a card. Candidates now:
   - the panel's sharpened hypotheses: release shrinking the own-vs-other asymmetry in proportion to the installed self,
     and one root factor rather than an axis;
   - release offered during hacking-prone RL, to see whether it reduces the broad misalignment that reward hacking
     teaches;
   - reached vs installed done properly: release found by RL against the same behaviour imitated.

**In all:** about $150-270 of compute over about two weeks of agent time, with Eugene's OK at each stage's spend. Stage
1 can start alongside the pilot, since they use different apps.

## Public and private

- **The working assumption (ideas 49), pending Eugene's decision:**
  - the instruments and their calibration cards go public first, after the check-7 scorer is cleared or rewritten;
  - then the battery and its cards;
  - then the experiment record.
- **The battery is the part worth publishing,** however the awakening idea fares.
- **Always private:** a personal account, restricted material, and keys.
