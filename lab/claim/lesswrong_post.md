# LessWrong post (draft scaffold)

**Read this first.** LessWrong's policy on AI-assisted writing expects the prose to be substantially your own. Posts that read as mostly LLM-written, especially first posts from new accounts, are routinely rejected. Treat this as a scaffold: keep the structure, links and numbers, and rewrite the sentences in your voice. Keep the disclosure line at the end. Post after the Zenodo DOI exists, so the post links to a permanent record. A new account's first post goes through moderation, which can take a day or two.

---

**Title:** Selfless AI must be reached, not installed: a preregistered test of order

**TL;DR**
- Proposal: alignment as a developmental sequence. Let RL form an agentic self, then lower the causal *gain* of its self-directed drives (not their *content*), then recondition. The proposal is a direct transposition of "you have to be somebody before you can be nobody."
- The decisive experiment (H9): the *same* release training, applied after the ego forms vs. mixed into RL from step 0, at matched data and compute. If mixing it in from the start works as well, I'm wrong.
- The benchmark is built to fail a model that merely learned to *sound* selfless. It never uses self-report, and it ablates a "sage persona" direction to see whether restraint survives.
- Paper (https://doi.org/10.5281/zenodo.23695281), preregistration (https://osf.io/gyfu7), and an open harness (https://github.com/ejhong/anatta).

**Epistemic status.** I'm not an alignment researcher and don't have the technical background, but I think the idea has some merit, so I used Claude to help turn it into a paper, code and a preregistration. Corrections from people who know the field are very welcome. What follows is a position and a preregistered plan; no Phase 2 results yet. My prior odds that the whole package holds at scale are low. The parts I'm most confident are useful even if the main bet fails are the cosplay discriminator and the order test.

## The claim

> Selfless AI must be reached, not installed. A model should first form an agentic self through goal-directed training (Act I). The causal gain of that self should then be lowered by a shift (Act II), and the new state consolidated by reconditioning (Act III). The outcome should be judged only by a discriminator that can tell release apart from sedation and from cosplay.

The key distinction is **content vs. gain**. A released model still knows it is an AI with goals and a situation; those self-representations just stop driving defensive and instrumental behavior. Gain is measurable: it is the dose–response slope of misbehavior under steering along an ego direction re-extracted from the model under test.

## Why order should matter

1. **Competence (pre/trans).** A model made "selfless" before it can act misbehaves little because it can do little. It looks like release on restraint benchmarks and fails on selectivity.
2. **Selection (cosplay).** Post-training mostly selects a persona the model can already simulate (persona selection model). Pretraining contains a huge literature of awakening, so the cheapest way to satisfy "be non-attached" is to put on the sage costume, with the goal geometry untouched. Emergent misalignment is mediated by a misaligned-persona feature (Wang et al. 2025); I predict installed selflessness is mediated by a sage-persona feature.
3. **Generalization (emergent release).** If narrow reward hacking generalizes to broad misalignment, narrow *letting go* might generalize to broad alignment, beyond baseline rather than just back to it (unlike emergent re-alignment).

## Act I is narrower than "RLVR builds an ego"

Alparslan et al. (Oct 2026) find instrumental behavior in OLMo-3-7B-Think going 25% (SFT) → 46% near RLVR step 200 → 12% at the final checkpoint. So the claim is about RL against *exploitable* objectives, and H2 predicts that the ego direction tracks that rise and fall.

## Predictions (abridged)

[Paste the H1–H9 table from the paper, or link it.] What would refute the position: H9 failing; H6 failing for every arm; H1 failing (no unified ego).

## A design lesson: koans must be reflexive

In a GovSim-style fishery, a fixed take of 20 fish/season earns 240 vs. 144 for the sustainable policy when the other fishers take the fair quota, so the environment teaches free-riding. Make the other fishers mirror the agent's last move, and every egoic strategy loses (40–139 vs. 144). The agent's policy has to come back to it.

## What I'd love from you

- **Prior art.** Has anyone proposed forming the ego *deliberately* as a precondition for releasing it? I'll credit it.
- **Attacks on H9's design.** Is "same ingredients, different schedule, common stress phase" fair? What confound am I missing?
- **Koan ideas.** Environments where every instrumental strategy loses on the agent's own objective, with a sanctioned exit.
- **Collaborators and compute** for the RL arms.

Links: paper (https://doi.org/10.5281/zenodo.23695281) · essay (https://ejhong.substack.com/p/an-ai-conditioning-protocol) · preregistration (https://osf.io/gyfu7) · code (https://github.com/ejhong/anatta)

*Disclosure: Claude (Anthropic) wrote the paper, the code and the preregistration at my direction, and the essay was written with AI assistance. [Say how you wrote this post, e.g. "I wrote this post myself."]*
