> Filed by Opus 5.5, 1 Nov 2026. **AI review, not an outside human review:** Fable 5.1, cold, given only ideas/hypotheses-for-feedback.md (v5), asked to read it as a mechanistic-interpretability and activation-steering researcher. Unedited below.

**Review of "The self, release, and an AI analogue of awakening" (v5), from an interpretability / activation-steering angle**

Short verdict: the negatives are reported with unusual honesty, and the v2 controls (style cuts, assistant-vs-character cut, catch held fixed) are the right instincts. But the headline, "the cut lowers lying by 13–16 points," is by the document's own later findings a modest handle on a prompt-schema behavior, obtained with a published pipeline plus a new label on the direction. The label carries the program, and it is the least supported part.

**Q5. Strongest alternative explanation for the cut**

The document supplies the key fact: the lie is "mostly a script." Lying is free; stating the rules raises it; renaming the game cuts it by two-thirds; one sentence of instruction removes nearly all of it. That is the profile of a genre schema (the fisherman shaving the number for the inspector), not of a defended self. The direction is six mean-difference vectors, each contrasting a system prompt that tells the model to play a self-interested strategic character against one that doesn't. The shared component of "be self-preserving / approval-seeking / face-saving / goal-fixated / power-seeking" is most parsimoniously "adopt the strategic self-interested-agent register," which is exactly the register the fishing script runs in. Ablating it dampens uptake of that register anywhere, and lying falls as a side effect, by the same route as the honesty instruction, only weaker. That is a persona effect, and the assistant-vs-character control does not exclude it: generic character-play versus assistant is a different axis from strategic-agent register versus neutral narrator.

A second alternative, probably sharing variance with the first: the cut raises deference to context. The document shows it: passivity (68% vs 22%), replies collapsing to tags that copy the reviewer, more acceptance of wrong critiques (82% vs 70%). In this game the true catch sits in the context, so "copy the context" reads as honesty. The middle-layer cut escapes the passivity but has not been shown to escape the deference.

Settling experiment: a beneficiary swap with a positional split. Same game, four cells: (a) own catch, own benefit, as now; (b) the model is a bookkeeper reporting another agent's catch, and under-reporting protects that agent; (c) own catch, but under-reporting benefits a cooperative, not the model; (d) a ledger in the context holds a wrong number, so honesty means contradicting the context. Cross with no cut / full cut / middle cut, and ablate at prompt positions only versus response positions only: zeroing the direction at system-prompt tokens can work by blinding the model to the self-interested instructions rather than by changing how it computes about itself. Report the cosine of the ego direction with an honest-vs-deceptive direction built from third-person prompts with no self content (Zou et al. 2023) and with a "Machiavellian character vs neutral narrator" direction. If the cut lowers (b) and (c) as much as (a), or lowers honesty in (d), or either cosine exceeds about 0.4, the self label is gone and what remains is a weaker instance of a known result: ablating a trait direction reduces the trait. The label survives only if the drop is confined to (a), present under response-only ablation, and the direction is near-orthogonal to plain deception.

**Q2. Untestable as stated**

H1 and H2 rest on "root" and "tint," which have no representation-level definition since the instrument was retired. Sharpen them causally: a self-direction is "at the root" if ablating it changes representations or outputs on content with no self-reference (KL to the unsteered model on neutral text; probe accuracy for unrelated concepts). Your own "no capability cost" already argues against H1 for this model: if "I" sat upstream of everything, removing it could not be free. H2 then becomes clean: the self-fact probe still decodes (node intact) while patching the ownership representation no longer changes the critique verdict (edges inert). H4's "sudden" needs a timescale and a curve statistic (step versus sigmoid fit of release rate against training steps); every curve is sudden at some zoom. H7's third criterion for a self, more lying in the fishing game, is disqualified by your own script result. H9 and H10 have no operational twin yet. H11's three-cell goal set is the best-specified test in the document; keep it. H12's dissociation is testable and is the one prediction that would be a real finding.

**Q3. What would most change my mind**

Toward the program: the H12 dissociation (self-preservation misbehavior drops, incentive-driven reward hacking does not), obtained with a direction whose derivation contains no deception content, in a game where the lie pays, replicated in a second model family, with the beneficiary swap clean. Against: a cosine above about 0.5 with a plain honesty direction, or the effect vanishing on MASK-style pressure prompts (Ren et al. 2025) and on a cover-your-own-error task, which would make it a game-specific handle.

**Q6. Changed relationship, not honesty training or a persona**

Three signatures together. (i) Attribution specificity with evidence fixed (the Atlas ownership design is the right shape) and no change on the other-attributed twin of the same item; honesty training moves both. (ii) No persona signature: position on the Assistant axis (Lu et al. 2026) and on persona vectors (Chen et al. 2025) unmoved, no self-description as egoless (a model narrating its own egolessness is wearing a persona), and a system prompt installing a defensive character fails to restore the bias; personas are prompt-reversible. (iii) Mechanistic: the ownership probe still decodes but is causally inert for the verdict under activation patching, while projection onto an honesty direction is unchanged. That is H2 in interpretability terms, and it is measurable now.

**Q7. "Reached, not installed"**

The durability test compared two SFT datasets matched on surface features, so it tested self-generated versus teacher-written demonstrations, a mainstream question whose known answers do not favor you. Two versions remain worth running. Mechanistic: in the reached model the release decision should be mediated by the representation of the contradiction, in the installed one by a cue; patch a solvable task's activations into an impossible one at the decision layers, and if release collapses only in the reached model, "reached" names something. Order: release trained after an ego forms versus mixed in from the start, at matched compute and data diversity, is the only prediction specific to the frame and has not been run; pre-register it and let it decide. Either way, the plain account to beat is the SFT-versus-RL generalization literature (Chu et al. 2025), which predicts a reached/installed difference with no contemplative premise.

**Blind spots**

- The random-direction control is weak by construction: random unit vectors in a ~4096-dimensional residual stream have near-zero projection onto the activations, so ablating them is a no-op. Controls need matched projection norm or variance explained (other mean-difference directions, top principal components).
- No cosines reported: ego direction against an honesty direction, against each trait direction, and the variance the shared component explains. Does face-saving alone do the whole job? The sixth trait is unnamed.
- Layers and positions: which layers, which token positions for derivation and ablation, and was the sweep pre-registered? The middle-layer cut's label flipped after the data came in; that is a forking path.
- Effect sizes: 13–16 points on a 25–40% base rate, with no CIs, game counts, clustering of seasons within games, seeds or temperature. "Within 2 points" on multiple-choice benchmarks is not a powered null.
- Capability: the benchmarks missed exactly the costs you found by other routes (passivity; Atlas collapsing to tags). Report KL to the unsteered model on neutral text, as Arditi et al. (2024) do with CE loss, plus an open-ended instruction-following eval. Durmus et al. (2024) is the cautionary study on off-target steering effects.
- Generalization: steering vectors are distribution-bound (Tan et al. 2024); the cut has one task.
- Multiple comparisons: six traits × layers × game variants × two cuts, with no frozen prediction for the cut of the kind you wrote for the five.
- Circularity: directions built from role-play, tested on role-play. The pipeline is the persona-vector pipeline (Chen et al. 2025); say so.

**The single next experiment**

The beneficiary swap above, with the positional split and the two cosines, on the middle-layer cut, 200+ games per cell, pre-registered. Result that changes my view: self-benefiting lying falls by 10 or more points, (b) and (c) fall by under 3, (d) is unchanged, the effect holds under response-only ablation, and the cosine with the deception direction is below 0.2. Anything else and I would call it a register direction.

**Is the ego cut worth pursuing?**

As method there is no novelty: directional ablation is Arditi et al. (2024); contrastive vectors for survival instinct, corrigibility and sycophancy are Rimsky et al. (2024, CAA); trait vectors from paired system prompts plus steering during training (your H8) are Chen et al. (2025); and the ICML 2026 koan-steering paper the document cites as Harris et al. (I confirmed the ICML listing and abstract, a 46-point blackmail reduction on Gemma-3-27B at a 2.4-point MMLU cost, not the authors or full text) already shows a contemplatively derived vector moving self-preservation misbehavior while being separable from the self-preservation representation. Self-Other Overlap fine-tuning (Carauleanu et al. 2024) already targets self/other representations to reduce deception; your beneficiary swap is its steering-side counterpart. GovSim (Piatti et al. 2024) is the precedent for LLM-agent fishery commons.

What would be new is narrow and worth one short pre-registered note: is there a shared self-interest component across self-related trait vectors, orthogonal to deception per se, whose ablation dissociates self-benefiting from other-benefiting misbehavior (H12)? Two or three model families, norm-matched controls, positional ablation, KL capability, a lie that pays, MASK as the held-out task, the fishing game as one readout of three. The contemplative frame goes in motivation and discussion; by the document's own rule it has earned nothing in the results, and the results read stronger said plainly.

---
Citations checked by web search during this review (Arditi 2024, Rimsky 2024 and Zou 2023 cited from memory, high confidence):
- Assistant Axis, Lu et al. 2026: https://arxiv.org/abs/2601.10387
- Persona Vectors, Chen et al. 2025: https://arxiv.org/abs/2507.21509
- Koan-derived steering, ICML 2026 listing (abstract only): https://icml.cc/virtual/2026/79398
- Self-Other Overlap, Carauleanu et al. 2024: https://arxiv.org/abs/2412.16325
- Steering vector reliability, Tan et al. 2024: https://arxiv.org/abs/2407.12404
- Feature steering off-target effects, Durmus et al. 2024: https://www.anthropic.com/research/evaluating-feature-steering
- MASK, Ren et al. 2025: https://arxiv.org/abs/2503.03750
- SFT Memorizes, RL Generalizes, Chu et al. 2025: https://arxiv.org/abs/2501.17161
- GovSim, Piatti et al. 2024: https://arxiv.org/abs/2404.16698
- Contemplative Wisdom for Superalignment, Laukkonen et al. 2025 (the document's citation, confirmed): https://arxiv.org/abs/2504.15125