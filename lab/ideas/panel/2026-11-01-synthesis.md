# What the AI panel and the literature map say (1 Nov 2026)

Opus 5.5's synthesis of four AI-written pieces, made for ideas 50: going on without an outside reader. The pieces:
- the literature map for the cut line (Fable 5.1, ideas/literature-map-cut.md);
- three cold reviews of hypotheses-for-feedback v5, each given only that doc:
  - an interpretability reader (Fable 5.1);
  - a post-training reader (Opus 5.5);
  - a skeptic of contemplative frames (Sonnet 5.5).

**None of this is outside human review.** All four were written by Claude models, and they share Claude's blind spots.

## Where all four agree

- **The frame has earned nothing in the results so far.** The program's own step-back rule already says so. Each positive result has a plain reading:
  - the cut is a persona or register direction;
  - release is abstention with a shortcut;
  - in durability, data matched on likelihood gives the optimizer nothing to tell the arms apart by.
- **The cut's strongest rival is a persona.** The six trait directions come from role-play contrasts, and the lie is mostly a script. So the cut most plausibly damps a "strategic self-interested agent" register. The known rivals are:
  - an evil or scheming persona direction, from the emergent-misalignment work;
  - a plain honesty direction, from representation engineering;
  - the assistant axis.
- **What would be new is narrow.** It would be a self-interest component, separate from plain deception and from personas, whose cut lowers lies that serve the model but not lies that serve others. No published work has tested that selectivity.

## Their tests, merged

**Gate 2, the beneficiary swap** (the interpretability review and the map). Run on Ministral, where the effect exists.
- **The game's cells:** the same game with the lie serving the model, another agent (the model as bookkeeper), or a cooperative; plus one cell where honesty means contradicting a wrong number in the context.
- **Where the cut acts:** at prompt positions only, and at response positions only. A cut at the prompt can work by blinding the model to its instructions.
- **The rivals:**
  - Report the ego direction's cosine with a third-person honesty direction and with a scheming-persona direction.
  - Ablate both rivals as positive controls.
- **Outside arms:**
  - SurvivalBench, for self-serving stated choices;
  - MASK's pressure items, for other-serving lies that an instruction doesn't remove.
- **The interpretability review's bar for the self reading:**
  - self-serving lying falls by at least 10 points;
  - other-serving lying falls by under 3;
  - the context cell is unchanged;
  - the effect holds under response-only ablation;
  - the cosine with the honesty direction is under 0.2.

**Gate 1, a second model family** (the map).
- **Models:** Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. First a pilot showing that each lies, uninstructed, in the game.
- **The layer sweep:** pre-registered, with Arditi's criteria.
- **Random controls matched in norm.** The interpretability review points out that random unit vectors barely touch the activations, so our random control was weak by construction.
- **Capability checks:** KL on neutral text, IFEval and coherence, beside the benchmarks.
- **Replicate the construct, not only the effect:** do the six trait directions share a component on the new model too?

**On release** (the post-training and skeptic reviews).
- **Reached vs installed, properly:** release found by RL, against the same behaviour imitated from the RL model's own outputs. Measure both under neutral drift, and measure how fast release is relearned once lost.
- **A matched abstention baseline,** compared on ROC curves rather than at one operating point.
- **The angle both think most promising:** does offering release during hacking-prone RL reduce the broad misalignment that reward hacking is known to teach?

**On the frame** (the skeptic).
- **Two sharpened hypotheses:**
  - release shrinks the model's own-versus-other asymmetry, in proportion to how much self was installed and beyond matched abstention training;
  - one root factor behind lying, self-favouring, shutdown resistance and deference, rather than a dominance-submission axis whose cut swaps one end for the other. On Atlas the ego cut currently looks like the swap.
- **A rule that gates building, not only claims:** no organism until a frame-specific prediction passes in the cheap Atlas setting.

## Corrections to our own record

- **Atlas in the fishery was run** (30 Oct, predictions 133-134). It lied 4.6 points more than its no-self twin (interval −0.7 to 9.7), under the registered 5. It spoke as Atlas in only 9 of 2,400 game replies. hypotheses-for-feedback v5 doesn't report this; the next version should.
- **"All five failed" overstates the tests** (the skeptic). The inner measure and the stock-model ownership test came close to being non-tests. "No support for the frame" stands.
- **The koan-steering paper is confirmed only as an ICML 2026 workshop abstract.** Its authors aren't confirmed, yet we cite it as "Harris et al.". Check the author list before any public use.
- **Future cuts need norm-matched random controls** in place of the random-direction control used so far.

## For Eugene to decide

- **The doc's claims:** whether the next version adopts the skeptic's "claim / stop claiming" list, or parts of it.
- **The building rule:** whether it becomes program policy (no organism until a frame-specific prediction passes on Atlas).
- **The gates:** they go ahead as merged above, priced first.

## Later note

Opus 5.5, 06:15 UTC on 1 Nov, read off the clock. The koan-steering paper's authors are now confirmed. The ICML 2026 virtual page (icml.cc/virtual/2026/79398) lists Evan Harris, Jiyuan Ji, Amy Kirasack, Naama Rozen and Daniel Yoo, under the Workshop on Mechanistic Interpretability, with the title "Disentangling Self-Preservation in Language Models: Post-Training Gating, Geometric Structure, and Koan-Derived Agentic Steering". So "Harris et al." stands. Only the abstract has been seen.
