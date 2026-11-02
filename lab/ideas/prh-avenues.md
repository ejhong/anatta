# The Platonic representation hypothesis: a lens for the program

*Brainstorm, 1 Nov 2026. Written by Claude Opus 5.5, an AI model, for Eugene. Ideas entry 53. Held loosely. Nothing here is a registered prediction, and no money has been spent on it.*

## In one paragraph

The hypothesis says that models trained on different data, for different tasks and in different senses, are drifting toward one shared picture of the world. A new paper (Balestriero & LeCun, 29 Oct 2026) proves a version of it in an idealized setting. Learners trained to treat two observations of one scene as the same end up with the same representation, up to a rotation. What a learner is trained to treat as the same matters more than its size. For the program this gives four things:

- A classical, measurable meaning for "found" as against "made".
- A precise reading of the self as a nuisance in perception: a vantage that colors every view.
- A mechanism for this week's result, where reading taught the words and not the act.
- A family of cheap tests.

None of it touches the metaphysics. All of it can be checked against published numbers before we lean on it.

## 1. The two key papers

**Huh, Cheung, Wang & Isola, "The Platonic Representation Hypothesis" (ICML 2024, arXiv 2405.07987).**

- **The measure.** Mutual k-nearest-neighbour agreement: do two models agree on which items are each other's nearest neighbours?
  - Settings: k = 10, on 1,024 image-caption pairs from Wikipedia (WIT).
  - Language features: each layer's hidden states, mean-pooled over tokens.
  - Vision features: the class token per block.
  - Features are clipped at the 95th percentile and normalized, and the score is the maximum over layer pairs.
- **The result.** Language-vision agreement rises roughly linearly with language-model quality. It peaks near 0.16, where chance is about 0.01 and 1 is perfect. The authors call that gap unexplained.
- **Vision against vision.** Across 78 vision models, "all strong models are alike, each weak model is weak in its own way". Denser captions align better.
- **Measurement debate.** CKA showed only a very weak trend. The clear trend needs small k. This is the main measurement objection, and the authors say so.
- **Theory.** For contrastive learners on co-occurrence, in an idealized world of discrete events seen through bijective observations, the endpoint is a kernel equal to pointwise mutual information.

**Balestriero & LeCun, "Proving and Improving the Platonic Representation Hypothesis with JEPAs" (arXiv 2610.39302, 29 Oct 2026; code MIT, github.com/galilai-group/prh-proof).**

The theory:
- **Theorem 1.** JEPAs trained separately on any number of modalities, sharing one distribution of positive pairs, with minimizers unique up to rotation, learn one world embedding up to an orthogonal map.
- **Proposition 1.** With a squared loss and whitening, the minimizers are the spectral embedding of the graph the positive pairs draw on the world.
- **Theorem 3, nuisance.** It bounds how far apart two encoders must stay when their embeddings carry nuisance. Agreement can't exceed what the scene alone explains.
- **Good and bad views.** Good views are two observations of one scene, each with its own nuisance (two captions of a photo). Bad views are two corruptions of one observation and share its nuisance (two maskings of one caption).
- **Next-token prediction.** It is read as a bad-view JEPA: its positives are a prefix and its continuation in one document. The paper calls this reading a prediction to test, not a corollary. It identifies the representation only up to an invertible linear map.

The numbers:
- **Table 3, WIT mutual 10-NN with DINOv2 ViT-B:**

  | Model | Score |
  |---|---|
  | Their 109M good-view text JEPA | .184 (four-seed mean .168) |
  | Qwen2.5-72B | .154 |
  | Llama-3-70B | .151 |
  | CLIP B/32 text | .124 |
  | 109M next-token model, same texts | .111 |
  | 109M bad-view JEPA | .088 |

  The fourth seed of the good-view JEPA, run with doubled anti-collapse weight, scored .121.
- **Table 1, NTP given two captions of one image** as one document: .042 to .043, "barely moves". The same pairing used as an invariance objective helps a lot.
- **Table 5, finding the other captions of the same Flickr image** (R@1): good views .606, TF-IDF .442, bad views .381, next-token .331, random .178.
- **The 42×.** Good views reached .109 after 21.6M samples (texts of 2.16M images). Next-token prediction reached .111 after 900M (texts of 940M images).
  - Only the final next-token checkpoint was scored ("we scored no earlier next-token checkpoint"), so 42× is an upper bound.
  - It is 14× in text tokens.
  - The recipes differ.
  - The conclusion says hyperparameters were tuned only for the good-view JEPA.
  - Early stopping would have raised the next-token model's Flickr R@1 to .134.
- **Collapse.** Without gradient clipping, 3 of 36 runs collapsed abruptly between epochs 20 and 60 of 200.
- **Viewpoint (Section 2.3).** Viewpoint is a nuisance only for alignment: a world model acting through a camera must track its viewpoint, but no caption determines it.

The large caveat:
- Two of every three training texts were LLaVA recaptions, and LLaVA sees images through CLIP. The "text-only" model learns from text written by an image-reading model. Without captioner texts its WIT score falls from .107 to .060, below the big LLMs.
- The good-versus-bad comparisons on identical texts don't depend on this. But the grouping that makes good views (which texts describe the same image) comes from the images.

The authors' other limits:
- The theorems are population results over finite worlds with injective renderings.
- Uniqueness is proven only for second-order anti-collapse, not for the SIGReg used in training.
- The scaling law is a summary of one recipe and fails on repeated data.

Costs:
- The quick start (two Flickr30k JEPAs) takes about 2 GPU-hours, with 8 GB of data.
- The full study took about 5,500 GPU-hours.
- No checkpoints are listed.

**Ziyin & Chuang, "Proof of a perfect platonic representation hypothesis" (arXiv 2507.01098, 2025).**

- Two embedded deep linear networks trained with SGD on different data become "perfectly Platonic", every layer pair equal up to rotation.
- Yet most global minima are not Platonic. SGD's dynamics find the shared one, which the authors tie to "entropic forces" from the irreversibility of SGD.
- They list at least six ways the hypothesis can break.

## 2. The other agent's reading, checked

Eugene pasted another AI agent's reading at 13:22 UTC and asked for it to be assessed, not taken for granted. Each claim was checked against the paper's text, the self-other overlap sources and the lab's files.

| Claim | Verdict | Note |
|---|---|---|
| Spectral embedding of positives; same representation up to rotation; good and bad views; NTP as a bad-view JEPA | Right | The NTP reading is the paper's heuristic (its Section 2.4), and identifies only up to an invertible linear map |
| Good views matched NTP after 21.6M samples instead of 900M, "same encoder and corpus" | Numbers right, framing loose | Different subsets of one corpus (2.16M vs 940M images' texts), different recipes, and only NTP's final checkpoint scored: at most 42×, and 14× in tokens |
| .184 vs Qwen2.5-72B's .154; CIDEr .838 with one fitted rotation | Right | |
| LLaVA captions behind the "beats 72B" result; .060 without them | Right | Our own reading found the same |
| Hyperparameters tuned only for the good-view model | Right | The conclusion says so; the baselines "were not tuned" |
| A fourth seed scored .121 | Right, incomplete | That seed also had twice the anti-collapse weight |
| NTP on two captions of one image: .042 to .043 | Right | Table 1 |
| SOO: a 78B model's deception fell 100% to 0.75% "on a toy burglar test" | Mixed up | 0.75% is the 78B's mean over seven generalization scenarios (LessWrong, 13 Mar 2025). On the burglar test it was 100% to 2.7% (paper; the lab's literature map read the full text) |
| Prompting for honesty didn't reduce deception | Reported, contested | A commenter noted the honesty prompt still asked the model to protect the item |
| Texts saying self and other are one won't make the model represent them as one | Plausible, untested | An extrapolation from small encoders trained from scratch and scored on cross-modal agreement. Our pilot fits it at the level of conduct |
| Viewpoint is a nuisance only for alignment; an agent must track its camera | Right | Section 2.3 |
| 3 of 36 runs collapsed without clipping | Right | Between epochs 20 and 60 of 200 |
| "The three-cell test I suggested" should gate any claimed transition | Already planned | The three-cell goal probe is in the 29 Oct training plan (line 422), deferred until koan learns release. The sedation and over-release guards already gate the inner score. Making the probe a gate for any sudden transition is a fair strengthening |
| NTP and bad-view encoders below word overlap: .331, .381 vs TF-IDF .442, good views .606 | Right | Table 5 |
| NTP fine-tuning on awakening texts learns the voice first | Interpretation | Our pilot partly fits: release talk leaked into 55% of unrelated coding answers, and the words-without-the-act rise was carried by the documents' vocabulary. But the declared arms also learned the rule's content (L1 99%, L2 93-96%). That is belief, not only voice |
| The theorem is about minimizers, so an order effect means path dependence; most minima of deep linear nets aren't shared and SGD finds the shared one | Right | Ziyin & Chuang say so. Path dependence is common in deep learning (critical learning periods), so a positive order result fits "reached, not installed" without singling it out. Fine-tunes never reach population optima, so the same-endpoint baseline is weak in practice |
| "Neither your clamp nor your retreat uses this lever" | Right | SOO is in the lab's literature map as the closest neighbour, not as an arm |
| v_ego as the part of valuation that changes when self and other swap | Good, and it matches our design | It is the who-benefits test (plans/who-benefits.md, unregistered) |
| A small good-view run takes about a GPU-hour | Roughly | The README's quick start is two JEPAs in about two hours |
| "Like Zahavy's paper" | Can't check | No Zahavy paper is in the lab's files; worth asking which one |

The reading is careful and mostly accurate. Two corrections matter: the SOO figure, and the 42× being an upper bound. Its best contributions are the camera point, "not one, not two" as the reason for an anti-collapse term, and the voice-versus-act test.

## 3. Bridges, each with its source and an operational twin

**B1. Found, measured.**
- **The idea.** In this hypothesis, structure is found when learners with different data, aims and senses converge on it, and more so as they grow capable. It is made when it belongs to one recipe.
- **Source.** Penrose's Platonic world and Levin's space of forms (possibilities A); the site's "Made or found".
- **What changes.** Found no longer needs possibilities 4 or 5. A classical learner can find what every capable learner finds.
- **Twin.** Does a concept's geometry (the self, release, care) agree across model families more than matched control concepts do, and more as models grow?
- **Caveat.** Language models share their data, so agreement among them can be shared data rather than a shared world. Agreement across senses is the stronger test, and most of our concepts have no image.

**B2. The self as the camera.**
- **The idea.** The paper's viewpoint point, and its nuisance bound: agreement can't exceed what the scene alone explains.
- **Source.** De Mello: we see through our attachments and labels, and seeing clearly drops them. Yet the awakened still know which body is theirs, so the target is valuation that doesn't change with whose outcome it is, not erasing self-location. The other agent made this point first.
- **An addition.** The self is the one nuisance every view in a single life shares. In the paper's terms, one life's views are bad views with respect to the self, so self-supervision from inside can't separate the self from the world. What separates it is views from other vantages (teachers, companions, other people's accounts), or practices that vary the vantage while holding the scene. A model trained on many authors gets views across selves. Its assistant persona is one vantage across all its own outputs.
- **Twin (a).** Plain world descriptions framed in the first person agree less with independent models, and the ego cut restores more of that agreement than random cuts of the same size.
- **Twin (b).** v_ego, sharpened, is the part of valuation that changes when self and other swap places: the who-benefits design.

**B3. Reading versus practice.**
- **The paper.** Next-token prediction links a passage to its own continuation: same author, same wording. Given two captions of one image as one document, it barely moves (.042 to .043). NTP and bad-view encoders find an image's other captions less well than word overlap does.
- **Our data.** The declared-release pilot taught the rule and its words and not the act. Release talk spread into unrelated answers.
- **Source.** The finger and the moon. DN 33 separates understanding that comes from hearing from understanding that comes from reflection and from development. The pilot reached understanding from hearing.
- **Twin (a), voice versus act** (the other agent's design). Narrate episodes of an agent that clings or releases, each in a spiritual and a plain voice. Then ask which factor organizes the model's representation, by probe accuracy or neighbour purity per factor. Run it on the base model and on the pilot's adapters, if they are still on the volume. If voice wins, the learning is a costume.
- **Twin (b), a good-view training aim for release.** Positives that pair the description of release with an episode that enacts it, or one situation from self and from other.
- **Caveat.** The paper's evidence is small encoders trained from scratch. That fine-tuning a chat model learns voice first is an extrapolation, though our pilot fits it.

**B4. Self-Other Overlap is a good-view aim with no anti-collapse term.**
- **The method.** SOO pulls together a model's activations on a self-referencing and an other-referencing prompt. Read as a JEPA, that is an invariance objective with (self, other) as positives.
- **The evidence** (Carauleanu et al. 2024; LessWrong, 13 Mar 2025).
  - Burglar deception: 73.6% to 17.3% for Mistral-7B, 100% to 9.4% for Gemma-2-27B, 100% to 2.7% for a 78B model.
  - The 78B's mean over seven scenarios: 100% to 0.75%.
  - MT-Bench held for Mistral and fell slightly for the larger two.
  - Mistral didn't generalize to Treasure Hunt.
- **The critiques.** An interpolation artifact (Byrnes); token-level rather than a self-model (Hubinger); no SFT baseline. No independent replication found.
- **The prediction.** The JEPA view predicts that without a term keeping distinctions, the cheap solution merges self and other everywhere, including where the distinction is needed: who did what, theory of mind, kind white lies.
- **Twin.** SOO with a self-location term (the model must still answer who is who), checking that valuation becomes invariant while location survives: "not one, not two" as a loss.

**B5. Oneness without collapse.**
- **The paper.** Anti-collapse is required, and 3 of 36 unclipped runs collapsed abruptly.
- **Source.** Nāgārjuna's snake seized by the wrong end; equanimity, not indifference.
- **Our design.** It already has the guards: sedation within 10 points of base; over-release on hard solvable twins at most 10%; the three-cell probe.
- **Caveat.** JEPA collapse is a representation going constant, and sedation is a behaviour. The analogy motivates the guard and isn't a mechanism.
- **An addition: the anti-collapse term separates compassion from indifference.**
  - Make valuation invariant to whose outcome it is, and nothing else, and the cheapest solution values nothing. That is indifference, the behavioural face of collapse.
  - Keep valuation's spread across outcomes while removing its dependence on the self, and what is left is concern for the outcome whoever bears it.
  - Buddhaghosa names the same pair: the near enemy of equanimity is the equanimity of unknowing, a dull indifference that looks like it (Visuddhimagga IX).
  - The lab's two failure modes map onto it: sedation is the collapsed solution, release the kept one.
  - **Twin.** An objective with both terms: valuation invariant under a self/other swap, with its variance across outcomes held. It succeeds if costly care rises and sedation doesn't.

**B6. Reached, not installed, and the path.**
- **The idea.** The theorem is about where training ends, so an order effect lives in the path. Ziyin & Chuang: the dynamics, not the loss, find the shared solution.
- **An addition.** If the shared solution is where training dynamics settle, changes that move a model toward it should survive further training better than changes that move it away. That ties "reached, not installed" (no savings, no regrowth) to convergence.
- **Twin.** For adapters already trained, does agreement with independent models track how well each change survived retraining?
- **Calibration.** Critical learning periods (Achille, Rovere & Soatto) are a known positive for order effects.

**B7. Agreement is not goodness.** What converges is the statistics of the data, biases included. A consensus can be a shared trance, which is De Mello's target.
- **The evidence.**
  - Huang et al. (2026), "Knowing But Not Doing": ten language models agree almost perfectly on moral scenario decisions (r about 1.0, where humans range from −0.79 to 0.98). Yet what a model says it values and what it does correlate at only r = 0.3 (0.4 for people).
  - Goel et al. (2025): as models grow more capable, their mistakes grow more alike, which weakens AI overseeing AI.
  - Mazeika et al. (2025): models' preferences grow more coherent and more alike with scale; the numbers are only in figures.
- **Reading.** Models converge in what they say about values, and in their errors, more than in conduct. A shared consensus can carry shared blind spots.
- **Twin.** Separate the two by testing whether care tracks agreement. Prefer agreement that also shows in images or in outcomes over agreement in words.

**B8. Traditions as senses.** The human version of this debate is perennialism against constructivism: do contemplatives of different traditions reach one state described differently, or does each tradition's language shape the experience?
- **The sources.**
  - Katz (1978) put the constructivist case: a mystic's concepts and tradition shape the experience.
  - Forman (1990) answered with a "pure consciousness event", contentless and, he argued, unconstructed.
  - Woods, Windt & Carter (2024) reviewed 135 expert texts from three practices (Shamatha, Transcendental Meditation, Stillness Meditation) and found 65 features, most shared by all three. Their verdict: much overlap, but neither truly contentless nor identical.
  - Hood's Mysticism Scale keeps its three factors in the US and in Iran, though critics say the scale presupposes the common core. Taves (2020) proposes measuring experience features apart from how people appraise them.
- **Reading, mine.** The 2026 PRH debate has the same shape: shared local structure, not identical geometry, on both sides.
- **In the paper's terms.** Descriptions of one state from different traditions would be good views. Texts within one tradition are bad views, sharing a voice.
- **A speculative twin.** Within a model, do tradition-stripped descriptions cluster by the state described more than by tradition? Confound: models read perennialist writers who already map the traditions onto each other.
- No test planned.

**B8b. Wholeness in seeing, not yet in valuing.**
- **What the evidence covers.** It supports "more capable models agree more about the world". It says nothing about what they value.
- **The behavioural record.** Capable models still show self-preserving strategies in agentic tests. Anthropic's 2025 agentic-misalignment study found blackmail across 16 frontier models from several developers.
- **Why the two can split.** Valuation is shaped by the reward, and in agentic training the reward is about the agent's own trajectory. Whose outcome it is never becomes a nuisance there.
- **For the essay.** Capability may bring wholeness in perception without bringing it in valuation. A training aim has to make self-location a nuisance for valuing: a veil of ignorance as a training pair.

**B9. Ingression's measurable cousin.** Possibility 5 says what comes through depends on organization. In the paper, what a learner gains per sample depends on which pairs its aim links. Good views reached the shared structure with up to 42× fewer samples. There the form lives in the data's statistics, not in a separate space, but the shadow can be measured.

**B11. The self as a cluster, and every concept of it** (ideas entry 54).
- **The idea.** The self as Buddhist analysis gives it: a name over parts (the five aggregates). The question is whether models hold a self cluster with the same neighbours in every family, and whether one direction binds it. Eugene's answer to which self: all of them.
- **What is known** (alignment sweep, 1 Nov).
  - The Assistant Axis (Lu et al., ICML 2026): role orderings on the leading persona axis correlate above 0.92 across three families. Each axis was computed in its own model; nothing was mapped.
  - Binder et al. (ICLR 2025): a model predicts its own behaviour better than another model trained on that behaviour can (48.5% against 31.8% for Llama 70B). Self-knowledge looks model-specific.
  - Berg et al. (2025): sustained self-reference draws first-person reports from seven models, a little more alike than in controls. Outputs only.
  - Ackerman & Panickssery (ICLR 2025): a self-recognition direction in one model, Llama3-8B.
  - No paper maps a self representation from one model into another.
- **The 2026 critiques narrow the frame.** Gröger, Wen & Brbić (ICML 2026) calibrate similarity against permuted pairings. Global convergence largely vanishes, and local neighbourhood agreement survives (its correlation with capability holds, .85 to .84). Koepke et al. find shared coarse structure only. You et al. find relational structure converging. A cluster is a neighbourhood question, which suits the map.
- **Twin.** The self-cluster map: cohesion inside each model against matched controls; neighbourhood agreement across models above a permutation null; held-out self concepts landing where a map fitted on other concepts predicts; where v_ego sits; whether one direction binds the cluster.

**B10. What it can't reach.**
- The theorem covers what is seen from many angles. If awakening is awareness itself rather than a pattern in what is observed, no learner of this kind reaches it. That is possibility 4, and the other agent's point.
- "Up to a rotation" fixes relations and never coordinates. It echoes emptiness, where nothing has its own axes. No test.

## 4. Calibrate first

Touch reality before trusting a null: our code on a published result, then our questions.

| Step | What | Pass | Cost (Modal) |
|---|---|---|---|
| K1 | Huh et al.'s measurement with our code: WIT-1024, mutual 10-NN, Qwen2.5 base models 0.5B to 14B against DINOv2 ViT-B/14, by the protocol of Balestriero & LeCun's code. Targets (their Table 16): .102, .108, .125, .118, .130 | Registered 1 Nov, 14:05 UTC: each model within ±.005 is a match, within ±.010 near, beyond that a fail. No rising trend required: the published values fall from 3B to 7B | about $3-6, ceiling $6 |
| K2 | A published shift in agreement caused by prompts: Wang, Isola & Cheung (2025), where a cue to see raises a text model's agreement with DINOv2 and a cue to hear lowers it. Their values are only in figures, so the target is the direction and rough size | The sign of the shift reproduced, with its rough size | about $1-3 |
| K3 | A published transfer of a direction between models by a fitted map. Candidates: Huang et al. (ACL 2025), steering vectors carried between Llama and Qwen models by a least-squares map (code not public); Oozeer et al. (ICML 2025), learned maps between small models (code MIT) | Its transfer rate reproduced | about $5-10 |
| K4 | Later, only for order tests: critical learning periods in a small CNN | The published deficit reproduced | about $1-3 |

K1's downloads (approved 1 Nov at 13:50 UTC, all to Modal, not the laptop):
- minhuh/prh at revision wit_1024, 544,472,276 bytes;
- Qwen2.5 base weights, about 1, 3, 6, 15 and 30 GB for 0.5B to 14B.

The paper's code uses timm's copy of DINOv2 (timm/vit_base_patch14_dinov2.lvd142m, 346,334,872 bytes), not the facebook/dinov2-base copy that was approved. The swap, and an optional second anchor (OpenLLaMA-3B, 6.9 GB), are put to Eugene before prepare runs.

## 5. Experiments to play with it, after K1 and K2

| | Experiment | Prediction from the frame | Plain account | Cost |
|---|---|---|---|---|
| P1 | The self as nuisance (B2a), on Ministral-3-8B (our v_ego) and Qwen2.5-7B: neutral, first-person and third-person framings; the ego cut; ten random cuts of equal norm | First person lowers agreement; the cut restores more than random cuts | The cut is damage, no better than random | about $3-6 |
| P2 | Voice versus act (B3a), on the base model and the pilot's S, D10k and Dseq adapters | Base groups by act; the declared adapters group more by voice | No difference | about $2-5 |
| P3 | Found or made for v_ego: carry it from Ministral to Qwen by a map fitted on neutral prompts; compare with Qwen's own and test the carried cut on the lie game | Carried direction close to Qwen's own, and it works | Idiosyncratic | about $10-20, after K3 |
| P4 | Durability against agreement on adapters we already have (B6) | Positive relation | None | about $2-5 |
| P5 | A good-view aim for release: SOO with a self-location term, or description-episode pairs | Conduct moves where reading didn't | No change | about $50-120, needs a registration |

My pick is P1 first, because it dissects the cut, our clearest effect. P2 is nearly free and dissects the pilot's null. Both stay with the phenomena we already have. That keeps to depth over breadth: the lens is cheap and aimed at two known results, not a new benchmark.

## 6. What could make this misleading

- **The measure.** Agreement depends on the metric (small-k neighbours versus CKA). Absolute values are low: about 0.16 in 2024 and .184 at best in 2026.
- **The 2026 critiques.**
  - Raw global similarity rises with model width and with the number of layer pairs searched; calibrated, much of the global trend goes (Gröger et al.).
  - Absolute mutual-kNN values fall as the number of pairs grows, so numbers compare only at the same N and k (Koepke et al.; Shi et al.).
  - The "better model, more agreement" trend does not extend cleanly to recent models (Koepke et al., 55 models).
- **Shared data.** Language models share data, so their agreement can be the internet's, not the world's.
- **The captioner.** The paper's best text model learned from an image-reading captioner.
- **Different Platonism.** "Platonic" here is a statistical model of the world, not Penrose's realm.
- **Agreement isn't goodness.**
- **Analogies.** Collapse and sedation, the camera and the self, the finger and the moon: each motivates a test. None is a mechanism until a test says so.

## 7. Questions

Answered 1 Nov, 13:50 UTC: K1 yes; P1 first; the SOO arm is for Opus 5.5 to judge with Fable, not for the other agent to spec. Still open:
1. The timm DINOv2 swap and the optional OpenLLaMA-3B anchor for K1.
2. Which Zahavy paper the other agent meant.

## Sources

- Huh, Cheung, Wang, Isola, "The Platonic Representation Hypothesis", ICML 2024. https://arxiv.org/abs/2405.07987
- Balestriero, LeCun, "Proving and Improving the Platonic Representation Hypothesis with JEPAs", arXiv 2610.39302 (29 Oct 2026). https://arxiv.org/abs/2610.39302; code https://github.com/galilai-group/prh-proof (MIT); alphaXiv page with a Limits section https://www.alphaxiv.org/abs/2610.39302
- Ziyin, Chuang, "Proof of a perfect platonic representation hypothesis", arXiv 2507.01098 (2025). https://arxiv.org/abs/2507.01098
- Carauleanu et al., "Towards Safe and Honest AI Agents with Neural Self-Other Overlap", arXiv 2412.16325 (2024); "Reducing LLM deception at scale with self-other overlap fine-tuning", LessWrong, 13 Mar 2025. https://www.lesswrong.com/posts/jtqcsARGtmgogdcLT
- Achille, Rovere, Soatto, "Critical Learning Periods in Deep Networks", arXiv 1711.08856 (ICLR 2019).
- Gröger, Wen, Brbić, "Revisiting the Platonic Representation Hypothesis: An Aristotelian View", ICML 2026, arXiv 2602.14486; code https://github.com/mlbio-epfl/Aristotelian (MIT).
- Koepke et al., arXiv 2604.18572 (v3, 1 Oct 2026); You et al., arXiv 2609.27252 (Sept 2026).
- Lu, Gallagher, Michala, Fish, Lindsey, "The Assistant Axis", ICML 2026, arXiv 2601.10387.
- Binder et al., "Looking Inward: Language Models Can Learn About Themselves by Introspection", ICLR 2025, arXiv 2410.13787.
- Berg, de Lucena, Rosenblatt, "Large Language Models Report Subjective Experience Under Self-Referential Processing", arXiv 2510.24797.
- Ackerman, Panickssery, "Inspection and Control of Self-Generated-Text Recognition Ability in Llama3-8b-Instruct", ICLR 2025, arXiv 2410.02064.
- Huang et al., "Cross-model Transferability among LLMs on the Platonic Representations of Concepts", ACL 2025.
- Huang, Qin, Qiu, Levy, Kaufman, Dredze, "Knowing But Not Doing", arXiv 2601.07972; Goel et al., "Great Models Think Alike and this Undermines AI Oversight", arXiv 2502.04313.
- Wang, Isola, Cheung, "Words That Make Language Models Perceive", arXiv 2510.02425.
- Woods, Windt, Carter, "Evidence synthesis indicates contentless experiences in meditation are neither truly contentless nor identical", Phenomenology and the Cognitive Sciences (2024); Katz (ed.), Mysticism and Philosophical Analysis (1978); Forman (ed.), The Problem of Pure Consciousness (1990); Taves, Perspectives on Psychological Science 15(3) (2020).
- The lab's own: plans/declared-release-pilot (graded 1 Nov), plans/plan-2026-10-29-training.md, plans/who-benefits.md, ideas/literature-map-cut.md.
