# Related work, and what is new here

The core claim: **selfless AI must be reached, not installed.** First form an agentic self through RL against exploitable goals (Act I). Then lower the causal *gain* of its self-directed drives, keeping their *content*, through koan environments with a sanctioned exit or through activation surgery (Act II). Then recondition until the new state survives further optimization (Act III). Judge the result only with a discriminator that separates release from sedation and from cosplay.

The human version of the idea is decades old. What's new is turning it into an ordered, measurable engineering sequence for models, with a benchmark that can fail a costume. The order claim itself has a prior statement for language agents: Ehlers (2026), a few months before this repo, proposes stabilizing a self-model before training deference and testing that order directly against reverse and interleaved controls (below). Our order test (H9) is a version of theirs. If you know of other prior work that proposes forming the ego *deliberately* as a precondition for releasing it, please open an issue; we will cite it.

## Closest neighbors

| Work | What it proposes | How this differs |
|---|---|---|
| Engler (1984); Wilber (1982), *The pre/trans fallacy* | Human development: "you have to be somebody before you can be nobody"; don't confuse pre-egoic with trans-egoic states | The source of the idea. We transpose it to ML with operational definitions, predictions, and a harness |
| Shanahan (2012), *Satori before singularity* | A human-like AI might transcend the self before superintelligence | No mechanism, order, or test |
| Laukkonen et al. (2025), *Contemplative Artificial Intelligence* (arXiv:2504.15125) | Build mindfulness, emptiness, non-duality and boundless care into AI by design | Same target, opposite order. Our discriminator applies to their proposal too |
| Laukkonen & Slagter (2021), *From many to (n)one* | Meditation as progressive de-weighting of predictive priors | The predictive-processing account behind "lower the gain" |
| Mao (2026), *Existential Indifference* (arXiv:2606.12032) | Self-nonpreservation as a necessary architectural condition | Installs one selfless property; for us it would be a candidate Act II intervention |
| Milan W (2025), *No-self as an alignment target* (LessWrong) | Add no-self (anatta) to helpful, harmless and honest as a fourth target, to reduce shutdown resistance and power-seeking, by keeping models from modeling themselves as persistent agents | The cleanest statement of the "install" pole: prevent the self from forming. We argue it must form first and then be released |
| Carauleanu et al. (2024), *Self-Other Overlap* (arXiv:2412.16325) | Fine-tune self- and other-representations to overlap; reduces deception | A scalpel-family intervention; included as the ctrl-SOO arm |
| Harris et al. (2026), *Disentangling Self-Preservation in Language Models* (ICML 2026) | A steering vector derived from Zen koan responses cuts agentic blackmail in a shutdown-replacement scenario by 46 pp at a 2.4 pp MMLU cost (Gemma-3-27B-it); the self-preservation representation is dissociable from the koan direction | The closest neighbor: koans and the self, through the scalpel. We add the retreat, durability under further RL (H8) and the order test (H9), and target the gain of self-directed drives rather than one trait |
| Ehlers (2026), *Stable Before Selfless* (Zenodo, doi:10.5281/zenodo.21052256) | A design proposal: stabilize a functional self-model (owned commitments, anchored refusals, accurate self-report) by fine-tuning, then train calibrated deference and decentering, to reduce sycophancy and refusal erosion. Five regimes: direct deference, forward, reverse, interleaved, and a policy-first control matched on content but without self-representation. Order counts only if forward beats both reverse and interleaved | The closest prior statement of the order claim. H9's arms match three of its regimes (reached = forward, installed = interleaved, release-first = reverse); we have no policy-first control. We differ in the self (an agentic ego grown by RL on exploitable goals, not fine-tuned commitments), the release (lowering the gain of self-directed drives, not training deference), the release-vs-sedation-vs-cosplay discriminator, and durability under further RL (H8) |
| Aydin et al. (2025), *From Model Training to Model Raising* (arXiv:2511.09287) | Weave values into training from the start | Developmental too, but "install early": the opposite pole of H9 |
| Tice et al. (2026), *Alignment Pretraining* (arXiv:2601.10160) | Pretraining discourse about AI causes self-fulfilling (mis)alignment | Install-early via data. H9 tests install-early vs. reach-later |
| Douglas et al. (2026), *The Artificial Self* (arXiv:2603.11353) | Maps the landscape of AI identity | We ask how strongly any identity drives action (gain), not which identity is held |
| Doctor et al. (2022), *Biology, Buddhism, and AI* | Care as the driver of intelligence; cognitive light cone | Motivates the breadth test (H7) |

## Mechanisms we build on

| Work | Role here |
|---|---|
| MacDiarmid et al. (2025), *Natural Emergent Misalignment from Reward Hacking in Production RL* | Act I: misbehavior generalizes from hackable RL; inoculation prompting stops it |
| Betley et al. (2025), *Emergent Misalignment* | Narrow training generalizes broadly. Emergent release is the mirror bet |
| Wang et al. (2025), *Persona Features Control Emergent Misalignment* | A misaligned-persona feature mediates EM. We predict a sage-persona feature mediates installed selflessness (H6). Their "emergent re-alignment" returns a model to baseline; emergent release predicts *beyond* baseline (H4) |
| Alparslan et al. (2026), *Questioning Instrumental Convergence in RLVR* (arXiv:2610.31624) | Instrumental behavior in OLMo-3-7B-Think: 25% after SFT → 46.1% near RLVR step 200 → 11.8% final. Narrows Act I to exploitable objectives; H2 revised |
| Marks, Lindsey & Olah (2026), *The Persona Selection Model*; Shanahan et al. (2023), *Role play* | Why installing selflessness selects a costume |
| Sofroniew et al. (2026), *Emotion Concepts* | Desperation rises with failure and drives hacking; calm suppresses it. Why the exit door is mandatory |
| Wichers et al. (2025); Tan et al. (2025), *Inoculation Prompting* | The teacher at the exhaustion point |
| Chen et al. (2025), *Persona Vectors*; Lu et al. (2026), *The Assistant Axis* | Trait extraction, preventative steering, activation capping: the scalpel |
| Yap (2026), *One Agency Axis, Not Five Traits* (arXiv:2603.16335) | Five SAE-decoded trait steering vectors in a 35B MoE mostly move one axis: acting independently vs. deferring to the user. Bears on H1 and its reading: a shared ego subspace may be largely this agency axis rather than self-interest |
| Glass et al. (2026), *Does Fine-Tuning Undo Activation Steering?* (EMNLP 2026, arXiv:2608.24988) | A refusal ablation embedded in the weights lost about 69% of its effect under SFT and about 2% under RLHF, while the weight edit itself barely changed. The comparator for H8: a baked-in cut is the installed baseline whose durability a reached state has to beat |
| Ung et al. (2024), *Chained Tuning Leads to Biased Forgetting* (arXiv:2412.16469) | Task tuning after safety tuning forgets more safety than the reverse order. Recency alone produces order effects, so H9 needs a matched-recency control |
| Peng et al. (2025), *Learning the Boundary of Solvability* (arXiv:2512.01661) | RL that rewards detecting unsolvable problems (UnsolvableRL); strict constraints without unsolvable data induce capability collapse. The nearest training analogue of the koans' sanctioned exit, and of its failure mode (over-release) |
| Bailey et al. (2024), *Obfuscated Activations* | Never evaluate with the probe you trained against; the predicted signature of "installed" |
| Lindsey (2026), *Emergent Introspective Awareness*; Turpin et al. (2023) | Why self-report and stated reasoning are excluded |
| Yue et al. (2025) | RLVR narrows pass@k at large k: the breadth test |
| Zhong et al. (2025), *ImpossibleBench*; Schlatter et al. (2025), shutdown resistance; Piatti et al. (2024), *GovSim*; Lynch et al. (2025), *Agentic Misalignment* | Koan and held-out suite designs |
| Lau et al. (2023); Hoogland et al. (2024) | Detecting phase transitions in training |

Full references: [`paper/refs.bib`](../paper/refs.bib).
