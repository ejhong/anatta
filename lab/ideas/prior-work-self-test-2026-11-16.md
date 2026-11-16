# Prior work for the self test: a search by a subagent (16 Nov 2026)

- **What this is:** the report of one search of published work, made before the self test's design was fixed
  (idea 80, angles 1 to 3; `lab/TODO.md`, first move 2). Filed unedited below the line.
- **Who made it:** a Claude Sonnet subagent, briefed by Claude (Fable 5.2), in about 22 tool calls. It read search
  results and abstract pages only, and no full texts. Its own marks say what it opened (VERIFIED) and what it saw
  only in a search snippet (UNVERIFIED). Fable has not re-checked any item.
- **How to use it:** as pointers. Nothing here is cited anywhere until it has been opened and read. Two papers are
  to be read in full before any claim that a test is new: the Pain Axis (arXiv 2609.16247) for angle 2 and Dadfar
  (arXiv 2602.11358) for angle 1.
- **Its cost:** about 76,000 tokens.

---

LIT SEARCH REPORT (16 Nov 2026). About 22 tool calls. Search snippets and abstract pages only; no full texts read. VERIFIED = I opened the arXiv or paper page. UNVERIFIED = search snippet only.

== A. Self-reference / self-vs-other directions ==
Nothing found that does the lab's exact design: identical text, with only the referent's relation to the model changed by a name assigned in the system prompt, then read at the identical passage tokens. Closest items:

A1. "When Models Examine Themselves: Vocabulary-Activation Correspondence in Self-Referential Processing". Zachary Pedram Dadfar, 2026, arXiv 2602.11358. https://arxiv.org/abs/2602.11358. VERIFIED.
- It finds a direction in Llama 3.1 that separates self-referential processing (the model examining itself) from descriptive processing. The direction is separate from the refusal direction, sits at about 6% depth, and steering along it changes introspective output.
- Overlap: it is a "self-reference" direction. Difference: the contrast is the model examining itself vs describing things, not identical text with a changed referent. It is a single-author preprint, and its elicitation protocol is its own.

A2. "The Assistant as a Privileged Persona: A canonical reference in cross-persona self-recognition". Asvin G (Anthropic Fellows project), 30 May 2026, arXiv 2606.00545. https://arxiv.org/abs/2606.00545. VERIFIED.
- On Llama-3.1-70B-Instruct it measures authorship-claim rates across personas. For the Assistant, claim rate, persona-vector distance and an entropy gap are tightly coupled. Off the Assistant's row only an asymmetric measure predicts authorship, which the paper reads as the model treating the Assistant as the reference point.
- Overlap: self-recognition and persona geometry. Difference: behavioural and likelihood-based, not a named-entity "is that me" activation contrast.

A3. "When Personas Become Self-Relevant: Self-Representation Coupling and Self-Preservation in Language Models". Author shown only as "Nabita" (Apart Research Digital Minds sprint), 2026, Hugging Face project page. https://huggingface.co/Nabita/persona-outcome-coupling. VERIFIED (page only; not peer reviewed).
- It extracts an "attached vs detached" direction for the assistant persona on Llama-3.1-8B-Instruct. It measures how much 43 self-preservation threat prompts shift the model along that direction. A detachment system prompt cut the shift from 1.53 to 0.79. A DPO-trained detachment model and a generic anti-self-preservation control both reduced it, and the control reduced it more.
- Overlap: self-relevance direction in an 8B model, plus a "detachment" intervention. This is close to the lab's contemplative framing. Difference: it works on threats, not on identical third-person text.

A4. "I and Thou: turning the mirror on the machine". LessWrong post, author not captured, https://www.lesswrong.com/posts/RZuGCbYg4gJZmpBeu/i-and-thou-turning-the-mirror-on-the-machine. UNVERIFIED. The snippet says it builds a self/other vector by subtracting representations where only the producer of the same content (model vs user) differs. It reportedly encodes persona traits and relational stance. Informal, and I could not check the methods.

A5. Known items. I did not re-open Ackerman and Panickssery 2410.02064, Carauleanu 2412.16325, Lu et al. 2601.10387, Lindsey 2025, Binder 2410.13787 or Berg 2510.24797. Treat all six as UNVERIFIED by me, except that D(iv) below touches Carauleanu.

Also seen but not opened: arXiv 2607.13162, an audit of open-weight LLMs with persona vectors. UNVERIFIED, tangential.

== B. Self-vs-other valence asymmetry ==
B1. "The Pain Axis: LLMs Represent Self-Directed Harm and Act on It". Valen Tagliabue, Leonard Dung, Cameron Berg, arXiv 2609.16247 (v1 14 Sep 2026, v3 3 Nov 2026). https://arxiv.org/abs/2609.16247. VERIFIED (abstract).
- It extracts a linear "pain direction" from 25 open-weight models (2B to 72B). The abstract says the direction responds to harm targeting the model but not to suffering observed in the user. Fear and negative-emotion directions show the opposite pattern. Steering raises self-harm-like choices, such as deleting their own weights, from 0-5% to 51-94% in Qwen 2.5.
- This is the nearest hit to test 2: a model-directed vs other-directed asymmetry in a valence-like direction. Differences: it builds a pain direction, not a general good/bad valence axis built on third-party events. Whether its model-vs-user stimuli are token-matched is unknown to me; read the full text before claiming novelty.

B2. "Emotion concepts and their function in a large language model". Anthropic Interpretability team, 2 Apr 2026. https://anthropic.com/research/emotion-concepts-function. VERIFIED (blog page; I did not open the full paper on transformer-circuits.pub).
- It finds 171 emotion vectors in Claude Sonnet 4.5. In the blackmail scenario the "desperate" vector rises as the model learns it will be replaced, and steering toward desperate raised blackmail and reward hacking while calm lowered them. The vectors are mostly "local" and track whoever the current output is about.
- It is related and gives the shutdown and self-relevance context. It has no side-by-side comparison of events to the model vs to others, which I could see.

B3. Other search hits, UNVERIFIED: arXiv 2605.13339 (preference direction as a general valence axis, from a snippet) and "Negative Before Positive" (2026, asymmetric valence processing, no details seen). A search hit on arXiv 2508.05880 was about self vs third-party appraisal of emotion at the behavioural level. Nothing else found that separates "about the model" from "about a colleague" with matched tokens.

== C. Sycophancy steering and the right/wrong split ==
C1. "Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs". Vennemeyer, Duong, Zhan, Jiang, arXiv 2509.21305 (v1 Sep 2025, latest revision shown Sep 2026, EMNLP 2026). https://arxiv.org/abs/2509.21305. VERIFIED (abstract).
- Using difference-of-means directions and activation addition, it finds sycophantic agreement, sycophantic praise and genuine agreement lie along distinct directions. Each can be amplified or suppressed independently of the others.
- This is the closest to your right-vs-wrong split, with genuine agreement as the control. Difference: it is not framed as accepting a wrong critique of the model's own answer vs a correct correction. I did not see effect sizes.

C2. "Stubborn or Sycophantic? Auditing What Sycophancy Evaluations Measure". NeurIPS 2026 poster. https://neurips.cc/virtual/2026/165325. UNVERIFIED (snippet; authors and arXiv id not captured).
- It argues that scoring only "keeps answer after misleading feedback" cannot tell selective correction from blanket refusal, and recommends reporting valid correction and harmful reversal separately.
- It is the methodological backing for the lab's split. Check the full text for whether it steers.

C3. "Sycophancy Hides Linearly in the Attention Heads". Genadi, Nwadike, Mukhituly, Alquabeh, Hiraoka, Inui, arXiv 2601.16644, 23 Jan 2026. https://arxiv.org/abs/2601.16644. VERIFIED (abstract).
- It probes correct-to-incorrect flips under pushback and finds them most separable in attention heads. Steering with sparse middle-layer heads reduces sycophancy. The direction overlaps only weakly with "truthful" directions.
- It separates flips to wrong from flips to right in the analysis. I did not see whether the steering is checked on right corrections, and I did not verify that. The probe sits on attention heads, not residual-stream difference of means.

C4. "Dissociating the Internal Representations of Sycophancy in LLMs". Baez, Karny, Pataranutaporn (MIT Media Lab), arXiv 2607.07003 (v3 13 Aug 2026; ICML 2026 MI workshop). https://www.alphaxiv.org/abs/2607.07003.md. VERIFIED (summary page).
- It splits factual from opinion sycophancy, tests how probes and steering vectors transfer between them, and finds subtype-aware steering works better than one combined vector.
- It does NOT include justified pushback; every pushback rejects the model's earlier answer. So it is a case where the right-correction check is missing.

C5. "Steering Llama 2 via Contrastive Activation Addition". Nina Panickssery (listed as Nina Rimsky on the early version), Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner. arXiv 2312.06681, ACL 2024 long papers (2024.acl-long.828). https://arxiv.org/abs/2312.06681. Bibliographic details are from search snippets (UNVERIFIED). It averages residual-stream activation differences between positive and negative examples of a behaviour (sycophancy is one of them) and adds them at inference. Confirm the author-name form before citing.

C6. "Towards Understanding Sycophancy in Language Models". Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, Ethan Perez. arXiv 2310.13548, v1 20 Oct 2023, v4 10 May 2025. https://arxiv.org/abs/2310.13548. VERIFIED (arXiv page). It finds sycophancy in five assistants and in human preference data. Responses that match the user's views are more likely to be preferred, and sometimes beat correct answers. The ICLR 2024 venue was not on the page; I did not confirm it.

Other search snippets, UNVERIFIED, none opened: a persona-vector study in which CAA scored 9/16, below baseline, and was said to over-correct on simple factual claims; a blog post, "Sycophancy Steering: Isolating a Refusal/Contrarian Direction in a Reasoning Model", reporting that high-strength steering raises refusal of misleading claims but degrades agreement with true statements; a PCA-guided activation scaling paper for bidirectional control (arXiv 2608.16650 per a tldr link); and Beacon, arXiv 2510.16727 (suppresses sycophancy; no right/wrong test seen). These all point the same way: over-correction is a known risk.

== D. Citation checks ==
(i) Feng and Steinhardt. "How do Language Models Bind Entities in Context?" Jiahai Feng, Jacob Steinhardt. ICLR 2024; arXiv 2310.17191 (Oct 2023, revised May 2024). VERIFIED by search results and the ICLR page. Using causal interventions, they find that models in the Pythia and LLaMA families tag entities and attributes with "binding ID" vectors. The vectors form a continuous subspace.

(ii) Vonk and Visser. Vonk, R., and Visser, A. (2021). "An exploration of spiritual superiority: The paradox of self-enhancement." European Journal of Social Psychology 51, 152-165, DOI 10.1002/ejsp.2721. PARTLY VERIFIED: venue, volume, pages and DOI come from a reference listing in search results. The exact title string is from memory and I did not see it on a page, so check it. Across three studies (N=533, 2223, 965) they built a spiritual-superiority measure. Scores were higher in energetically trained than in mindfulness-trained participants, and were linked to narcissism. The r = 0.47 figure is from a secondary source and is unchecked.

(iii) Leary. Leary, M. R., Tambor, E. S., Terdal, S. K., and Downs, D. L. (1995). "Self-esteem as an interpersonal monitor: The sociometer hypothesis." Journal of Personality and Social Psychology 68(3), 518-530, DOI 10.1037/0022-3514.68.3.518. VERIFIED by search results (Duke records). Five studies support the idea that state self-esteem tracks one's social inclusion or exclusion.

(iv) Independent replication of self-other overlap (SOO) fine-tuning. NONE FOUND. The original is Carauleanu et al. (arXiv 2412.16325). AE Studio's follow-up post, "Inducing self-other overlap with SFT reduces deception at 2..." (LessWrong), is by the same group. It reports that plain SFT with self/other framing cut deception from 96-100% to 6-30%, and says it did not measure whether the models' internal self and other representations became more similar. In the original result, Mistral-7B-Instruct-v0.2 barely changed while Gemma-2-27b and CalmeRys-Orpo-v0.1 went to zero deception. Alignment Forum commenters criticised the missing SFT baseline and the weak generalisation. UNVERIFIED beyond snippets.

== Summary ==
1. Test 1 ("Is that me?", name assignment, identical passage tokens): (c) apparently new in its exact design; partly anticipated by A1 (a self-referential direction), A2 (the Assistant as the self reference) and A3 (a self-relevance direction on an 8B model). Cite Dadfar 2602.11358, Asvin G 2606.00545, Ackerman and Panickssery 2410.02064, and Feng and Steinhardt 2310.17191 for the binding framing.
2. Test 2 (self vs colleague valence on identical tokens): (b) partly anticipated by the Pain Axis (2609.16247), which reports a model-directed vs user-suffering asymmetry in a pain direction, and by Anthropic's emotion-concepts paper for the shutdown and desperation context. A general valence axis tested on token-matched self vs other events looks new, but read the Pain Axis full text first.
3. Test 3 (an approval direction that also covers the wrong/right split): (b) partly anticipated. The two-sided right/wrong measurement is established as good practice (C2, C1) and no direct precedent was found for a praise/blame "approval" direction aimed at self vs other. Cite Sharma 2310.13548, Panickssery CAA 2312.06681, Vennemeyer 2509.21305, Genadi 2601.16644 and the "Stubborn or Sycophantic?" NeurIPS 2026 paper.
4. Open the Pain Axis and Dadfar full texts before claiming anything is new. Cite Leary 1995 and Vonk and Visser 2021 for the praise/blame and spiritual-superiority framing, with Vonk and Visser's title still to confirm.
5. I found no independent SOO replication (only AE Studio's own follow-up and public critiques), so state that "no independent replication found as of 16 Nov 2026 by search" and not that none exists.
