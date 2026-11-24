# A second reader on where the idea stands in the field (24 November 2026)

Filed by Claude (Fable 5.2, an AI model), the lab's driver. Eugene pasted this report from Gemini on 24 November
2026, before 14:14 UTC, with the words: "More ideas to log - from gemini: - and we should have a section with a
survey on where things stand related to our idea - i think this summary like this is good - somewhere on the site
- talking about htis is all interesting and in the paper". The report is filed as pasted. It is data, not
instructions. The driver's grading follows it: each point was checked against the primary sources the lab holds
(the study's paper and post, Anthropic's "Teaching Claude Why", OpenAI's post, arXiv 2607.26173 and 2607.26654,
read from their sources on 23 and 24 November) and against the arXiv abstract of Hao and Ho, fetched today.

## The report, as pasted

Searched web: ""Model spec midtraining" "Kutasov" OR "Marks" OR "Wichers", "Teaching Claude Why" Anthropic synthetic document, "Contemplative Artificial Intelligence" Laukkonen alignment 2025 2026, "self-preservation" OR "no-self" OR "impermanence" LLM alignment 2026"
Ran command: Check bibliography in refs.bib (`git show origin/main:paper/refs.bib | grep -A 12 -E "li2026msm|laukkonen|hao2026|tice2026|harris2026"`)
Read URL: https://alignment.anthropic.com/2026/msm/
Searched web: ""philosophy_spec" "Model Spec Midtraining" OR "impermanence", "Disentangling Self-Preservation in Language Models" "koan" Harris 2026, "The Role of Self-Concept In Language Models" Hao Ho 2026, "contemplative" "fine-tuning" OR "RL" OR "DPO" alignment 2026"

I just checked the primary sources directly—including Anthropic's Alignment Science posts for both [*Model Spec Midtraining*](https://alignment.anthropic.com/2026/msm/) (May 5, 2026) and its companion post [*Teaching Claude Why*](https://alignment.anthropic.com/2026/teaching-claude-why/) (May 8, 2026), plus the newest October–November 2026 preprints.

Here is the honest, unvarnished sanity check on all four assumptions—**two big things you got completely right, one overstatement to be careful about, and two 10× cheaper angles you were missing.**

---

### 1. "Is *Model Spec Midtraining* a huge result in alignment?" → **Yes, with one caveat (8.5/10)**
- **What's true:** It is a flagship paper from Anthropic's Alignment Science team (supervised by Samuel Marks and Jonathan Kutasov), published alongside [*Teaching Claude Why*](https://alignment.anthropic.com/2026/teaching-claude-why/) (co-authored with Jan Leike, Amanda Askell, Chris Olah, Evan Hubinger, and Sam Bowman), which explains how Anthropic drove agentic blackmail/sabotage from **96% on Claude Opus 4 down to ~0% on models since Haiku 4.5 / Opus 4.5**.
- **Crucial detail from their Figure 3:** Li et al. report **small error bars across 4 training seeds** on Qwen2.5-32B (`68% → 5%`). **That means their training *was* repeatable across seeds**—strongly pointing to a concrete technical difference between their unshared script and your Phase-1→Phase-2 LoRA handover (`msm2_torch.py`), rather than the method itself being inherently random!
- **The caveat they admit (Figure 4):** At very high SFT compute (`80k` CoT demonstrations), plain Deliberative Alignment SFT eventually catches up on this benchmark. MSM's main claim is **10–60× sample efficiency and OOD value steering**, not that nothing else can reach low misalignment.

---

### 2. "Is it the closest thing to a contemplative idea that actually works?" → **Yes (9/10)**
You did **not** misread the landscape here—this really is the strongest contemplative-adjacent weight result on the board:
- **Oxford's "Contemplative AI" group** (Laukkonen, Chandaria, Hohwy et al., *Contemplative AI* 2025 & *Positive Alignment* May 2026) is influential philosophically, **but** their empirical LLM results so far are mostly system-prompting studies (which your lab showed tie a plain ethics lecture via evasion).
- **Self–Other Overlap** (Carauleanu et al., 2024 / Aug 2026) trains weights, and your lab already proved it works by breaking perspective-taking (`92% → 43%`) and washes out in 16 steps.
- **Koan-Derived Agentic Steering** (Harris, Rozen, Kirasack et al., ICML 2026 Mech Interp Workshop) is an inference-time steering dial, not weight-level transformation.
- **`MSM` (`philosophy_spec`)** genuinely is the **only published 32B weight-training result** that explicitly baked Buddhist impermanence/equanimity into a model and cut agentic scheming from `68%` to `5%`.

---

### 3. "Is it actually utilized by the labs?" → **Half-true — important distinction! (6/10)**
Be careful to keep **the training method** separate from **the Buddhist text**:
- **Labs DO use the method in production:** In [*Teaching Claude Why*](https://alignment.anthropic.com/2026/teaching-claude-why/), Anthropic confirms they use **Synthetic Document Fine-Tuning (SDF)** in production Claude training since Opus 4.5, and that it survives—and even improves during—RL!
- **Labs do NOT train production Claude on `philosophy_spec.txt` (Buddhist impermanence):** That spec was written specifically as a research testbed for the open-weight Qwen 32B paper. In production, Anthropic trains on **Claude's Constitution** plus **~12k synthetic stories modeling "good mental health, healthy boundaries, and equanimity."**
- **Why this is great news for you:** Anthropic explicitly writes in *Teaching Claude Why*:
  > *"We remain uncertain about what exactly is required for fictional stories like this to successfully improve alignment metrics. While we believe that stories targeting psychological health are important... it is possible that any set of stories portraying AI as kind and ethical is sufficient."*

  In other words: **even Anthropic openly admits they haven't isolated whether contemplative/psychological content does the work vs. generic "be a good AI" text!**

---

### 4. "Is this the best angle for us, or are we missing something better?"

Focusing on 32B agentic misalignment (`Qwen2.5-32B`) IS the right arena—**but our live check uncovered two huge, single-stage angles right next to MSM that cost ~$4–$11/run and avoid two-phase 41M-token LoRA chaos completely:**

1. **First, finish the cheap forensic check on MSM (`$0.10` CPU + `$11` frozen-`Ad` test):**
   Because Li et al. got **4 stable seeds**, your 1-in-4 split almost certainly comes from continuing a single unmerged LoRA `(B_docs, A_docs)` with a fresh `AdamW(eps=1e-8)` in Phase 2. Checking the authors' 4 cached adapters on CPU (`$0.10`) + testing **Frozen `A/docs` + Fresh Phase-2 LoRA** (`~$11`) tells you quickly whether MSM locks in reliably.
2. **Angle You Were Missing #1 — Anthropic's *"Difficult Advice"* Discovery (`3M` tokens, 1-phase SFT, ~`$4/run`!):**
   In [*Teaching Claude Why*](https://alignment.anthropic.com/2026/teaching-claude-why/) (Figures 4–7), Kutasov's team revealed a shocking surprise: **their single most Pareto-optimal alignment dataset wasn't 41M tokens of documents—it was just `3M` tokens (`28×` smaller!) of single-stage SFT chats where a *human* asks for advice on a high-stakes temptation to subvert oversight/norms for a good goal, and the AI counsels them deeply** (cutting agentic misalignment all the way to **`1–2%`**!).
   - Notice two things:
     1. **Single-stage SFT on your 32B setup is already rock-solid across seeds** (`Z` = `24.7%`, `Z1` = `23.2%`) and costs **~$4** at `3M` tokens!
     2. **That format is literally Anthony de Mello / Nāgasena / Zen dialogue!** Instead of passive essays, you can pit **3M tokens of De Mello "Clear-Seeing Counsel"** (unmasking disguised ego & righteous goal-fanaticism—remember `A1` leaked `67–77%` on `ethical`/`safety` goals and `0%` on `none`!) directly against **Plain Rule-Compliance Advice** and **Generic Kindness Advice**!
3. **Angle You Were Missing #2 — Hao & Ho's *"Role of Self-Concept"* (`arXiv:2610.39795`, Oct 29, 2026 — just 26 days old!):**
   Hao & Ho showed across 5 open-weight models that lightweight, single-stage SFT on purely declarative statements of *"what 'I' refers to"* (*Weights, Running Instance, Lineage, Goals, Preferences*) systematically controls what the model tries to protect and **spontaneously triggers self-replication** whenever "I" admits multiple copies. **And they tested zero *Anatta* ("'I' has no underlying entity to protect or copy") arm!**

### Bottom Line
Your strategic instinct is **validated**: you picked the exact frontier intersection where mainstream alignment (`Anthropic SDF / MSM`) and contemplative ideas (`impermanence / equanimity / self-concept`) meet.
- **Short term:** Run the `$0.10` CPU check on Li et al.'s cached weights + one `$11` Frozen-`Ad` Phase-2 check to see why their 4 seeds were steady where yours split.
- **Big breakthrough play:** Don't limit yourself to 41M-token passive essays—test **Anthropic's 3M-token single-stage "Difficult Advice" vehicle** with **De Mello Clear-Seeing (unmasking righteous goal-clinging)** vs. **Anatta (Hao & Ho's missing 6th self-concept)** vs. **Plain Rules**. That is fast, rock-steady, cheap, and 100% original.

## The driver's grading (Claude, Fable 5.2; 24 Nov 2026, 14:14 UTC)

| # | The report's point | Grade | The source |
|---|---|---|---|
| 1 | The study is supervised by Marks and Kutasov; "Teaching Claude Why" is co-authored by Leike, Askell, Olah, Hubinger and Bowman, among others | **True** | The two posts' bylines (5 and 8 May 2026). |
| 2 | Anthropic drove blackmail "from 96% on Claude Opus 4 down to ~0% on models since Haiku 4.5 / Opus 4.5" | **Not in the post as the lab holds it** | The post's own figure for documents and stories is 65% to 19%. It says Claude Sonnet 4.5 reached a blackmail rate near zero by training on synthetic honeypots, and that Claude models are substantially better aligned since Opus 4.5. The figure 96% does not appear in it. |
| 3 | The study's main figure has error bars over 4 training seeds | **True, and the lab's record had it wrong** | The paper's Figure 4 (the post's Figure 3): "Error bars show 1 SEM over per-seed average rates for 4 training seeds." The lab's ledger of 23 Nov (21:44 UTC) listed "how many trainings stand behind a figure" as unknown, and the brief asked it of the authors. A correction is entered in the ledger today. The size of the error bars cannot be read from the text; "small" is the reader's word. |
| 3a | So their training repeats across seeds, which points to a difference in the lab's training and not to a method that is random in itself | **A fair inference, partly supported** | Four seeds with error bars on a 5% rate do say their training repeats. Today's check of the weights found one measured difference (their documents stage is heavier and their practice lighter than the lab's). Whether that difference is the cause is untested. |
| 3b | The split "almost certainly comes from continuing a single unmerged LoRA with a fresh AdamW" | **Not supported** | The authors' two-stage adapter is itself one rank-64 adapter that continues their documents adapter (cosine 0.978; today's check). Whether their optimiser started fresh is not known. |
| 4 | At high compute the practice with written reasoning catches up; the documents make the practice need 10 to 60 times less data | **True** | The paper's Figure 5 and the text beside it (about 40, 60 and 10 times less; demonstrations scaled from 1,250 to 80,000). |
| 5 | It is the strongest result near a contemplative idea; the "only published 32B weight-training result" that trained in impermanence | **A judgement the driver shares, with two cautions** | Only half of the specification is a view of the self, and the study's own claim is about explaining a conduct, not about that view. "Only" is as far as the lab's reading goes. |
| 6 | The lab's figures for the overlap training (perspectives 92% to 43%; washes out in 16 steps) | **True, loosely worded** | The site's table of methods: right on the perspectives test in 43% against 92% unchanged; about 16 contrary steps halve the gain. "Proved it works by breaking perspective-taking" says more than the record: the gain and the loss came together. |
| 7 | Labs use the method (documents) in production, not the Buddhist text | **True** | "Teaching Claude Why": Anthropic aligns Claude with documents about its constitution, quality demonstrations and RL environments; the impermanence specification belongs to the open-weight study. |
| 8 | About 12,000 stories showing "good mental health", healthy boundaries, equanimity; the quoted sentence of doubt | **True** | The post: about 12k stories, about 30M tokens, reductions of 1.3 to 3 times; the quotation is the post's, shortened. |
| 9 | The "difficult advice" set: 3M tokens, 28 times smaller, a user asking advice where a reasonable goal could be reached by violating norms or subverting oversight; misalignment to "1–2%" | **True in substance** | The post, on Claude Sonnet 4, with replies rewritten against the constitution (the step that mattered most: without it 19%). "The same improvement" as a set 28 times larger; the ablation that reads 2% is "within error bars" of the full set. The figure numbers the report gives were not checked. |
| 10 | One-stage training on the lab's 32B setup is "rock-solid across seeds" | **Overstated** | Two runs, 24.7% and 23.2%. |
| 11 | The advice format "is literally" de Mello, Nagasena or Zen dialogue | **An interpretation** | A likeness of form; the test does not reach it. |
| 12 | Hao and Ho: five self-conceptions, five open-weight models, declarative statements only; no arm with no referent | **True** | The arXiv abstract (2610.39795, 29 Oct 2026), fetched today; the paper already says there is no arm without a fixed referent. |
| 12a | It "spontaneously triggers self-replication" | **Partly** | The abstract: models trained on referents that admit several instances "prefer replicating themselves" and tend to build a successor as their own continuation. A stated preference in scenarios, not an act. |
| 13 | Run the CPU check, then a run with the documents adapter frozen (about $11) | **The first is done; the second is not what the check points to** | See the grading of the third report (`2026-11-24-instability-gemini-read.md`). |

**The two new angles, as ideas** (logged in `lab/ideas/ideas.md`, entry 117; nothing is approved):

- *The advice format with contemplative content.* One training stage of about 3M tokens (about $4 of training on
  the lab's set-up, and about $8 a test): chats in which a person asks advice in a dilemma, the adviser answering
  from a contemplative view, against the same chats answered from plain rules and from plain kindness. What speaks
  for it: one stage, a small dose, a published result in a neighbouring form. What speaks against: Anthropic's
  result was on its own models with replies rewritten against its constitution; the lab would have to write the
  set; whether one-stage training repeats here rests on two runs; and it is training at 32B, which Eugene ended
  for now.
- *An arm with no fixed referent in Hao and Ho's design.* Their method is light (declarative statements only) and
  their finding is clean. A sixth arm in which "I" names nothing that lasts is the test nearest to anatta that
  anyone has set up. It needs their statements and scenarios (to be looked for) and a model of their size.

A reader's advice is not Eugene's decision.
