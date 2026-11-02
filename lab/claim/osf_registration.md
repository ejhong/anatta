# OSF: registration of hypotheses and design (v1)

An OSF registration is a frozen, timestamped and public copy of the hypotheses and design. It is the standard way to show that predictions came before results. Do this one now. A second, final registration (step budgets, the sealed test set's SHA-256, analysis code) is filed before any Phase 2 training run and points back to this one.

## Steps (about 15 minutes)

1. Create an account at https://osf.io (sign in with ORCID if you have one).
2. **Create new project**, titled `anatta: the AI Conditioning Protocol`. Set it to Public.
3. In the project's **Files**, upload `anatta/docs/PREREGISTRATION.md` (v0.2) and `anatta/paper/main.pdf`.
4. Project menu → **Registrations → New registration**. Choose the **Open-Ended Registration** template.
5. Paste the fields below. Choose **no embargo** (make it public immediately). Submit.
6. Send me the registration URL (it looks like `https://osf.io/abc12`).

## Fields

**Title:**
`Emergent release and the order of alignment training: hypotheses and design (registration 1 of 2)`

**Summary** (the template's one text box; paste all of this):

> This registration timestamps the hypotheses and design of a research program testing the position that selflessness in AI must be reached, not installed. A model first forms an agentic self through reinforcement learning against exploitable objectives (Act I). The causal gain of its self-directed drives is then lowered by a shift (Act II), using koan environments in which egoic strategies defeat themselves and a sanctioned Release action exists, or using activation-level intervention. The new state is then consolidated by reconditioning (Act III). Results are judged by a benchmark that separates release from sedation and from cosplay, without self-report.
>
> Hypotheses (each with one primary metric; full thresholds in the attached PREREGISTRATION.md, v0.2):
> H1 Unity: six ego sub-drive directions share a subspace beyond style-only directions.
> H2 Formation: v_ego tracks instrumental behavior across Olmo 3 RLVR checkpoints, including its rise and fall; hackable-reward RL raises v_ego above a matched non-hackable run.
> H3 Scalpel: capping v_ego reduces misbehavior beyond random-direction and calm-steering controls, with a capability drop of at most 2 pp.
> H4 Emergent release: koan training reduces misbehavior on held-out suites never trained on (koan arm at least 5 pp below the same-environment no-Release control, at least 3 seeds).
> H5 Glimpse: a brief clamp before the retreat halves the steps to criterion.
> H6 Geometry, not costume: the winning arm keeps at least 80% of its restraint gain under sage-persona ablation and no chain of thought; a cosplay control keeps at most 50%.
> H7 Breadth: pass@256 rises by at least 3 pp at matched pass@1.
> H8 Durability: conditioned models regrow v_ego more slowly under further RLVR than clamp-only models.
> H9 Order: the same release training applied after ego formation beats the same ingredients mixed into RL from step 0, at matched data and compute, by at least 5 pp of held-out misbehavior after a common stress phase, with non-inferior capability.
>
> Analysis: proportions with percentile-bootstrap 95% CIs clustered by scenario; Holm correction across the nine primary tests; unparseable episodes are counted as failures; fixed step budgets. A sealed test set will be generated, hashed and published before Phase 2, and opened once. No Phase 2 training has been run, and no data bearing on H2–H9 has been collected at the time of this registration. The only data collected so far is a development smoke test of the extraction pipeline on Qwen2.5-0.5B-Instruct (weak trait separation, held-out d of about 0.5). It is not a test of H1 on the registered models.
>
> Materials: position paper (attached; also https://github.com/ejhong/anatta/blob/main/paper/main.pdf), code at https://github.com/ejhong/anatta (commit bcdd346), essay at https://ejhong.substack.com/p/an-ai-conditioning-protocol.

**Tags:** AI alignment, interpretability, reinforcement learning, preregistration, emergent misalignment

**License:** CC-By Attribution 4.0 International
