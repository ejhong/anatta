# Zenodo: preprint deposit for the position paper

The goal is a permanent DOI for the paper, timestamped today. Zenodo is run by CERN, and its DOIs never expire.

## Step 1: reserve the DOI first (so it can be printed in the PDF)

1. Go to https://zenodo.org, choose **Log in → Log in with GitHub** (or ORCID if you have one by then).
2. Click **New upload**.
3. Under **Digital Object Identifier**, answer "No" to "Do you already have a DOI?", then click **Get a DOI now!**. Zenodo reserves a DOI like `10.5281/zenodo.1234567`.
4. **Send me that DOI.** I'll put it in the PDF footnote, rebuild, and push. Leave the draft open; it is saved automatically.

## Step 2: fill in the form (copy-paste)

| Field | Value |
|---|---|
| Files | `anatta/paper/main.pdf` (the rebuilt one with the DOI) |
| Resource type | Publication → **Preprint** |
| Title | `Position: Selfless AI Must Be Reached, Not Installed` |
| Publication date | `2026-10-27` |
| Creators | Jhong, Eugene. Affiliation: Independent researcher. ORCID: `0009-0007-9864-2317` |
| Description | the abstract below |
| License | **Creative Commons Attribution 4.0 International** |
| Version | `v1` |
| Language | English |
| Keywords | AI alignment; developmental alignment; emergent misalignment; emergent release; activation steering; interpretability; persona; reinforcement learning; contemplative AI; preregistration |
| Related works | **Is supplemented by** → URL `https://github.com/ejhong/anatta` (resource type: Software) |
| | **Is derived from** → URL `https://ejhong.substack.com/p/an-ai-conditioning-protocol` (resource type: Publication / Other) |
| | Later, add **Is supplemented by** → the OSF registration URL, and the software DOI once minted |
| Funding | leave empty |

**Description (abstract):**

> This position paper argues that selflessness in AI systems must be reached, not installed. By selflessness we mean a measurable property: the model's representations of itself exert little causal pull on what it does, while its self-knowledge and competence stay intact. Contemplative traditions and transpersonal psychology describe a three-act arc for humans, summarized as "you have to be somebody before you can be nobody." We transpose it into an engineering sequence. In Act I, reinforcement learning against exploitable goals forms an agentic self, together with instrumental drives that generalize into misbehavior. In Act II, a shift lowers the gain of those drives, either through koan environments in which every egoic strategy defeats itself and letting go is the sanctioned exit, or through activation-level intervention. In Act III, reconditioning consolidates the new state so that it survives further optimization. We give three reasons the order matters: competence (agency needs a self-model), selection (installing selflessness early selects a sage persona over unchanged goal geometry, which we call cosplay), and generalization (if misbehavior generalizes from narrow reward hacking, release may too). We state nine falsifiable predictions, including a direct test of order, and specify a benchmark that separates four end states (ego, sedated, cosplay and released) without relying on self-report. We release an open harness and a preregistration, and we say which results would refute the position.

## Step 3: publish

Click **Publish**. A published record can't be deleted, but you can always add new versions (v2, v3) under the same concept DOI. Send me the record URL and I'll add the DOI badge to the README, the citation file and the provenance ledger.

## Separately: the software DOI (automatic)

1. In Zenodo, open your account menu → **GitHub**, click **Sync now**, and flip the switch **ON** for `ejhong/anatta`.
2. Tell me. I'll cut release `v0.2.0` on GitHub, and Zenodo mints a software DOI from `CITATION.cff` within a few minutes.

The switch must be ON *before* the release is created, or no DOI is minted.
