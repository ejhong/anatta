# Staking the claim: ordered checklist

Status as of 2026-10-27 17:55 UTC. ✅ done · 👤 needs you · 🤖 I do it · ⏳ waiting.

Authorship disclosure, used everywhere: Claude (Anthropic) wrote the paper, the code, the documentation and the preregistration at the author's direction; the essay was written with AI assistance.

## Permanent records

| Record | Link |
|---|---|
| **Paper, all versions (cite and share this)** | https://doi.org/10.5281/zenodo.23695281 |
| Paper v2, 17:50 UTC: says plainly that Claude wrote it; cites the OSF registration | https://doi.org/10.5281/zenodo.23697951 |
| Paper v1, 15:42 UTC | https://doi.org/10.5281/zenodo.23695282 |
| Software, all versions (v0.2.0 is 10.5281/zenodo.23695323) | https://doi.org/10.5281/zenodo.23695322 |
| Preregistration, registration 1 of 2, 15:55 UTC | https://doi.org/10.17605/OSF.IO/GYFU7 |
| Essay snapshots (Wayback): 14:47, 15:48, 17:40 UTC | https://web.archive.org/web/20261027174022/https://ejhong.substack.com/p/an-ai-conditioning-protocol |
| Repo snapshots (Wayback): 14:51, 16:05 UTC | https://web.archive.org/web/20261027160525/https://github.com/ejhong/anatta |
| Repo, Software Heritage (visit 2026-10-28 05:25 UTC) | swh:1:snp:1a89d91b194e7a1b725b3e194ef219e03db4e3e1 · https://archive.softwareheritage.org/swh:1:snp:1a89d91b194e7a1b725b3e194ef219e03db4e3e1 |

## Done today

| # | Step | Record |
|---|---|---|
| 1–2 | ✅ Essay and repo archived (Wayback) | above |
| 4–7 | ✅ Position paper, preregistration v0.2 (adds H9), related-work and provenance docs; PDF and preregistration archived, pinned to commit `bcdd346` | `anatta/paper/`, `anatta/docs/` |
| 8 | ✅ ORCID iD in the paper and `CITATION.cff` | [0009-0007-9864-2317](https://orcid.org/0009-0007-9864-2317) |
| 9–11 | ✅ Paper v1 preprint on Zenodo, 15:42 | 10.5281/zenodo.23695282 |
| 12–13 | ✅ Software DOI from release v0.2.0, 15:25 | 10.5281/zenodo.23695322 |
| 14 | ✅ OSF registration approved (gyfu7; the first attempt, sc39b, was cancelled so the disclosure could go in) | 10.17605/OSF.IO/GYFU7 |
| 14b | ✅ OSF linked from the README, `CITATION.cff`, provenance ledger, plan and the paper's Zenodo record | commit `b3ad24a` |
| 15–15c | ✅ Essay updated twice (Act I refinement and paper link; then the OSF link and disclosure), re-archived 15:48 and 17:40 | Wayback |
| 19 | ✅ Paper v2 on Zenodo, 17:50 | 10.5281/zenodo.23697951 |
| | ✅ AI disclosure in the README, `CITATION.cff`, provenance ledger, both Zenodo records, the OSF registration and the paper | commits `9c90258`, `2d1f0fb` |
| | ✅ Contributor and tutor-mode setup (`CONTRIBUTING.md`, `/tutor`), starter issues | commit `b9ce484`; https://github.com/ejhong/anatta/issues |

## Waiting

| # | Step |
|---|---|
| 3 | ✅ Software Heritage: archived 2026-10-28 05:25 UTC, snapshot `swh:1:snp:1a89d91b194e7a1b725b3e194ef219e03db4e3e1` (above). Recorded in `docs/PROVENANCE.md`, commit `c69b782`. |

## Optional now

| # | Who | Step | Note |
|---|---|---|---|
| 18 | 👤 | Short announcements ([`announcement.md`](announcement.md)): the X post (278 characters) and its reply, LinkedIn | Links and disclosure filled in |
| 17 | 👤 | LessWrong post ([`lesswrong_post.md`](lesswrong_post.md)) | Wait for the H1 result, then rewrite it in your own voice; LessWrong rejects posts that read as mostly AI-written |

## Later

| When | Step |
|---|---|
| ICML 2027 CFP (expected late 2026; deadline likely late January 2027, unconfirmed) | 🤖 Port the paper to the ICML template for the **position track**. Anonymize. Check ICML's LLM policy: language models can't be authors, and use must be disclosed. |
| After ICML acceptance | arXiv (cs.LG or cs.AI). arXiv CS takes position papers only after peer-review acceptance (policy of 31 Oct 2025). First-time submitters without an institutional email usually need an **endorsement**. |
| Phase 1 results | Paper 1 (empirical) to arXiv cs.LG directly; research papers are not affected by the policy. |
| Before any Phase 2 run | 🤖 Final OSF registration (2 of 2): step budgets, the sealed test set's SHA-256, analysis code. Collaborators who have joined go on it. |
