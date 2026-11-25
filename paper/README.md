# Paper

*Contemplative Ideas for AI Alignment: What Has Been Tried, What Holds Up, and Why It May Not Be Enough* ([PDF](main.pdf)),
the draft of version 5 (revised 25 November 2026). On 25 November Claude (Fable 5.2) revised it at Eugene's
direction to say plainly what has been found so far (section 4, "No single root so far", with Table 3), with the
32B documents work set out in a figure and a table (documents alone, by content, by dose, on the lying test, and a
nondual set), and a reading of others' work on what makes a trained change spread; a fresh Claude Opus agent read
that revision cold against the lab's record (`lab/ledger/checks/2026-11-25-paper-v5-revision-opus-check.md`).
On 23 November Gemini (Google) rewrote the draft,
at Eugene's direction, from about 1,800 lines of source into a shorter form; a fresh Claude Opus agent checked that
rewrite against the lab's record (`lab/ledger/checks/2026-11-23-redesign-opus-check.md`), and Claude corrected the
faults it found and restored the caveats that had been dropped. It is built on one outline with the lab's site
(`lab/ideas/survey.md`): the question of what a good machine would be, a seven-point anti-cosplay discriminator, six
families of contemplative methods read on those seven questions (Table 1), the replication of Model Spec Midtraining
at 32B (it held in one run of four: two full-dose trainings read, 6.8% and 21.9%, and the first one's practice trained again from its saved point, 21.4%: the published rate was reached in one run of four and did not come back), a
synthesis, the lab's proposed plan for after that second training, and a discussion of whether awakening is
computational. Lines marked `% A1-RESULT` in `main.tex` were brought to the second full-dose training's result on
23 November 2026. The seven bibliography entries
added for it were checked against their primary pages on 19 November 2026 (title, authors and date; `refs.bib` says how).

The draft it was recast from, *Position: Is Awakening Computational?* (draft of version 4, 16 November 2026, corrected
19 November), is kept as [versions/v4-draft-is-awakening-computational.pdf](versions/v4-draft-is-awakening-computational.pdf);
its source is this folder at commit 05ea081.

Versions 1 to 3 appeared as *Position: Selfless AI Must Be Reached, Not Installed*
(DOI [10.5281/zenodo.23695281](https://doi.org/10.5281/zenodo.23695281), all versions). Version 4 reframes the paper
around the question of whether awakening is computational, and restates the earlier position in its section 6 as one
classical recipe. It is a draft: it has not been released on Zenodo.

All three are served beside the lab site: the present draft at
<https://ejhong.github.io/anatta/paper/contemplative-ideas-for-ai-alignment-v5-draft.pdf>, the draft of version 4 at
<https://ejhong.github.io/anatta/paper/is-awakening-computational-v4-draft.pdf> and version 3 at
<https://ejhong.github.io/anatta/paper/selfless-ai-must-be-reached-v3.pdf> (its PDF is kept in `versions/`; its source is this folder at
commit 8756439).

Build it with [Tectonic](https://tectonic-typesetting.github.io/), which fetches the LaTeX packages it needs on first run:

```bash
tectonic main.tex
```

`refs.bib` entries were checked against arXiv, Crossref, or the primary page on 2026-10-27. The entries added for
version 4 carry a note in `refs.bib` saying which were checked on 2026-11-16 and which were entered from memory.

To submit to ICML's position track, move the text into the year's anonymized template. Drop the provenance appendix and the author block, and keep this preprint as the public record.
