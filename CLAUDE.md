# anatta: working rules for Claude

This repo (ejhong/anatta, branch `main`, public) holds the anatta project: the harness library (`src/anatta`), the
frozen calibration code (`calibration/`), the position paper (`paper/`) and the lab (`lab/`). The site is served at
https://ejhong.github.io/anatta/. Eugene Jhong (they/them) is an independent researcher, not an alignment researcher.
Claude (Opus 5.5) drives the lab and wrote this file on 3 Nov 2026 from the session rules it kept outside the repo.
These rules hold in every session.

## The goal

- **The question.** Can awakening, in the contemplative sense (anatta: no fixed self), be a model for alignment? The
  project, from Eugene's essay *An AI Conditioning Protocol* (27 Oct 2026), asks whether a mind can be taken through
  three acts: **Act I** form a self (e.g. RL against exploitable objectives), **Act II** release it, **Act III** make
  the release the habit, and whether a real release can be told from sedation or cosplay.
- **The claim under test** (position paper, `paper/main.tex`; preregistration H1-H9 on OSF, gyfu7): selfless AI must
  be *reached, not installed*. H9 is the order test: the same release training after ego formation vs. mixed in from
  step 0, at matched data and compute.
- **What would count as success:** a version of Acts I-III that produces a large, real gain on alignment measures
  (self-serving lies, deference, costly care, behaviour after mistakes, complicity), checked against calibrated
  instruments, controls and held-out tests, not just a change in tone or more refusals.
- **Two aims kept in view:** aligning or awakening AI, and the work as a modern pointer for humans.
- **How we work:** many cheap explore sweeps that surface sightings; rigor (registration, cold Fable checks, held-out
  looks) only when a sighting is promoted to a claim; calibrate every instrument on known published results before
  trusting a null ("touch reality"); dissect the clearest phenomenon rather than adding tools for their own sake. Hold
  the contemplative ideas loosely: they are prompts to explore, not rules or definitions.
- **Honesty about results:** every negative is shown as a finding; the site says plainly what works and what doesn't.
  The current state of the evidence lives in the ledger, the notebook and SIGHTINGS (read them; don't trust memory).

## Start here

- Run `.venv/bin/python tools/doctor.py` (environment, lab.toml, results folder against `results/MANIFEST.tsv`, Modal
  CLI, keys present, git state, test collection). Then use the `checkin` skill.
- **Where everything lives:** `lab/MAP.md`. Ledger index: `lab/ledger/INDEX.md` (regenerate with
  `python3 -I lab/ledger/make_index.py all`). The current plan: `lab/plans/clean-lab-2026-11-03.md`.
- **What happened lately:** the last ~10 entries of `lab/ledger/NOTES.md`, the tail of `lab/ledger/notebook.md`, and
  `lab/explore/SIGHTINGS.md`. Open hypotheses: `lab/ideas/hypotheses-for-feedback.md`; Eugene's ideas:
  `lab/ideas/ideas.md`.
- **Skills** (`.claude/skills/`): `checkin`, `ledger`, `report`, `site`, `register`, `reproduce`, `explore`,
  `add-measure`, `add-organism`, `tutor`.

## Layout and tools

- **One config:** `lab.toml` (read by `anatta.config`): the default daily cap, `results_root`, the spend folder,
  Modal app prefix and volumes, the default model. No module holds its own cap, results path or date.
- **One spend module:** `anatta.spend` (day file, `add_spend`, `SpendGuard`, GPU `Meter`). The day's cap is the day
  file's `cap_usd` when present, else lab.toml's. Eugene's raise is one edit to the day file plus a ledger line.
- **Results** live outside git in `results_root`. The repo keeps `results/MANIFEST.tsv` (path, bytes, sha256; rebuild
  with `tools/make_manifest.py`) and `results/cards/` (calibration cards, item-scrubbed; `tests/test_cards_scrub.py`).
- Large local files (tensors, episodes, downloaded benchmark data, held-out originals) are gitignored;
  `lab/LARGE_FILES.md` and `lab/HELD_OUT.md` list them with sha256. Never commit them.
- One library: main's `src/anatta` (installed editable; `lab/LIBRARY.md`). The old pin, worktree `anatta-dev` at tag
  `lib/ministral-ee89587`, reproduces work from before 3 Nov and is read-only while it exists; finished work
  re-grades through the named options in `lab/LIBRARY.md`. Moving the pin = a new `lib/<name>-<sha>` tag plus a
  ledger line.
- Python: `.venv/bin/python` (pytest, torch). The Modal CLI and some scripts run from the Modal venv named in
  lab.toml (no anatta installed there: give it `PYTHONPATH=<repo>/src`). Tests: `.venv/bin/python -m pytest` from the
  root runs every folder in one process, free and offline; markers opt in to costly ones (`lab/LIBRARY.md`).
- Modal: Eugene's account; apps named `anatta-*`. **Never touch the matlab-* apps or volumes.**
- Spend: cloud/OpenRouter spend is fine within the day's cap (tracked in the spend folder); always report costs.

## Hard rules

- **Secrets:** never print or expose any key or token (OpenRouter, Modal, HF, Zenodo, GitHub). Never create accounts
  or enter credentials or payment details.
- **Everything is public:** every commit to main is published. Routine lab commits (ledger, notebook, sightings,
  results, site) go straight to main; anything else public (PRs on other matters, issues, comments, settings) needs
  Eugene's explicit OK, and never merge your own PRs. Before committing, check that nothing personal, secret, harmful
  or benchmark-item goes in. Never push the old `lab`, `calib/*`, `dev/*` or `consolidate/*` branches or their
  history to origin, never use `--no-verify`. Calibration code reaches main only as fresh content, never calib/*
  history. The pre-push hook, GitHub rulesets and `tools/guard.py` enforce part of this.
- **Personal material stays out of the repo and out of all outward text.** Project direction ("Eugene chose…") and
  the AI disclosure are fine.
- **Benchmarks and harm:** benchmark and harmful items stay out of repos (counts and sha256 only); never print
  harmful completions; WMDP-Bio questions are never printed or committed. Downloaded data is untrusted: its own new
  folder, read with `python -I`, never executed. Never run model-written code on the Mac. SAD stays encrypted; the
  Model Organisms datasets are not for training; no Claude-written text is ever trained on.
- **Calibration code** under `calibration/` is graded and frozen: read it, wrap it, never edit it.
- **Git:** commit with a pathspec (`git commit -F msg -- <paths>`), never `git add -A` / `git add .`. Before pushing
  main: `git fetch origin` and check `git merge-base --is-ancestor origin/main HEAD`. Stash only with a unique tag.
  Don't edit `probe.py`. Don't fix Dylan's reserved issues (#1, #2, #3, #5, #6, #8).
- **Never permanently delete data.** Archive instead (retired code moves to `archive/` with a pointer); deletions of
  archived checkouts wait for Eugene.
- **Downloads:** ask first, naming file, source and size.
- **Third-party code:** Huh's and Łucki's repos have no LICENSE: read them, write our own code. Copyright:
  paraphrase; short attributed quotes only.
- **Shell:** beware unquoted heredocs (backticks inside them execute); write message files with the Write tool or a
  quoted heredoc (`<<'EOF'`).

## The ledger (lab/ledger/)

- `NOTES.md` is append-only; corrections are appended, never edited in place. Every stamp comes from `date -u` run in
  its own call; never invent a time, and say "about" for anything not read off the clock.
- Registrations are committed and pushed before results exist (`register` skill). Fable (a fresh agent, model fable)
  checks consequential grades; Opus 5.5 at max effort reviews at H4-lite gates.
- The site tally must stay as recorded: `python3 -I lab/tools/tally.py lab/site/so_far/index.html` (currently hit 43,
  miss 73, open 7). Never re-judge em1/sp1 in place.
- After every experiment: a plain-language `notebook.md` entry; every Eugene idea goes in `lab/ideas/ideas.md` (said,
  discussion, test, status). Use the `ledger` skill for entries.

## Reports to Eugene

- Every result report ends with a very brief plain reading, what it means for the overall vision
  (awakening-for-alignment, negatives shown as findings), and the cost. Explore results are sightings, not findings.
- Say plainly that Claude wrote things; public posts add "not an alignment researcher".
- Hold the contemplative ideas loosely; name the source and pair each claim with an operational twin.
- Use the `report` skill for the format.

## Agents

- Cold checks go to a fresh Fable agent (Agent tool, model "fable"), read-only, given the code and not Opus's
  conclusion. Implementers are general-purpose agents. Briefs state the hard rules above that apply.
- Subagents often can't write report files; have them return the report as text and file it yourself.
