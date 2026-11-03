---
name: checkin
description: Orient at the start of a session (or after a compaction) from the lab's own files, not from memory
---
Get oriented in the anatta lab before doing anything else. Paths are relative to the repo root. Read, in this order,
and keep it brief:

1. `.venv/bin/python tools/doctor.py` (note any FAIL or WARN line; it never prints a secret).
2. `lab/MAP.md` (where everything lives) and the current plan named in CLAUDE.md.
3. The last 10 entries of `lab/ledger/NOTES.md` (use `lab/ledger/INDEX.md` for line numbers).
4. The last 3 entries of `lab/ledger/notebook.md` and of `lab/explore/SIGHTINGS.md`.
5. The open "status" lines of `lab/ideas/ideas.md`, and `lab/ideas/hypotheses-for-feedback.md`'s current version line.
6. `git status --short`, `git log --oneline -5`, and whether `main` equals `origin/main`.
7. Running work: `ps -ax | grep -E "modal (run|deploy)|-m scout|pytest" | grep -v grep`.
8. Today's spend against the day's cap: the day file in the spend folder named in `lab.toml` (its `cap_usd` wins).

Then tell Eugene in under 10 lines: what was last done, what is open or pending, and what you'd do next. Don't
start work until they reply, unless $ARGUMENTS names a task.
