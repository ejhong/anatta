---
name: ledger
description: Append a ledger entry to NOTES.md with a real clock stamp, commit it with a pathspec, push to main (public)
---
Write a ledger entry about: $ARGUMENTS

Procedure (every step, in order; paths relative to the repo root):
1. Run `date -u +"%H:%M:%S UTC"` in its own call. Use that time; never invent one.
2. Append (never rewrite) to `lab/ledger/NOTES.md` with a quoted heredoc (`cat >> ... <<'EOF'`) or Edit.
   Format: a blank line, then `- **<one-line headline>** (the driving model, e.g. Fable, HH:MM UTC on D Mon, read off the clock).`,
   then indented sub-bullets: what ran (run ids, commits, tags), numbers with intervals, spend, limits, and what's
   next. A correction says what it corrects and appends; it never edits an old entry.
3. If it is a registration, use the `register` skill: it must be committed and pushed before any result exists.
4. If it is an experiment result, also append a plain-language `lab/ledger/notebook.md` entry ending with
   "*In plain terms, for the vision:*", and a `lab/explore/SIGHTINGS.md` entry for explore results.
5. Commit with a message file and a pathspec: `git commit -q -F <msgfile> -- <paths>` (trailers: Co-Authored-By and
   Claude-Session lines). Everything committed is public: check nothing personal, secret, harmful or benchmark-item
   is in it. Then `git fetch -q origin`, check `git merge-base --is-ancestor origin/main HEAD`, and
   `git push -q origin main`.
6. If the site's content changed, use the `site` skill. Regenerate `lab/ledger/INDEX.md` with
   `python3 -I lab/ledger/make_index.py ledger` and include it in the commit.
