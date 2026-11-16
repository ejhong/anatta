---
name: explore
description: Run a cheap explore sweep that surfaces sightings (not findings), within the sweep's and the day's spend caps
---
Explore: $ARGUMENTS

Explore is the fast speed of the research loop: cheap, many, and honest that results are sightings.

**How much paperwork** (Eugene's yes of 16 Nov 2026 to Fable's proposal; `lab/ideas/ideas.md`, entry 76). At this
speed there is no design note, no addendum, no stage gate written into code and no ledger entry per stage. There is:

- **Five lines before the run**, committed before anything is seen: the question, the one comparison, what counts as
  a difference, the one control, the cost cap. They go at the top of the step's script or in a short file beside it.
- **A small try first**, in which the driver opens raw replies and reads them, and counts, blind to labels, the
  answers that look misread (a reader that misreads numbers is not caught by a try that hides them).
- **The run, then one ledger entry, one notebook entry and one SIGHTINGS entry.** Changes made on the way are said in
  that ledger entry.
- **A cold check by a fresh Opus agent, on the design, only** when a sighting is promoted, or when a new design's
  negative will be stated on the site (a negative is worth something only if the test could have seen a yes).

Full registration (`register`) stays for promotion and for claims. The hard rules in CLAUDE.md are not paperwork and
do not change.

1. Say what would make the sweep interesting or dull before running it (it is part of the five lines).
2. Use the existing tools: scout (`lab/explore/scout`, OpenRouter models, dev split only; the held-out split is never
   read in explore), the steering bench (`lab/explore/bench`), or a spec under `experiments/explore/` once the runner
   exists. No new one-off script when a tool can do it.
3. Set a sweep cap in the spec; the day's cap comes from `anatta.spend` (day file's `cap_usd`, else lab.toml). Check
   today's total first. Every paid call goes through SpendGuard or the Meter.
4. Include the matched controls the tool asks for (random-matched directions, length-matched texts, a baseline).
5. Read results by numbers and intervals; never print harmful completions or benchmark items.
6. Write a `lab/explore/SIGHTINGS.md` entry (what looked alive, with intervals, and the next step: replicate, drop or
   promote) and a notebook entry. Promotion to a claim goes through `register`.
7. Report cost and what it means for the vision (`report`), labelled as a sighting.
