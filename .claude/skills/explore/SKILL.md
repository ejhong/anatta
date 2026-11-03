---
name: explore
description: Run a cheap explore sweep that surfaces sightings (not findings), within the sweep's and the day's spend caps
---
Explore: $ARGUMENTS

Explore is the fast speed of the research loop: cheap, many, and honest that results are sightings.

1. Say what would make the sweep interesting or dull before running it (one line in the spec or the sweep's notes).
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
