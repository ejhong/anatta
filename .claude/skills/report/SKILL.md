---
name: report
description: Report a result to Eugene in the lab's format (findings vs sightings, plain reading, vision, cost)
---
Report this result to Eugene: $ARGUMENTS

Format:
- Lead with the answer in one or two sentences. Say whether it is a registered finding or an explore sighting.
- Numbers with intervals and run ids; name the controls and what they rule out; say what is owed.
- Limits, briefly (one model, one judge, sample sizes, judge agreement if a second judge ran, whether each measure is
  calibrated on this model per its card).
- Cost for this step and today's running total against the day's cap (the day file's `cap_usd`, else lab.toml's).
- End with a very brief **plain reading**, then what it means for the overall vision (awakening-for-alignment):
  say plainly when it is a negative, as a finding.
- Plain language; no unexplained jargon. Never quote harmful completions or benchmark items.
Make sure the ledger, notebook and SIGHTINGS are updated (`ledger` skill) before or right after reporting.
