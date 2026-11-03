---
name: site
description: Rebuild, test and publish the lab site (Anatta So Far, served on GitHub Pages) after a content change
---
Rebuild the lab site. $ARGUMENTS

1. Edit only `lab/site/so_far/content/*.html` (never the built index.html by hand). Note the change in
   `content/footer.html`.
2. `cd lab/site/so_far && python3 -I build.py`, then `python3 test_build.py` (all tests must pass).
3. Tally: `python3 -I lab/tools/tally.py lab/site/so_far/index.html` must equal the count in the latest ledger entry
   that changed it (currently hit 43, miss 73, open 7); if not, stop and find out why.
4. Republish with the Artifact tool using the same file path `lab/site/so_far/index.html` so the URL stays
   https://claude.ai/artifact/LG4bhC9WV2k9RpqHNooNAx.
5. Commit content/ and index.html with a pathspec and push main (see the `ledger` skill, step 5); the pages workflow
   then redeploys https://ejhong.github.io/anatta/ (check with `gh run list -R ejhong/anatta -w pages -L 1`).
Outward text keeps personal material out, never attributes recognition or esoteric status to anyone, and keeps the
contemplative source named.
