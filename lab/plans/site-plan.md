# The main site: a plan

Written by Opus 5.5 at about 20:00 UTC on 31 Oct 2026, after Eugene's request: "make sure you are working on also how
to present very beautifully in a site all the ideas and things of stuff we are doing and trying to achieve on the main
site." It extends ideas entries 29, 45 and 46. Claude wrote this for Eugene Jhong.

## What the site is for

The main site is *Anatta So Far*, the private page at claude.ai/artifact/CZmnYwpZpU6W3XNRJoqhtL. It becomes one place
that shows four things:
- what the program is trying to achieve: awakening as a model for alignment, held loosely;
- every idea it has had, and where each stands;
- everything it has done (experiments, calibration, and the negatives as findings);
- what comes next.

Eugene reads it first, and later the people Eugene shares the link with. The draft is private.

## Principles, from the standing preferences

- Beautiful and calm, and built from the lab's files, so it can't drift from the data.
- The awakening frame stays the aim. What works and what doesn't are shown side by side, and every negative is a finding
  with what it rules out.
- Each idea names its contemplative source and is paired with its operational twin, the thing that could be measured.
- Ideas are held loosely: each is a conjecture, a test and a status, not a rule.
- No personal attribution. The site never says where an idea came from in anyone's experience. It doesn't quote
  Eugene's private messages, and it uses nothing from the private account folder.
- It says plainly that Claude wrote it, and that Eugene is an independent researcher, not an alignment researcher.
- Every change to Eugene's essay is noted in the footer.

## Shape: one page, seven rooms

The site is one artifact with hash-routed views, so each room has its own link.

1. **#aim:** the aim in a paragraph and the three acts in a figure:
   - Act I, a mind with a self;
   - Act II, the root cut while the self stays usable;
   - Act III, conditioning afterward.
   It also shows where things stand at a glance (holds, failed, open), the frame's five tests, a strip of the
   instruments' checks, and the next three steps.
2. **#ideas:** the contemplative ideas, organized as the current page's part one is, and every entry of the ideas log
   as a card. A card gives the idea in a sentence or two, its source, its test, its status and links to the experiments
   that tested it. One row of filter chips sorts by status and by theme.
3. **#possibilities:** after awakening, the physics-of-mind possibilities: classical, a spark, a light that keeps
   recognizing, a living contact, and ingression. It covers what "good" would mean in each and what experiments can and
   can't settle. Added 31 Oct at Eugene's request (ideas entry 48); its content is hand-written in
   content/possibilities.html.
4. **#experiments:** each experiment as a card: its question, its design in a line, its result, its verdict, what it
   rules out or points to, its cost and its check. Cards are grouped by question: the lie, the self, release and
   transfer, durability, emergent misalignment, and flattery.
5. **#instruments:** the calibration area.
6. **#record:** the scorecard, the specimens, and how the ledger works (registration before data, cold checks,
   errata).
7. **#next:** the plan.

The footer holds the essay changes, the disclosure and the links.

## Look

- Keep the current identity and extend it: paper and ink, verdigris and copper, Newsreader, Literata and Plex Mono, one
  book column with a margin rail.
- New on #aim: a map. The sources (the Upanishads, the Pali canon, the Zen records, living reports) sit on the left,
  the ideas in the middle, the experiments and their verdicts on the right. The lines are colored by verdict and labeled
  in words, so color never carries meaning alone. It is inline SVG and works in both themes.
- Cards carry a quiet status mark, and the filters sit in a single row of chips.
- Designed at phone width first. Dark mode is designed in its own right, not inverted.

## Built from files

lab/site/so_far/build.py writes the page from:
- lab/ideas/ideas.md: each entry's title, discussion, test and status. It never uses the "Eugene said" lines or the
  index's "Raised" column;
- hand-kept content files for the narrative, carried over from anatta_so_far.html;
- experiments.json, one card per experiment, seeded from the current page;
- the calibration cards;
- scorecard.json, the scorecard table moved out of the HTML, still recounted by tally.py.

One command rebuilds the page. A test checks that every idea in ideas.md appears and that no private source is read.

## Phases

1. **Draft:** #aim and #ideas are built new. The other rooms are carried over from the current page's sections. It is
   published at a new private link for Eugene to look at.
2. **After Eugene's notes:** experiments become cards, and the map and filters arrive. Then the draft replaces the
   current page at its link.
3. **Later:** per-experiment re-run commands and result cards (ideas entry 46), and the lab-website features (ideas
   entry 45).
