# Anatta So Far: the main site

*Anatta So Far* is the program's main page: what it is trying to do, how it takes the self apart, every idea in the lab's ideas log, what the experiments showed, how the instruments were checked, the record of graded predictions, and what comes next. It is one HTML page with eight rooms, each opened by its hash: `#aim`, `#self`, `#ideas`, `#possibilities`, `#experiments`, `#instruments`, `#record` and `#next`. The page used to be hand-edited as `lab/site/anatta_so_far.html`. It is now built from files, so the ideas log on the page can't drift from `lab/ideas/ideas.md`.

Claude (Anthropic) wrote the builder, the tests, the new rooms, the design and this README for Eugene Jhong. Eugene is an independent researcher, not an alignment researcher, and the page's footer says so.

## Rebuild

From the worktree root:

```
python3 lab/site/so_far/build.py            # writes lab/site/so_far/index.html
python3 lab/site/so_far/build.py --dump     # prints what was read from the ideas log, and every omission
python3 lab/site/so_far/test_build.py       # the checks (or: python3 -m pytest lab/site/so_far/test_build.py)
```

Standard library only. Rebuild after any change to `lab/ideas/ideas.md` or to a file in `content/`. One test fails while `index.html` is stale, as a reminder.

`index.html` follows the Artifact page contract: no `<!doctype>`, `<html>`, `<head>` or `<body>`; it opens with the title, the viewport-fit meta, the Google Fonts link and the style, and ends with one inline script. It loads nothing else: no external scripts, images or `fetch()`, and nothing depends on browser storage. Publishing it is a separate step, done by whoever reviews it.

## The design

Gold ink on indigo paper, after the *konshi kinji* sutra scrolls. In light mode the page is cool unbleached paper with indigo ink and an ochre gold; in dark mode it is the indigo sheet itself. Gold is kept for numerals, key terms, hairlines and the chariot. Headings are set in Shippori Mincho B1, the text in Spectral and labels in DM Mono. Each section has a rail: its number in gold and its label set on its side, beside a gold ruled line, with the text in one column of about 66 characters and the margin free for the Limits boxes. Every color is a token on `:root`, with the dark values set twice (for the system setting and for an explicit choice), and a test fails on any color written outside those blocks. Status is always a word with a shape beside it (a disc for working, a ring for failed, a dash for open), never color alone; the field of marks and the chariot use the same shapes.

## Where each part comes from

| Room | Files in `content/` | Generated or carried over |
|---|---|---|
| Aim | `aim.html` (template), `hero.html` | **Assembled.** The hero is carried over, its first sentence set apart as the thesis, with a figure of Indra's net drawn at build. New for this site: four reading paths, a key to the two kinds of statement (the frame and a result) and a door to the Self room with a small drawing of the chariot. The three acts use the page's own wording. The glance columns, the five-test strip, the instrument rows and the next three steps are parsed from `standing.html`, `frame.html`, `calibration.html` and `plan.html`, so editing those updates Aim. |
| The self | `self.html`, `self_prh.html`, `self_work.html` | **Written for this site** (1 Nov). The chariot (the word "I" drawn from ten parts and a binder) with one card per part, the idea beside its operational twins, the Platonic link with a mutual k-NN toy, what published work shows, what the lab has done, what comes next with the K1 chart, and the sources. The seven Platonic cards ("Seven bridges") moved here from Possibilities unchanged. |
| Ideas | `ideas_organized.html`, `ideas_log.html` (template) | Part one, "The ideas, organized", is carried over. Two of its cards link to Possibilities. The ideas log is **generated** from `lab/ideas/ideas.md` every build: one card per `## N.` entry. |
| Possibilities | `possibilities.html` | Written for this site (31 Oct). It holds the physics-of-mind possibilities and what "good" would mean after awakening. Its old section E moved to the Self room; the lede and the "Made or found" card now point there, and the old section F is now E. |
| Experiments | `experiments.html`, `learned.html` | Carried over, with a Limits box beside each of the six results and glossary links on its terms. The cut chart's caption calls the timidity bar red, its new color. |
| Instruments | `instruments_head.html`, `calibration.html` | Carried over, with a short head whose lede is made of sentences from the page. |
| Record | `record_head.html`, `scorecard.html`, `specimens.html`, `caveats.html` | Carried over, with a short head as above. Above the scorecard table, the **field of marks** is drawn at build from the table: one mark per graded prediction, grouped by result or in table order. |
| Next | `plan.html` | Carried over. |
| Footer | `footer.html` | The glossary (written for this site), then the disclosure, the links and the essay changes (carried over), plus one generated line saying how the page is built. |

Also in `content/`:
- `site.css` is the whole stylesheet. The old `base.css` is retired.
- `site.js` does the routing, the glossary popovers, the chariot, the k-NN toy, the field's regrouping and the log's filter.
- `data.json` and `figures.js` are the old page's chart data and drawing code, restyled; the builder wraps the drawing code in `drawFigures()`, which redraws when the theme changes.

The carried-over files keep the old page's prose and numbers. Only markup changed, plus the Limits boxes and glossary links named above. Two ids belong to rooms, so the old sections were renamed: `ideas` became `ideas-organized` and `experiments` became `experiments-part`. The eyebrows "Part one" to "Part four" read Ideas, Experiments, From the result files, In plain terms and Next.

### Drawn at build

- **The chariot.** `build.py` reads the part cards in `self.html` (`<article class="part-card" id=... data-n=... data-status=... data-short=...>`), expects parts 1 to 10 and then the binder, and cuts a capital I into ten pieces, one per part, in card order. A part's status (`tested`, `known` or `untested`) decides how its piece is drawn when the letter is taken apart: solid, outlined or dashed. Change a card's status and the drawing follows.
- **The K1 chart** is drawn from the targets table in `self_work.html` (`<table class="k1">`).
- **The field of marks** is drawn from the scorecard table, counted the way the scorecard's tally is counted; the build stops if the two disagree. A hit followed by "flagged" in the table is drawn half filled.
- **Indra's net** on Aim is drawn at build too, and the small gold I in the band is the chariot's letter.

### The ideas log

For each entry the builder shows:
- the number and the title;
- a summary: the first one to three sentences of the Discussion's first bullet, unparaphrased (entries 16 and 21 have no Discussion, so they use "Reading" and "Where we are");
- the Test line;
- the latest Status or Update line, or the index's "Where it stands" cell when an entry has neither (14, 15 and 16).

Clicking a card opens its full discussion, rendered from the markdown. A Test or Status written as sub-bullets shows its first line on the card and the whole list inside.

The filter chips group entries by status: Tested, Running, Planned, Adopted and Noted. The group is read off the status line by the rules in `GROUP_START` and `GROUP_RULES` in `build.py`, and the page's key explains each one. Status is always written as a word, never shown by color alone.

### Old links

The single-page anchors still work. `#calibration`, `#plan`, `#standing`, `#frame`, `#durability`, `#scorecard` and `#specimens` all open the room that now holds them (see `MOVED` in `site.js`). An id inside a room, such as `#idea-12`, a finding's `#gl-...` or a part's `#part-vego`, opens that room, opens the card and scrolls to it. A glossary entry (`#g-...`) scrolls to the footer and leaves the room as it was. If scripts don't run, CSS still shows the targeted room, or Aim when there is no hash, and the chariot is drawn taken apart with every card showing.

## Privacy rules the builder enforces

- It never opens anything under the private account folder (now kept outside the repo) or `lab/experiments/recognition/`. `read_text()` refuses those paths, and a test traces every open and directory listing during a build.
- It drops every bullet in `ideas.md` that starts with `**Eugene` (these quote private messages), together with its sub-bullets. A test checks that no run of six words and no quoted phrase from those bullets reaches the page. The generated log, the Possibilities room and the Self room are held to this strictly.
- From the index table it reads only the number and the "Where it stands" cell, never "Raised".
- It leaves out any sentence that refers to "the account" (the contemplative account), to someone's own experience or recognition, or that quotes a private message. When the reference sits inside a parenthesis, only the parenthesis goes; a label that names the account drops its whole bullet. `--dump` lists every omission, with the reason.
- Nothing on the page attributes recognition, awakening or any esoteric status to anyone. A test checks the whole page for wording that would attribute such a status to Eugene.

One thing the tests allow on purpose: a few runs of words in carried-over prose (Aim's Act I text, part one of Ideas, one sentence in Experiments) are shared with the private bullets. The hand-kept page put some of the same ideas in nearly the same words. They are listed in `CARRIED_OVER_SPANS` in `test_build.py`, and any run outside that list fails. No text on the page names either private folder, and `CARRIED_OVER_PATHS` is empty.

## Changing things

- **A new idea or a status change:** edit `lab/ideas/ideas.md` and rebuild. The card, the count and the chips follow.
- **A finding's standing:** edit `content/standing.html`. Aim's glance columns and the "where it stands" lists under the acts follow. Aim names findings by title (`{{stands:Title}}` in `aim.html`), so if you rename a finding, rename it in `aim.html` too; the build stops with a clear error otherwise.
- **A part of the self:** edit its card in `self.html`. Its five rows (Source, Twin, Known, Here, Next) are checked by the tests; its `data-status` and the status word in its header must agree.
- **A graded prediction:** edit the scorecard table in `scorecard.html`. The field of marks and its counts follow.
- **K1's targets or results:** edit the table in `self_work.html`; the chart follows.
- **A glossary term:** add a `<div class="g" id="g-...">` entry to `footer.html` and link to it with `<a class="term" href="#g-...">`. A test fails on a term with no entry.
- **The next steps on Aim:** `{{next:4,5,6}}` in `aim.html` picks the steps from `plan.html` by number.
- **A new room:** add it to `ROOMS` in `build.py` and in `site.js`, then add its name to the test's list. Use a plain ASCII word that claude.ai doesn't reserve: not `settings`, `sidebar`, `customize`, `upgrade`, `downloads` or `comment-*`.
