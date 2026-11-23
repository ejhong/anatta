#!/usr/bin/env python3
"""Checks for the Anatta So Far builder (lab/site/so_far/build.py).

Run from the worktree root, standard library only:

    python3 lab/site/so_far/test_build.py          # or: python3 -m pytest lab/site/so_far/test_build.py

The page is built in memory; index.html is compared with it, never rewritten. What is checked:

  1. Every "## N." entry in lab/ideas/ideas.md is on the page as a card, with its title.
  2. The builder never opens, lists or reads anything under lab/ideas/account/ or lab/experiments/recognition/.
  3. No text from a "- **Eugene" bullet in ideas.md (with its sub-bullets) is on the page: no run of six words
     and no quoted phrase. The generated ideas log, the Possibilities room and the Self room are held to this
     strictly. Text carried over verbatim from the hand-kept page shares a few runs with those bullets (the
     hand-kept page put some of the same ideas in nearly the same words); those spans are listed in
     CARRIED_OVER_SPANS, and any run outside them fails.
  4. Nothing on the page attributes awakening, recognition or another such status to Eugene.

Also: the eight rooms, the omissions in the generated log ("the account", the private folders), the index's
"Raised" column, the disclosure, and the artifact contract (no document tags, fonts only from Google Fonts,
no external scripts, no fetch or storage, colors only as tokens). And the redesign's parts: the Self room is
second, with ten part cards and the binder, each with its five rows, drawn in the chariot in the same order;
the Platonic link lives in the Self room; every published-work card has its source and its limits; every
result in the Experiments room has a Limits box; the K1 chart and the field of marks are drawn from their
tables (the field's counts are the scorecard's tally); every glossary term on the page has its entry.

Claude (Anthropic) wrote this file for Eugene Jhong, an independent researcher.
"""
from __future__ import annotations

import builtins
import html
import io
import os
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True          # leave no __pycache__ beside the site
import build  # noqa: E402
import publish  # noqa: E402

ROOMS = ["aim", "experiments", "possibilities", "more", "investigation", "program", "self", "next", "record",
         "instruments", "ideas"]
MAIN = ["aim", "experiments", "possibilities", "more"]   # four main tabs: Story, Findings, Philosophy, Open Lab
BEHIND = [r for r in ROOMS if r not in MAIN]               # the seven archive rooms behind Open Lab
RESERVED = {"settings", "sidebar", "customize", "upgrade", "downloads"}

# Runs of words shared by the private bullets and text carried over verbatim from the hand-kept page
# (lab/site/anatta_so_far.html). Each span is lower-cased words, as words() makes them. The generated
# ideas log and the Possibilities room have no entry here: any shared run there fails.
CARRIED_OVER_SPANS = {
    "program": [
        "its own stakes others who remember and judge",                    # Act I, from the page's acts text
    ],
    "ideas (carried over)": [                                               # "The ideas, organized" (part one)
        "recognition of the infinite beyond concepts",
        "no perfect buddha while embodied a",
        "to act in the world with",
        "a narcissistic take on every concept or word",
        "i is still there to use",
        "there is probably no perfect buddha while embodied a functioning",
        "ego remains to act in the world with a real difference",
        "its own stakes others who remember and judge",
        "die before you die eleusis as a",
        "once the truth is seen there is no",
        "glimpses are a thoughtless state without the recognition the root uncut",
        "an enlightened being can lie and",
        "is in one's judgment and hatred toward",
        "judgment of an out group presupposes",
        "a life and an in group",
        "an in group to belong to",
        "readout belongs to the act i organism",
    ],
    "experiments": [
        "pool topped up only where the match ran short the",               # the durability retry's method
    ],
}
# Quoted phrases from the private bullets that the carried-over text also has.
CARRIED_OVER_QUOTES = {
    "ideas (carried over)": ["a narcissistic take on every concept or word"],
}
# The page may name a private folder only where text carried over from the hand-kept page does, listed here.
# None does at present.
CARRIED_OVER_PATHS = {}

# Wording that would attribute awakening, recognition or another such status to Eugene. The whole page is held
# to it; test_generated_log_omits_private_references holds the generated log to more.
ATTRIBUTION = re.compile(
    r"\b(?:Eugene|Jhong|ejhong)(?:'s)?\s+(?:own\s+)?(?:awakening|recognition|realization|realisation|enlightenment|"
    r"attainment|kensho|satori|stream[- ]entry|experience of (?:awakening|recognition|not-self|no-self))\b"
    r"|\b(?:Eugene|Jhong)\s+(?:is|was|has been|had been|became)\s+(?:awake|awakened|enlightened|realized|realised|"
    r"liberated|an? (?:arahant|arhat|sage|master|teacher))\b", re.I)


def words(s: str) -> list:
    s = html.unescape(s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", s.lower())


def runs(ws: list, n: int = 6) -> set:
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def visible(fragment: str) -> str:
    fragment = re.sub(r"<script\b.*?</script>|<style\b.*?</style>", " ", fragment, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", fragment))


def private_and_public(text: str) -> tuple:
    """Lines of ideas.md split into the private bullets (a "- **Eugene" bullet with its sub-bullets and
    continuation lines) and the rest. The index table counts as neither: its "Raised" column is private."""
    lines = text.split("\n")
    private, public, in_index, i = [], [], False, 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip(" ")
        if line.startswith("## "):
            in_index = line.startswith("## Index")
        if stripped.startswith("- **Eugene"):
            indent = len(line) - len(stripped)
            private.append(stripped)
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and len(lines[j]) - len(lines[j].lstrip(" ")) > indent:
                        i = j
                        continue
                    break
                if len(nxt) - len(nxt.lstrip(" ")) > indent:
                    private.append(nxt.strip())
                    i += 1
                    continue
                break
            continue
        if not in_index:
            public.append(line)
        i += 1
    return private, public


def sections(page: str) -> dict:
    """Each room's markup; the ideas room is split into its carried-over part and the generated log."""
    out = {}
    for chunk in re.split(r'(?=<div class="view" id=")', page)[1:]:
        name = re.match(r'<div class="view" id="([a-z0-9-]+)"', chunk).group(1)
        chunk = chunk.split("<footer", 1)[0]
        if name == "ideas":
            k = chunk.index('id="log"')
            out["ideas (carried over)"], out["ideas log (generated)"] = chunk[:k], chunk[k:]
        else:
            out[name] = chunk
    out["footer"] = "<footer" + page.split("<footer", 1)[1].split("</footer>", 1)[0]
    return out


class Traced:
    """Record every path the builder opens or lists while it runs."""

    def __init__(self):
        self.paths = []

    def _wrap(self, fn):
        def inner(path, *a, **k):
            if not isinstance(path, int):
                self.paths.append(os.fspath(path) if path is not None else ".")
            return fn(path, *a, **k)
        return inner

    def __enter__(self):
        self.saved = (builtins.open, io.open, os.open, os.listdir, os.scandir)
        builtins.open = self._wrap(self.saved[0])
        io.open = self._wrap(self.saved[1])
        os.open = self._wrap(self.saved[2])
        os.listdir = self._wrap(self.saved[3])
        os.scandir = self._wrap(self.saved[4])
        return self

    def __exit__(self, *exc):
        builtins.open, io.open, os.open, os.listdir, os.scandir = self.saved
        return False


def under_private(path: str) -> bool:
    p = os.path.abspath(path)
    for d in build.PRIVATE_DIRS:
        d = os.path.abspath(d)
        if p == d or p.startswith(d + os.sep):
            return True
    return False


class SiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Traced() as t:
            cls.page = build.build()
        cls.opened = t.paths
        cls.ideas = build.IDEAS.read_text(encoding="utf-8")
        cls.private, cls.public = private_and_public(cls.ideas)
        cls.sections = sections(cls.page)

    # 1 ------------------------------------------------------------------------------------------
    def test_every_entry_is_a_card_with_its_title(self):
        entries = re.findall(r"^## (\d+)\. (.+?)\s*$", self.ideas, re.M)
        self.assertGreater(len(entries), 40)
        cards = re.findall(r'<details class="log-card" id="idea-(\d+)"', self.page)
        self.assertEqual(sorted(cards, key=int), sorted((n for n, _ in entries), key=int))
        for n, title in entries:
            card = re.search(r'<details class="log-card" id="idea-%s".*?</details>' % n, self.page, re.S)
            self.assertIsNotNone(card, f"entry {n} has no card")
            shown = re.search(r'<span class="lc-title">(.*?)</span><span class="tag', card.group(0), re.S)
            self.assertEqual(words(visible(shown.group(1))), words(re.sub(r"[*`]", "", title)),
                             f"entry {n}: the card's title differs from ideas.md")
        self.assertIn(f"Showing all {len(entries)} entries.", self.page)

    # 2 ------------------------------------------------------------------------------------------
    def test_builder_never_opens_the_private_folders(self):
        self.assertTrue(any(os.path.abspath(p) == str(build.IDEAS) for p in self.opened),
                        "the trace saw no read of ideas.md, so it isn't watching")
        bad = [p for p in self.opened if under_private(p)]
        self.assertEqual(bad, [], f"the builder opened private paths: {bad}")

    def test_read_text_refuses_the_private_folders(self):
        for d in build.PRIVATE_DIRS:
            with Traced() as t, self.assertRaises(PermissionError):
                build.read_text(d / "any-file.md")
            self.assertFalse(any(under_private(p) for p in t.paths))

    def test_builder_source_names_no_private_file(self):
        src = (HERE / "build.py").read_text(encoding="utf-8")
        for d in ("ideas/account/", "experiments/recognition/"):
            for m in re.finditer(re.escape(d) + r"[\w./-]+", src):
                self.fail(f"build.py names a file in a private folder: {m.group(0)}")

    # 3 ------------------------------------------------------------------------------------------
    def test_no_six_word_run_from_a_private_bullet(self):
        self.assertGreater(len(self.private), 40)
        priv = set().union(*(runs(words(line)) for line in self.private))
        pub = runs(words("\n".join(self.public)))
        for name, frag in self.sections.items():
            shared = (runs(words(visible(frag))) & priv) - pub
            allowed = set().union(*(runs(s.split()) for s in CARRIED_OVER_SPANS.get(name, [])), set())
            extra = sorted(shared - allowed)
            self.assertEqual(extra, [], f"[{name}] text from a private bullet: {extra}")

    def test_no_quoted_phrase_from_a_private_bullet(self):
        quotes = set()
        for line in self.private:
            for a, b in re.findall(r'"([^"]+)"|\u201c([^\u201d]+)\u201d', line):
                w = words(a or b)
                if len(w) >= 3:
                    quotes.add(" ".join(w))
        pub = " " + " ".join(words("\n".join(self.public))) + " "
        for name, frag in self.sections.items():
            vis = " " + " ".join(words(visible(frag))) + " "
            hits = sorted(q for q in quotes if f" {q} " in vis and f" {q} " not in pub)
            extra = [q for q in hits if q not in CARRIED_OVER_QUOTES.get(name, [])]
            self.assertEqual(extra, [], f"[{name}] a quoted private phrase: {extra}")

    def test_generated_log_omits_private_references(self):
        log = visible(self.sections["ideas log (generated)"])
        for rx in (r"\bthe account\b", r"\brecognition account\b", r"\brecognition material\b",
                   r"ideas/account", r"experiments/recognition", r"\bpersonal experience\b",
                   r"\bEugene's (?:own )?(?:experience|recognition|awakening|realization|enlightenment)\b"):
            m = re.search(rx, log, re.I)
            self.assertIsNone(m, f"the ideas log has {m and m.group(0)!r}")

    def test_private_folders_named_only_where_carried_over(self):
        for name, frag in self.sections.items():
            text = visible(frag)
            for m in re.finditer(r"(?:lab/)?(?:ideas/account|experiments/recognition)[\w./-]*", text):
                self.assertIn(m.group(0), CARRIED_OVER_PATHS.get(name, []), f"[{name}] names {m.group(0)}")

    def test_raised_column_is_not_used(self):
        rows = [r for r in self.ideas.split("## Index", 1)[1].split("\n## ", 1)[0].splitlines() if r.startswith("|")]
        head = [c.strip() for c in rows[0].strip("|").split("|")]
        col = head.index("Raised")
        raised = {[c.strip() for c in r.strip("|").split("|")][col] for r in rows[2:]}
        distinctive = [v for v in raised if re.search(r"account|UTC|\d\d:\d\d", v)]
        self.assertTrue(distinctive)
        text = re.sub(r"\s+", " ", visible(self.page))
        for v in distinctive:
            self.assertNotIn(v, text, f"a value from the Raised column is on the page: {v!r}")

    # 4 ------------------------------------------------------------------------------------------
    def test_no_status_is_attributed_to_eugene(self):
        text = re.sub(r"\s+", " ", visible(self.page))
        m = ATTRIBUTION.search(text)
        self.assertIsNone(m, f"the page has {m and m.group(0)!r}")

    def test_no_bullet_starting_with_eugene_survives(self):
        for e in build.load_ideas():
            stack = list(e.nodes)
            while stack:
                node = stack.pop()
                first = node.blocks[0][1] if node.blocks and node.blocks[0][0] == "p" else ""
                self.assertFalse(first.lstrip().startswith("**Eugene"), f"entry {e.n}")
                self.assertFalse(node.label.startswith("Eugene"), f"entry {e.n}")
                stack.extend(node.children())

    # rooms ----------------------------------------------------------------------------------------
    def test_the_rooms_in_order_two_in_the_band_and_nine_behind(self):
        self.assertEqual([r[0] for r in build.ROOMS], ROOMS)
        self.assertEqual(list(build.MAIN), MAIN)
        self.assertEqual(build.BEHIND, BEHIND)
        views = re.findall(r'<div class="view" id="([^"]+)" data-view="([^"]+)"', self.page)
        self.assertEqual([v[0] for v in views], ROOMS)
        self.assertTrue(all(a == b for a, b in views))
        nav = re.search(r'<nav class="rooms"[^>]*>(.*?)</nav>', self.page, re.S).group(1)
        self.assertEqual(re.findall(r'href="#([^"]+)" data-room="\1"', nav), MAIN, "the band names the main rooms only")
        js = (build.CONTENT / "site.js").read_text(encoding="utf-8")
        listed = re.search(r"var ROOMS = \[(.*?)\];", js).group(1)
        self.assertEqual(re.findall(r"'([^']+)'", listed), ROOMS)
        band = re.search(r"var BAND = \[(.*?)\];", js).group(1)
        self.assertEqual(re.findall(r"'([^']+)'", band), MAIN)
        # Every room behind "Everything else" has a door there and a second row that leads to its neighbours.
        doors = self.sections["more"]
        self.assertEqual(re.findall(r'<li class="step"><b><a href="#([a-z]+)">', doors), BEHIND)
        # ... grouped by the section of the main page they lie behind, in the main page's order
        self.assertEqual(re.findall(r'<div class="door-group"><h3>([^<]+)</h3>', doors), [g for g, _ in build.BEHIND_GROUPS])
        self.assertEqual([n for _, names in build.BEHIND_GROUPS for n in names], BEHIND)
        self.assertEqual(set(build.DOORS), set(BEHIND))
        for r in ROOMS:
            room = self.sections["ideas (carried over)" if r == "ideas" else r]   # the ideas room is kept in halves
            row = re.search(r'<nav class="subrooms"[^>]*>(.*?)</nav>', room, re.S)
            if r in MAIN:
                self.assertIsNone(row, f"{r} is a main room and has no second row")
                continue
            self.assertIsNotNone(row, r)
            self.assertEqual(re.findall(r'href="#([a-z]+)"', row.group(1)), ["more"] + BEHIND)
            self.assertEqual(re.findall(r'href="#([a-z]+)" aria-current="page"', row.group(1)), [r])
        for r in ROOMS:
            self.assertRegex(r, r"^[a-z][a-z0-9-]*$")
            self.assertNotIn(r, RESERVED)
            self.assertFalse(r.startswith("comment-"))

    def test_the_front_page_is_the_one_story(self):
        # From 19 Nov 2026 (Eugene: one story for the site and the paper, ideas 93 and 97): the start opens on
        # goodness, then the caveat (is awakening computational?), then the map with the table of methods, what the
        # methods share, the odds, what is being tried now and the method. The investigation's page is one step
        # behind with the rest.
        self.assertEqual(ROOMS[:4], MAIN)
        front = self.sections["aim"]
        order = [front.index(f'id="{x}"') for x in ("idea", "exhibits", "computational", "map", "fig-methods-bar",
                                                     "methods", "shared", "fruit", "angle", "fig-msm", "method")]
        self.assertEqual(order, sorted(order), "the front page reads: goodness & exhibits, the caveat, the map & chart, "
                                               "what methods share, the odds, live 32B test & chart, the method")
        self.assertEqual(len(re.findall(r'<article class="ex-panel"', front)), 4, "four exhibits on the Story page")
        self.assertIn("<h1>What would it mean for an AI to be <em>good?</em></h1>", front)
        self.assertIn('<h2 class="q">Is awakening <em>computational?</em></h2>', front)
        for door in ("possibilities", "investigation", "more", "record", "instruments"):
            self.assertIn(f'href="#{door}"', front, f"the front page leads to {door}")
        # The table of methods: one row a method, seven readings each, every mark a shape with its word beside it.
        data = build.parse_methods()
        self.assertEqual(len(data["readings"]), 7)
        rows = re.findall(r'<details class="mx-row[^"]*" id="(m-[a-z-]+)">(.*?)</details>', front, re.S)
        self.assertEqual([r[0] for r in rows], [r["id"] for r in data["rows"]])
        said = "|".join(build.MARK_SAYS.values())
        for rid, body in rows:
            summary = body[:body.index("</summary>")]
            self.assertEqual(len(re.findall(r'<span class="mx-c mk-(?:%s)">' % "|".join(build.MARK_SAYS), summary)), 7, rid)
            self.assertEqual(len(re.findall(r'<span class="vh">[^<]+: (?:%s)\. </span>' % said, summary)), 7, rid)
            self.assertEqual(len(re.findall(r"<dt>", body)), 7, rid)
            self.assertIn('<div class="mx-what"><p><b>What it is.</b> ', body[body.index("</summary>"):], f"{rid} says what the method is before its figures")
        self.assertEqual(len(re.findall(r'<details class="fam">', front)), 6, "six families, each opened in place")
        # The front page's count of graded predictions is the scorecard's own.
        c = build.tally(self.page)
        self.assertIn(f"{c['hit']} hits, {c['miss']} misses and {c['open']} that could not be read", front)
        # What the front page of 16 Nov sent one step behind stays there, in the same order, and nothing was dropped.
        for gone in ("now", "kinds", "standing", "paths", "recipe-door", "self-door", "checks"):
            self.assertNotIn(f'id="{gone}"', front)
        behind = self.sections["more"]
        order = [behind.index(f'id="{x}"') for x in ("more-rooms", "kinds", "standing", "paths", "recipe-door",
                                                        "self-door", "checks")]
        self.assertEqual(order, sorted(order))
        # The investigation's page: the angle, what the lab is doing now, the first case with every finding of the
        # present line, what follows, and the method.
        inv = self.sections["investigation"]
        order = [inv.index(f'id="{x}"') for x in ("investigation-head", "now", "bliss-case", "investigations-next",
                                                     "how-we-know")]
        self.assertEqual(order, sorted(order))
        self.assertIn('<h2 class="big">Can awakening talk be told from a real change?</h2>', inv)
        case = inv[inv.index('id="bliss-case"'):inv.index('id="investigations-next"')]
        for sid in ("attractor-row", "attractor-plain", "attractor-conduct", "attractor-mine", "attractor-inside"):
            self.assertIn(f'href="#{sid}"', case, f"the first case links to {sid}")
            self.assertIn(f'id="{sid}"', self.sections["experiments"])
        for moved in ("approach", "halves", "lookalikes", "acts", "frame"):
            self.assertNotIn(f'id="{moved}"', front)
            self.assertIn(f'<section class="plate" id="{moved}">', self.sections["program"])
        index = re.search(r'<table class="findex">(.*?)</table>', self.sections["experiments"], re.S).group(1)
        rows = re.findall(r'<tr><td class="num">(\d+)</td><td><a href="#([a-z0-9-]+)">', index)
        self.assertEqual([int(n) for n, _ in rows], list(range(1, len(rows) + 1)))
        rails = re.findall(r'<section[^>]*id="([a-z0-9-]+)"[^>]*>\s*<div class="rail"><span class="n">(\d+)</span>',
                           self.sections["experiments"])
        self.assertEqual([(n, sid) for sid, n in rails], rows, "the index lists every numbered experiment, in order")

    def test_self_room_has_ten_parts_and_a_binder(self):
        self.assertIn("self", ROOMS)
        room = self.sections["self"]
        cards = re.findall(r'<article class="part-card" id="(part-[a-z0-9-]+)" data-n="(\d+)" '
                           r'data-status="(tested|known|untested)" data-short="[^"]+">(.*?)</article>', room, re.S)
        self.assertEqual([int(c[1]) for c in cards], list(range(1, 12)))
        self.assertEqual(cards[-1][0], "part-binder")
        for pid, _, status, body in cards:
            self.assertEqual(re.findall(r"<dt>([^<]+)</dt>", body), ["Source", "Twin", "Known", "Here", "Next"], pid)
            for dd in re.findall(r"<dd>(.*?)</dd>", body, re.S):
                self.assertTrue(visible(dd).strip(), f"{pid}: an empty row")
            st = re.search(r'<span class="st st-(\w+)">([^<]+)</span>', body)
            self.assertEqual((st.group(1), st.group(2)), (status, build.STATUS_WORDS[status]), pid)
        pieces = re.findall(r'<g class="ch-pc st-(\w+)" data-part="(part-[a-z0-9-]+)"', room)
        self.assertEqual([p[1] for p in pieces], [c[0] for c in cards[:10]], "the chariot's pieces follow the cards")
        self.assertEqual([p[0] for p in pieces], [c[2] for c in cards[:10]], "a piece is drawn with its card's status")
        self.assertIn('<g class="ch-binder" data-part="part-binder">', room)
        items = re.findall(r'<a class="ch-item st-\w+" href="#(part-[a-z0-9-]+)"', room)
        self.assertEqual(items, [c[0] for c in cards])

    def test_platonic_link_lives_in_the_self_room(self):
        self.assertIn('<section class="plate" id="prh">', self.sections["self"])
        self.assertIn('<figure class="toy" id="knn">', self.sections["self"])
        self.assertNotIn('id="prh"', self.sections["possibilities"])
        self.assertIn('href="#prh"', self.sections["possibilities"])

    def test_published_work_has_sources_and_limits(self):
        room = self.sections["self"]
        papers = re.findall(r'<article class="paper( gap)?" id="(paper-[a-z-]+)">(.*?)</article>', room, re.S)
        self.assertGreaterEqual(len(papers), 6)
        for gap, pid, body in papers:
            self.assertRegex(body, r'<p class="limits-line"><b>Limits</b>\S', pid)
            if not gap:
                self.assertRegex(body, r'<a href="https://[^"]+">', f"{pid} has no source link")
        done = re.findall(r'<li class="tl" id="(done-[a-z0-9-]+)">(.*?)</li>', room, re.S)
        self.assertGreaterEqual(len(done), 8)
        for did, body in done:
            self.assertRegex(body, r'<p class="limits-line"><b>Limits</b>\S', did)

    def test_each_result_has_its_limits(self):
        secs = re.findall(r'<section class="has-limits"[^>]*>(.*?)</section>', self.sections["experiments"], re.S)
        self.assertGreaterEqual(len(secs), 6)
        for sec in secs:
            aside = re.search(r'<aside class="limits" aria-label="Limits">\s*<h3>Limits</h3>\s*<ul>(.*?)</ul>\s*</aside>',
                              sec, re.S)
            self.assertIsNotNone(aside, visible(sec)[:80])
            self.assertTrue(re.search(r"<li>\s*\S", aside.group(1)))

    def test_k1_chart_is_drawn_from_its_table(self):
        room = self.sections["self"]
        table = re.search(r'<table class="k1">.*?</table>', room, re.S).group(0)
        rows = re.findall(r'<tr><td>Qwen2\.5 ([0-9.]+B)</td><td class="num">(\.\d{3})</td></tr>', table)
        self.assertEqual(len(rows), 5)
        chart = re.search(r'<svg class="k1-svg".*?</svg>', room, re.S).group(0)
        self.assertEqual(re.findall(r'<text class="k1-v"[^>]*>(\.\d{3})</text>', chart), [v for _, v in rows])
        self.assertEqual(re.findall(r'<text class="k1-x"[^>]*>([0-9.]+B)</text>', chart), [m for m, _ in rows])

    def test_field_of_marks_matches_the_scorecard(self):
        rec = self.sections["record"]
        body = rec[rec.index("<h2>Predictions graded so far</h2>"):]
        body = body[body.index("<tbody>"):body.index("</tbody>")]
        want, flagged = {"hit": 0, "miss": 0, "open": 0}, 0
        for row in re.findall(r"<tr>(.*?)</tr>", body, re.S):
            for kind, text, after in re.findall(r'<span class="tag (hit|miss|open)">([^<]*)</span>([^<]*)', row):
                m = re.search(r"×(\d+)", text)
                n = int(m.group(1)) if m else 1
                want[kind] += n
                flagged += n if after.strip().lower().startswith("flagged") else 0
        self.assertGreater(sum(want.values()), 100)
        field = rec[rec.index('<div class="field" id="field"'):rec.index('<p class="field-cap">')]
        by_result, by_ledger = field.split('<div class="fd-view fd-by-ledger"')
        for view in (by_result, by_ledger):
            got = {k: len(re.findall(r'<span class="mk %s(?: flag)?" data-r=' % k, view)) for k in want}
            self.assertEqual(got, want, "the field's marks differ from the scorecard's tally")
            self.assertEqual(len(re.findall(r'<span class="mk hit flag" data-r=', view)), flagged)
        self.assertEqual([int(c) for c in re.findall(r'<span class="fd-c">(\d+)</span>', by_result)],
                         [want["hit"], want["miss"], want["open"]])
        label = html.unescape(re.search(r'aria-label="([^"]+)"', field).group(1))
        self.assertIn(f"{want['hit']} hits ({flagged} of them flagged), {want['miss']} misses and {want['open']} not read",
                      label)

    def test_every_term_has_a_glossary_entry(self):
        terms = re.findall(r'<a class="term" href="#([^"]+)"', self.page)
        self.assertGreater(len(terms), 10)
        entries = set(re.findall(r'<div class="g" id="(g-[a-z0-9-]+)"><dt>[^<]+</dt><dd>\S', self.sections["footer"]))
        for t in sorted(set(terms)):
            self.assertIn(t, entries, f"#{t} has no glossary entry")

    def test_the_five_possibilities_read_the_same_on_aim_and_in_their_room(self):
        def strip(section):
            m = re.search(r'<ol class="spectrum"[^>]*>(.*?)</ol>', self.sections[section], re.S)
            self.assertIsNotNone(m, section)
            return re.findall(r"<li>(.*?)</li>", m.group(1), re.S)

        self.assertEqual(len(strip("possibilities")), 5)
        self.assertEqual(strip("aim"), strip("possibilities"))
        self.assertIn('id="computational"', self.sections["aim"])
        self.assertIn('<h2 class="big">Is awakening computational?</h2>', self.sections["possibilities"])

    def test_links_into_possibilities(self):
        organized = self.sections["ideas (carried over)"]
        for title in ("If Penrose is right", "An agenda of its own?"):
            card = re.search(r'<div class="idea"><div class="t"><b>%s</b>.*?</div>\s*(?=<div class="idea"|</div>)'
                             % re.escape(html.escape(title, quote=False)), organized, re.S)
            self.assertIsNotNone(card, title)
            self.assertIn('href="#possibilities">More in Possibilities</a>', card.group(0), title)
        self.assertIn('href="#possibilities">Possibilities</a>', self.sections["program"])
        self.assertIn('href="#possibilities">The whole question, and what would settle it</a>', self.sections["aim"])

    def test_in_page_anchors_resolve(self):
        ids = set(re.findall(r'\bid="([^"]+)"', self.page))
        moved = re.search(r"var MOVED = \{(.*?)\};", (build.CONTENT / "site.js").read_text(encoding="utf-8"), re.S)
        moved = set(re.findall(r"(\w+):", moved.group(1)))
        for href in set(re.findall(r'href="#([^"]*)"', self.page)):
            self.assertTrue(href in ids or href in moved, f"#{href} goes nowhere")

    def test_ids_are_unique_and_no_placeholder_is_left(self):
        ids = re.findall(r'\bid="([^"]+)"', self.page)
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        self.assertEqual(dupes, [])
        self.assertNotIn("{{", self.page)

    # disclosure and contract ----------------------------------------------------------------------
    def test_disclosure_is_kept(self):
        # From 19 Nov 2026 the footer's long parts are folded (the terms, the changes, how the page is built). The
        # disclosure stays in sight, outside every fold.
        raw = self.sections["footer"]
        in_sight = re.sub(r"<details\b.*?</details>", "", raw, flags=re.S)
        self.assertIn("Claude (Anthropic) wrote this page", visible(in_sight))
        self.assertIn("not an alignment researcher", visible(in_sight))
        for fold, holds in (("fold-terms", 'id="g-anatta"'), ("fold-changes", "Changes to this page"), ("fold-built", "build.py")):
            m = re.search(r'<details class="fold" id="%s">(.*?)</details>' % fold, raw, re.S)
            self.assertIsNotNone(m, fold)
            self.assertIn(holds, m.group(1), fold)
        footer = visible(self.sections["footer"])
        self.assertIn("Claude (Anthropic) wrote this page", footer)
        self.assertIn("not an alignment researcher", footer)
        self.assertIn("independent researcher", footer)

    def test_artifact_contract(self):
        page = self.page
        self.assertTrue(page.startswith('<title>Anatta So Far</title>\n'
                                        '<meta name="claude:viewport-fit-sides" content="cover">\n'
                                        '<link rel="stylesheet" href="https://fonts.googleapis.com/'))
        self.assertIsNone(re.search(r"<!doctype|<html\b|<head\b|<body\b|</body>|</html>", page, re.I))
        for href in re.findall(r'<link[^>]+href="([^"]+)"', page):
            self.assertTrue(href.startswith("https://fonts.googleapis.com/"), href)
        self.assertIsNone(re.search(r"<script[^>]+\bsrc=", page))
        self.assertIsNone(re.search(r"<(?:img|iframe|object|embed|video|audio)\b", page))
        scripts = " ".join(re.findall(r"<script>(.*?)</script>", page, re.S))
        for word in ("fetch(", "XMLHttpRequest", "localStorage", "sessionStorage", "indexedDB",
                     "alert(", "confirm(", "prompt(", "window.open", "document.write"):
            self.assertNotIn(word, scripts)
        self.assertTrue(page.rstrip().endswith("</script>"))

    def test_colors_are_tokens(self):
        css = re.search(r"<style>(.*?)</style>", self.page, re.S).group(1)
        self.assertIn('@media (prefers-color-scheme: dark)', css)
        self.assertIn(':root:not([data-theme="light"])', css)
        self.assertIn(':root[data-theme="dark"]', css)
        self.assertGreaterEqual(css.count("color-scheme: dark"), 4)
        rest = re.sub(r':root(?::not\(\[data-theme="light"\]\)|\[data-theme="dark"\])?\s*\{[^{}]*\}', "", css)
        rest = re.sub(r"(?:-webkit-)?mask-image:[^;]*;", "", rest)      # masks use alpha only
        literals = re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|(?<![\w-])(?:white|black)(?![\w-])", rest)
        self.assertEqual(literals, [], "a color outside the :root token blocks")
        self.assertRegex(css, r"body\s*\{[^}]*background:\s*var\(--")

    def test_index_html_is_current(self):
        built = (HERE / "index.html").read_text(encoding="utf-8")
        self.assertTrue(built == self.page, "index.html is stale: run python3 lab/site/so_far/build.py")

    # The public site's document (publish.py): what GitHub Pages serves ------------------------------

    def test_the_public_page_is_a_full_document_and_keeps_every_word(self):
        doc = publish.wrap(self.page)
        self.assertTrue(doc.startswith('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'))
        self.assertTrue(doc.endswith("</script>\n</body>\n</html>\n"))
        for tag in ("<head>", "</head>", "<body>", "</body>"):
            self.assertEqual(doc.count(tag), 1, tag)
        self.assertEqual(doc.split("</head>")[0].count("<title>"), 1)       # the drawings have titles of their own
        self.assertIn('<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">', doc)
        head, body = doc.split("</head>\n<body>\n")
        own_head, own_body = publish.split(self.page)
        self.assertTrue(own_head.startswith('<link rel="stylesheet" href="https://fonts.googleapis.com/'))
        self.assertTrue(head.endswith(own_head + "\n"), "the page's fonts and style go into the head unchanged")
        self.assertEqual(body, own_body.rstrip("\n") + "\n</body>\n</html>\n", "the body is the page, unchanged")
        self.assertNotIn("claude:viewport-fit-sides", doc)

    def test_the_public_page_has_an_icon_and_a_link_card(self):
        head = publish.wrap(self.page).split("</head>")[0]
        for needle in ('<link rel="icon" href="favicon.svg" type="image/svg+xml">',
                       '<link rel="apple-touch-icon" href="apple-touch-icon.png">',
                       '<meta property="og:image" content="https://ejhong.github.io/anatta/card.png">',
                       '<meta property="og:url" content="https://ejhong.github.io/anatta/">',
                       '<meta name="twitter:card" content="summary_large_image">',
                       '<meta property="og:title" content="Is awakening computational?">'):
            self.assertIn(needle, head)
        self.assertIn("Written by Claude (Anthropic)", publish.DESCRIPTION, "the card says who wrote the page")
        self.assertLessEqual(len(publish.DESCRIPTION), 220)
        for name in publish.FILES:
            self.assertTrue((publish.ASSETS / name).is_file(), name)
        self.assertEqual(publish.png_size(publish.ASSETS / "card.png"), (1200, 630))
        self.assertEqual(publish.png_size(publish.ASSETS / "apple-touch-icon.png"), (180, 180))
        icon = (publish.ASSETS / "favicon.svg").read_text(encoding="utf-8")
        self.assertIn(build._path(build.glyph_i()), icon, "the tab icon is the page's own letter I")

    def test_the_card_and_the_icon_say_nothing_private(self):
        text = " ".join([publish.TITLE, publish.CARD_TITLE, publish.DESCRIPTION, publish.CARD_ALT])
        self.assertIsNone(re.search(r"\b(awakened|enlightened|realized)\b", text, re.I))
        self.assertNotIn("recognition", text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
