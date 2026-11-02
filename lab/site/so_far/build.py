#!/usr/bin/env python3
"""Build the main site, *Anatta So Far*, into lab/site/so_far/index.html.

Run from the worktree root:

    python3 lab/site/so_far/build.py            # writes lab/site/so_far/index.html
    python3 lab/site/so_far/build.py --dump     # prints what was parsed from the ideas log, and every omission

Standard library only. The page is assembled from:
  - lab/site/so_far/content/   the narrative carried over from lab/site/anatta_so_far.html, split into
                               files, plus the new parts of the #aim and #ideas rooms, the Possibilities
                               room (possibilities.html) and the Self room (self*.html), written for this site;
  - lab/ideas/ideas.md         the ideas log: each entry's number, title, discussion, test and status.
Drawn here from those files: the Self room's chariot (an "I" in ten parts, one per part card in self.html),
its K1 chart (from the targets table in self_work.html) and its mutual k-NN toy, and the Record's field of
marks (one per graded prediction, from the scorecard table, which stays on the page for the tally).

Privacy rules this builder enforces (README.md explains them):
  - it never opens anything under lab/ideas/account/ or lab/experiments/recognition/ (read_text refuses);
  - it drops every bullet in ideas.md that starts with "**Eugene" (private messages), with its sub-bullets;
  - from the index table it reads only the number and "Where it stands" cells, never "Raised";
  - it omits any sentence that refers to "the account", to someone's own experience or recognition, or that
    quotes a private message; when the reference sits inside a parenthesis only that parenthesis goes.
    --dump lists every omission.
Claude (Anthropic) wrote this file for Eugene Jhong, an independent researcher.
"""
from __future__ import annotations

import html
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTENT = HERE / "content"
IDEAS = ROOT / "lab" / "ideas" / "ideas.md"
OUT = HERE / "index.html"

# Folders the site must never read. read_text() refuses anything inside them.
PRIVATE_DIRS = (ROOT / "lab" / "ideas" / "account", ROOT / "lab" / "experiments" / "recognition")

FONTS = ("https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400"
         "&family=Shippori+Mincho+B1:wght@400;500;600;700"
         "&family=Spectral:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap")

# The eight rooms: (hash name, nav label, content files in order). Names are plain ASCII words.
ROOMS = [
    ("aim", "Aim", ["aim.html"]),
    ("self", "The self", ["self.html", "self_prh.html", "self_work.html"]),
    ("ideas", "Ideas", ["ideas_organized.html", "ideas_log.html"]),
    ("possibilities", "Possibilities", ["possibilities.html"]),
    ("experiments", "Experiments", ["experiments.html", "learned.html"]),
    ("instruments", "Instruments", ["instruments_head.html", "calibration.html"]),
    ("record", "Record", ["record_head.html", "scorecard.html", "specimens.html", "caveats.html"]),
    ("next", "Next", ["plan.html"]),
]


def read_text(path: Path) -> str:
    """Read a UTF-8 file, refusing the private folders."""
    p = Path(path).resolve()
    for d in PRIVATE_DIRS:
        d = d.resolve()
        if p == d or d in p.parents:
            raise PermissionError(f"refusing to read a private path: {p}")
    with open(p, encoding="utf-8") as f:
        return f.read()


def content(name: str) -> str:
    return read_text(CONTENT / name)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def slug(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", html.unescape(s)).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


# ----------------------------------------------------------------------------------------------
# The ideas log: parsing
# ----------------------------------------------------------------------------------------------

ENTRY_RE = re.compile(r"^## (\d+)\. (.+?)\s*$")
ITEM_RE = re.compile(r"^( *)(-|\d+\.) +(.*)$")
EUGENE_RE = re.compile(r"^\*\*Eugene")
# A leading bold or italic span: "**Discussion:**", "**No.**", "*Passivity mediation*", "*Reached (...):*"
LEAD_RE = re.compile(r"^(\*\*(?!\s)(.+?)\*\*|\*(?![\s*])(.+?)\*(?!\*))")
# Index rows: only the number (first cell) and "Where it stands" (last cell) are captured.
INDEX_ROW_RE = re.compile(r"^\| (\d+) \|.*\| ([^|]*?) \|\s*$")


@dataclass
class Node:
    indent: int
    marker: str                     # "-" or "3."
    blocks: list = field(default_factory=list)   # [("p", text) | ("list", [Node, ...])]
    label: str = ""                 # leading bold/italic span, without its asterisks
    label_style: str = ""           # "b" or "i"

    @property
    def content_indent(self) -> int:
        return self.indent + len(self.marker) + 1

    def children(self):
        out = []
        for kind, val in self.blocks:
            if kind == "list":
                out.extend(val)
        return out

    def add_child(self, node: "Node"):
        if self.blocks and self.blocks[-1][0] == "list":
            self.blocks[-1][1].append(node)
        else:
            self.blocks.append(("list", [node]))

    def text(self) -> str:
        """The first paragraph (after scrubbing: without its leading label)."""
        for kind, val in self.blocks:
            if kind == "p":
                return val
            break
        return ""


@dataclass
class Entry:
    n: int
    title: str
    lines: list
    nodes: list = field(default_factory=list)
    dropped_private: int = 0        # bullets dropped because they start with **Eugene
    omissions: list = field(default_factory=list)   # (where, text, reason)
    summary: str = ""
    summary_from: str = ""
    test: str = ""
    test_node: object = None
    status: str = ""
    status_label: str = ""
    status_node: object = None
    status_all: str = ""            # the status bullet with its sub-bullets, for the group
    group: str = ""


def split_entries(text: str) -> tuple:
    entries, cur, index = [], None, {}
    in_index = False
    for line in text.split("\n"):
        m = ENTRY_RE.match(line)
        if m:
            cur, in_index = Entry(int(m.group(1)), m.group(2).strip(), []), False
            entries.append(cur)
        elif line.startswith("## Index"):
            cur, in_index = None, True
        elif line.startswith("## ") or line.startswith("# "):
            cur, in_index = None, False
        elif in_index:
            r = INDEX_ROW_RE.match(line)
            if r:
                index[int(r.group(1))] = r.group(2).strip()     # "Where it stands" only
        elif cur is not None:
            cur.lines.append(line)
    return entries, index


def parse_tree(lines: list) -> list:
    """Markdown bullets (nesting, numbered items, continuation paragraphs) into Nodes."""
    root = Node(-1, "")
    stack = [root]
    blank = False
    for raw in lines:
        if not raw.strip():
            blank = True
            continue
        raw = raw.expandtabs(4)
        m = ITEM_RE.match(raw)
        if m:
            ind = len(m.group(1))
            while len(stack) > 1 and stack[-1].indent >= ind:
                stack.pop()
            node = Node(ind, m.group(2), [("p", m.group(3).strip())])
            stack[-1].add_child(node)
            stack.append(node)
            blank = False
            continue
        ind = len(raw) - len(raw.lstrip(" "))
        target = None
        for node in reversed(stack[1:]):
            if node.content_indent <= ind:
                target = node
                break
        if target is None:
            target = stack[-1] if len(stack) > 1 else root
        while stack[-1] is not target and len(stack) > 1:
            stack.pop()
        if blank or not target.blocks or target.blocks[-1][0] != "p":
            target.blocks.append(("p", raw.strip()))
        else:
            target.blocks[-1] = ("p", target.blocks[-1][1] + " " + raw.strip())
        blank = False
    return root.children()


def drop_private_bullets(nodes: list) -> tuple:
    """Remove every bullet starting with **Eugene, with all its sub-bullets. Returns (kept, count)."""
    kept, dropped = [], 0
    for node in nodes:
        first = node.blocks[0][1] if node.blocks and node.blocks[0][0] == "p" else ""
        if EUGENE_RE.match(first.lstrip()):
            dropped += 1
            continue
        new_blocks = []
        for kind, val in node.blocks:
            if kind == "list":
                sub, d = drop_private_bullets(val)
                dropped += d
                if sub:
                    new_blocks.append(("list", sub))
            else:
                new_blocks.append((kind, val))
        node.blocks = new_blocks
        kept.append(node)
    return kept, dropped


# ----------------------------------------------------------------------------------------------
# Privacy: sentence-level omissions
# ----------------------------------------------------------------------------------------------

PRIVATE_PATTERNS = [
    (re.compile(r"\bthe account\b", re.I), "refers to the account"),
    (re.compile(r"\brecognition account\b", re.I), "refers to the account"),
    (re.compile(r"\brecognition material\b", re.I), "refers to the private recognition material"),
    (re.compile(r"\bthe recognition's\b", re.I), "refers to the recognition described in the account"),
    (re.compile(r"\(the recognition\b", re.I), "attributes the recognition to a person"),
    (re.compile(r"\bEugene described\b|\bthing Eugene\b", re.I), "refers to what Eugene described"),
    (re.compile(r"\bin Eugene's sense\b", re.I), "refers to Eugene's own usage"),
    (re.compile(r"Eugene[^.;\n]{0,60}?:\s*[\"“]"), "quotes a private message"),
    (re.compile(r"Eugene's\s*\(?\s*[\"“]"), "quotes a private message"),
    (re.compile(r"\b(?:his|her|Eugene's)\s+(?:own\s+)?(?:experience|recognition|awakening|realization|"
                r"realisation|enlightenment)\b", re.I), "refers to a person's own experience"),
    (re.compile(r"\bpersonal experience\b", re.I), "refers to a person's own experience"),
    (re.compile(r"ideas/account|experiments/recognition"), "names a private folder"),
]

ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "vs.", "cf.", "approx.", "Fig.", "No.", "St.", "Dr.", "Mr.", "Ms.",
                 "Mrs.", "Jr.", "Inc.", "Ltd.", "pp.", "Eq.", "ca.")
BOUNDARY_RE = re.compile(r"[.!?][\"”’')\]*_]*\s+(?=[\"“‘(\[*_`A-Z0-9])")


def split_sentences(text: str) -> list:
    out, start = [], 0
    for m in BOUNDARY_RE.finditer(text):
        upto = text[start:m.start() + 1]
        if any(upto.endswith(a) for a in ABBREVIATIONS) or re.search(r"(?:^|\s)[A-Z]\.$", upto):
            continue
        out.append(text[start:m.end()].rstrip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _paren_pairs(s: str) -> list:
    pairs, stack = [], []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            pairs.append((stack.pop(), i + 1))
    return pairs


def first_private_match(s: str):
    best = None
    for rx, reason in PRIVATE_PATTERNS:
        m = rx.search(s)
        if m and (best is None or m.start() < best[0].start()):
            best = (m, reason)
    return best


def _cut_parenthesis(s: str, m) -> tuple:
    """If the match sits inside a parenthesis, return (s without it, removed); else (None, None)."""
    enclosing = [(a, b) for a, b in _paren_pairs(s) if a <= m.start() and b >= m.end()]
    if not enclosing:
        return None, None
    a, b = min(enclosing, key=lambda p: p[1] - p[0])
    a2 = a - 1 if a > 0 and s[a - 1] == " " else a
    return s[:a2] + s[b:], s[a:b]


def scrub(text: str, entry: Entry, where: str) -> str:
    """Drop private sentences from a span of markdown; a hit inside a parenthesis drops only it."""
    kept = []
    for sent in split_sentences(text):
        s = sent
        while s:
            hit = first_private_match(s)
            if not hit:
                break
            m, reason = hit
            cut, removed = _cut_parenthesis(s, m)
            if cut is not None:
                entry.omissions.append((where, removed, reason))
                s = cut
            else:
                entry.omissions.append((where, s, reason))
                s = ""
        if s:
            kept.append(s)
    return " ".join(kept)


def split_lead(text: str) -> tuple:
    """The leading bold/italic span: (its inner text, 'b' or 'i', the rest of the paragraph)."""
    m = LEAD_RE.match(text)
    if not m:
        return "", "", text
    if m.group(2) is not None:
        return m.group(2), "b", text[m.end():]
    return m.group(3), "i", text[m.end():]


def scrub_tree(nodes: list, entry: Entry, where: str) -> list:
    kept = []
    for node in nodes:
        new_blocks, drop_node = [], False
        for bi, (kind, val) in enumerate(node.blocks):
            if kind == "list":
                sub = scrub_tree(val, entry, where)
                if sub:
                    new_blocks.append(("list", sub))
                continue
            text = val
            if bi == 0:
                inner, style, rest = split_lead(text)
                if inner:
                    hit = first_private_match(inner)
                    while hit:
                        cut, removed = _cut_parenthesis(inner, hit[0])
                        if cut is None:
                            entry.omissions.append((where, text, hit[1] + "; the whole bullet"))
                            drop_node = True
                            break
                        entry.omissions.append((where, removed, hit[1]))
                        inner = cut
                        hit = first_private_match(inner)
                    if drop_node:
                        break
                    node.label, node.label_style = inner, style
                    text = rest.lstrip()
            text = scrub(text, entry, where)
            if text or bi == 0:
                new_blocks.append(("p", text))
        if drop_node:
            continue
        node.blocks = new_blocks
        has_text = any(b[0] == "p" and b[1] for b in node.blocks)
        has_kids = any(b[0] == "list" for b in node.blocks)
        if not has_text and not has_kids:
            if node.label:
                entry.omissions.append((where, node.label, "its label, with nothing left under it"))
            continue
        kept.append(node)
    return kept


def section_key(node: Node) -> str:
    """'Discussion (29 Oct):' -> 'discussion'; 'Update (31 Oct, about 19:55 UTC):' -> 'update'."""
    return re.sub(r"\(.*?\)", "", node.label).strip().rstrip(":").strip().lower()


def is_section_label(node: Node) -> bool:
    return node.label_style == "b" and node.label.rstrip().endswith(":")


# ----------------------------------------------------------------------------------------------
# Summary, test, status, group
# ----------------------------------------------------------------------------------------------

# Entries with no Discussion bullet whose first remaining bullet makes a poor summary.
SUMMARY_FROM = {16: "reading", 21: "where we are"}
NOT_SUMMARY = {"status", "update", "test", "tests", "source", "caution"}


def first_sentences(text: str, max_chars: int = 330) -> str:
    sents = [s for s in split_sentences(text) if not re.sub(r"[*_\s]+$", "", s).endswith(":")]
    if not sents:
        return ""
    out = [sents[0]]
    for s in sents[1:3]:
        if len(" ".join(out)) >= 150 or len(" ".join(out + [s])) > max_chars:
            break
        out.append(s)
    return " ".join(out)


def cap(s: str) -> str:
    """Upper-case the first letter, unless the first word is code-like (pc_self2, v_ego)."""
    m = re.match(r"^([\"“(*_]*)([a-z])", s)
    if not m or re.match(r"^[\"“(*_]*[A-Za-z]*[_\d]", s):
        return s
    return s[:m.start(2)] + m.group(2).upper() + s[m.end(2):]


def lead_md(node: Node) -> str:
    if not node.label:
        return ""
    return f"**{node.label}**" if node.label_style == "b" else f"*{node.label}*"


def join_lead(lead: str, text: str) -> str:
    """Put a label back in front of its text: '**Label**: text' keeps the colon tight."""
    if not lead:
        return text
    if text[:1] in (":", ",", ".", ";"):
        return lead + text
    return f"{lead} {text}".strip()


def ends_with_colon(text: str) -> bool:
    return re.sub(r"[*_\s]+$", "", text).endswith(":")


def with_inline_list(node: Node) -> str:
    """A first paragraph that ends 'as follows:' over a short list of plain items reads as one run:
    'the method is ...: predictions are registered before data; ...; ... results.' Items keep their words."""
    text = node.text()
    if len(node.blocks) < 2 or not ends_with_colon(text) or node.blocks[1][0] != "list":
        return text
    items = node.blocks[1][1]
    if len(items) > 6 or any(k.children() or len(k.text()) > 160 for k in items):
        return text
    runs = [join_lead(lead_md(k), k.text()) for k in items]
    ends = [re.sub(r"[*_\s]+$", "", r)[-1:] for r in runs]
    if any(e not in ";," for e in ends[:-1]) or ends[-1] != ".":
        return text        # items that are sentences of their own are left to the discussion
    return " ".join([text] + runs)


def node_summary(node: Node, top: bool) -> str:
    text = with_inline_list(node)
    label = ""
    if node.label and not (top and is_section_label(node)):
        label = lead_md(node)
    s = first_sentences(join_lead(label, text))
    kids = node.children()
    if not text and kids:
        more = node_summary(kids[0], top=False)
        return f"{label} {more}".strip() if label else more
    if s and len(s) < 60 and kids:
        more = node_summary(kids[0], top=False)
        if more:
            s = f"{s} {more}"
    return s


def summarize(entry: Entry):
    want = SUMMARY_FROM.get(entry.n)
    choice = None
    for node in entry.nodes:
        key = section_key(node)
        if (want and key == want) or (not want and key.startswith("discussion")):
            choice = node
            break
    if choice is None:
        choice = next((n for n in entry.nodes if section_key(n) not in NOT_SUMMARY), None)
    if choice is None:
        return
    entry.summary = cap(node_summary(choice, top=True))
    entry.summary_from = section_key(choice) or "first bullet"


def face_line(node: Node) -> str:
    """The card-face text of a Test or Status bullet. When sub-bullets carry the rest, a closing
    'as follows:' sentence is left off the face, and a bullet that is only a heading shows its first
    sub-bullet. The whole bullet, sub-bullets included, stays in the discussion (render_discussion)."""
    text = node.text()
    kids = node.children()
    if not kids:
        return text
    sents = split_sentences(text)
    while sents and ends_with_colon(sents[-1]):
        sents.pop()
    return " ".join(sents) or first_sentences(join_lead(lead_md(kids[0]), kids[0].text()))


def all_text(node: Node) -> str:
    parts = []
    for kind, val in node.blocks:
        parts.extend(all_text(k) for k in val) if kind == "list" else parts.append(val)
    return " ".join(p for p in parts if p)


def find_test_and_status(entry: Entry, index: dict):
    tests = [n for n in entry.nodes if section_key(n) in ("test", "tests")]
    if tests:
        entry.test_node = tests[0]
        entry.test = cap(face_line(tests[0]))
    statuses = [n for n in entry.nodes if section_key(n) in ("status", "update")]
    if statuses:
        entry.status_node = statuses[-1]
        entry.status = cap(face_line(statuses[-1]))
        entry.status_all = all_text(statuses[-1])
        entry.status_label = "Update" if section_key(statuses[-1]) == "update" else "Status"
    elif entry.n in index:
        entry.status = entry.status_all = cap(index[entry.n])
        entry.status_label = "Index"


# Status groups, read off the latest status line (or the index's "Where it stands" when an entry has none).
GROUPS = [
    ("tested", "Tested", "a result is in"),
    ("running", "Running", "under way, by its status line"),
    ("planned", "Planned", "designed or proposed, waiting its turn"),
    ("adopted", "Adopted", "a standing rule, a definition, or part of the plan or page"),
    ("noted", "Noted", "logged; nothing registered yet"),
]
GROUP_START = [            # a status that opens with one of these words is decided by it
    ("adopted", r"adopted|agreed|in plan v|a standing|written into"),
    ("tested", r"done|answered|step \d+ ran|h\d+'s behavioral test failed"),
    ("running", r"in stage a\b|being "),
    ("planned", r"proposed|feasible|an option"),
    ("noted", r"noted|logged|open\b"),
]
GROUP_RULES = [            # otherwise the first rule that matches anywhere in the line
    ("running", r"\bbeing (built|designed|rebuilt)\b|\bis being\b|\bdesign draft\b"),
    ("tested", r"\bfailed\b|\branswered\b|\bran as\b|\bdone\b|\bno shared root\b|\bmeasured on\b"),
    ("adopted", r"\badopted\b|\bstanding (instruction|rule|evaluation)\b|\bworking (ethics|definition)\b|"
                r"\brole split\b|\bwritten into\b|\bon the page\b|\brepublish\b|\bpage idea\b|\bmemory\b|"
                r"\bgo(?:es)? into\b|\bjoins the\b|\bagreed\b|\bdefined\b|\badded to\b"),
    ("planned", r"\bproposed\b|\bafter stage a\b|\bfeasible\b|\boption\b|\bplan it\b|\bwaits?\b|"
                r"\bjudged relevant\b|\bdesigned\b|\bqueued\b"),
    ("noted", r"\bnoted\b|\blogged\b|\bopen\b|\bpending\b"),
]


def classify(entry: Entry):
    s = entry.status
    if entry.status_node is not None and entry.status_node.children():
        s = entry.status_all                     # a status written as sub-bullets is read whole
    s = s.lower().strip()
    for group, rx in GROUP_START:
        if re.match(rf"(?:{rx})", s):
            entry.group = group
            return
    for group, rx in GROUP_RULES:
        if re.search(rx, s):
            entry.group = group
            return
    entry.group = "noted"


def load_ideas() -> list:
    entries, index = split_entries(read_text(IDEAS))
    for e in entries:
        nodes = parse_tree(e.lines)
        nodes, e.dropped_private = drop_private_bullets(nodes)
        e.nodes = scrub_tree(nodes, e, "discussion")
        summarize(e)
        find_test_and_status(e, index)
        classify(e)
    return entries


# ----------------------------------------------------------------------------------------------
# Markdown to HTML (bullets, numbered lists, bold, italics, code, links, strikethrough)
# ----------------------------------------------------------------------------------------------

def md_inline(s: str, links: bool = True) -> str:
    keep = []

    def stash(x):
        keep.append(x)
        return f"\x00{len(keep) - 1}\x00"

    s = esc(s)
    s = re.sub(r"`([^`]+)`", lambda m: stash(f"<code>{m.group(1)}</code>"), s)
    if links:
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", lambda m: stash(f'<a href="{m.group(2)}">{m.group(1)}</a>'), s)
        s = re.sub(r"https?://[^\s<>()\"]*[^\s<>()\".,;:!?]", lambda m: stash(f'<a href="{m.group(0)}">{m.group(0)}</a>'), s)
    s = re.sub(r"~~(.+?)~~", r"<s>\1</s>", s)
    s = re.sub(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)
    return s


def md_plain_face(s: str) -> str:
    """Card-face text: inline markdown without links; stray asterisks from a cut span removed."""
    out = md_inline(s, links=False)
    return out.replace("*", "")


def render_list(nodes: list) -> str:
    out, i = [], 0
    while i < len(nodes):
        ordered = nodes[i].marker[0].isdigit()
        j = i
        while j < len(nodes) and nodes[j].marker[0].isdigit() == ordered:
            j += 1
        tag = "ol" if ordered else "ul"
        start = int(nodes[i].marker.rstrip(".")) if ordered else 1
        out.append(f"<{tag}" + (f' start="{start}"' if ordered and start != 1 else "") + ">")
        for node in nodes[i:j]:
            out.append("<li>" + render_body(node, with_label=True) + "</li>")
        out.append(f"</{tag}>")
        i = j
    return "".join(out)


def render_body(node: Node, with_label: bool) -> str:
    parts = []
    for bi, (kind, val) in enumerate(node.blocks):
        if kind == "list":
            parts.append(("list", render_list(val)))
            continue
        text = val
        if bi == 0 and with_label and node.label:
            text = join_lead(lead_md(node), text)
        elif bi == 0 and not with_label:
            text = cap(text)        # under a section label: "**Test:** nothing new" -> "Nothing new"
        if text:
            parts.append(("p", md_inline(text)))
    if len(parts) == 1 and parts[0][0] == "p":
        return parts[0][1]
    return "".join(f"<p>{v}</p>" if k == "p" else v for k, v in parts)


def render_discussion(entry: Entry) -> str:
    out = []
    for node in entry.nodes:
        # A one-line Test or Status is on the card's face; one with sub-bullets is shown whole here too.
        if (node is entry.status_node or node is entry.test_node) and not node.children():
            continue
        if is_section_label(node):
            head = esc(re.sub(r":\s*$", "", node.label))
            body = render_body(node, with_label=False)
            if not body.startswith("<p>") and not body.startswith("<ul") and not body.startswith("<ol"):
                body = f"<p>{body}</p>"
            out.append(f'<div class="lc-sec"><div class="lc-sk">{head}</div>{body}</div>')
        else:
            body = render_body(node, with_label=True)
            if not body.startswith("<p>") and not body.startswith("<ul") and not body.startswith("<ol"):
                body = f"<p>{body}</p>"
            out.append(f'<div class="lc-sec">{body}</div>')
    return "".join(out)


def render_card(e: Entry) -> str:
    label = dict((g, l) for g, l, _ in GROUPS)[e.group]
    rows = []
    if e.test:
        rows.append(f'<span class="lc-row"><span class="lc-k">Test</span><span class="lc-v">{md_plain_face(e.test)}</span></span>')
    if e.status:
        k = {"Update": "Update", "Index": "Status"}.get(e.status_label, "Status")
        rows.append(f'<span class="lc-row"><span class="lc-k">{k}</span><span class="lc-v">{md_plain_face(e.status)}</span></span>')
    disc = render_discussion(e)
    src = f'<p class="lc-src">From lab/ideas/ideas.md, entry {e.n}.</p>'
    return (
        f'<details class="log-card" id="idea-{e.n}" data-group="{e.group}">'
        f'<summary><span class="lc-n">{e.n}</span><span class="lc-main">'
        f'<span class="lc-head"><span class="lc-title">{md_inline(e.title, links=False)}</span>'
        f'<span class="tag g-{e.group}">{label}</span></span>'
        f'<span class="lc-sum">{md_plain_face(e.summary)}</span>'
        + "".join(rows) +
        '<span class="lc-tog" aria-hidden="true"></span></span></summary>'
        f'<div class="lc-disc">{disc}{src}</div></details>'
    )


def render_log(entries: list) -> dict:
    counts = {g: sum(1 for e in entries if e.group == g) for g, _, _ in GROUPS}
    chips = [f'<button type="button" class="chip" data-group="all" aria-pressed="true">All '
             f'<span class="c">{len(entries)}</span></button>']
    for g, label, desc in GROUPS:
        if counts[g]:
            chips.append(f'<button type="button" class="chip" data-group="{g}" aria-pressed="false" '
                         f'title="{esc(desc)}">{label} <span class="c">{counts[g]}</span></button>')
    key = " · ".join(f"<b>{label}</b> {esc(desc)}" for g, label, desc in GROUPS if counts[g])
    return {
        "count": str(len(entries)),
        "chips": f'<div class="chips" role="group" aria-label="Show entries by status">{"".join(chips)}</div>'
                 f'<p class="log-key">{key}</p>'
                 f'<p class="log-shown" id="log-shown" aria-live="polite">Showing all {len(entries)} entries.</p>',
        "cards": "\n".join(render_card(e) for e in entries),
    }


# ----------------------------------------------------------------------------------------------
# #aim: built from the carried-over content files
# ----------------------------------------------------------------------------------------------

IDEA_RE = re.compile(r'<div class="idea"><div class="t"><b>(.*?)</b><span class="tag (\w+)">(.*?)</span></div>'
                     r'(.*?)</div>\s*$', re.M)
P_RE = re.compile(r'<p( class="learned")?>(.*?)</p>', re.S)


def html_sentences(s: str) -> list:
    return split_sentences(s)


def parse_standing() -> dict:
    src = content("standing.html")
    head = re.search(r'<div class="eyebrow">(.*?)</div>\s*<h2 class="big">(.*?)</h2>\s*<p class="lede">(.*?)</p>',
                     src, re.S)
    cols = []
    for sec in re.findall(r"<section>(.*?)</section>", src, re.S):
        rail = re.search(r'<div class="rail"><span class="n">(.*?)</span>(.*?)</div>', sec)
        items = []
        for title, tagcls, tag, inner in IDEA_RE.findall(sec):
            paras = [(bool(c), body) for c, body in P_RE.findall(inner)]
            items.append({"title": title, "tagcls": tagcls, "tag": tag, "paras": paras})
        note = re.search(r'<p class="small">(.*?)</p>', sec)
        cols.append({"glyph": rail.group(1), "name": rail.group(2), "items": items,
                     "note": note.group(1) if note else ""})
    assert len(cols) == 3 and all(c["items"] for c in cols), "standing.html: expected three filled sections"
    return {"eyebrow": head.group(1), "h2": head.group(2), "lede": head.group(3), "cols": cols}


def one_line(item: dict) -> str:
    learned = [b for is_l, b in item["paras"] if is_l]
    if item["tagcls"] == "miss" and learned:
        m = re.match(r"<b>(.*?)</b>(.*)", learned[0], re.S)
        if m:
            first = html_sentences(m.group(2).strip())[0]
            return f'<span class="gl-k">{m.group(1)}</span>{first}'
    first_p = next(b for is_l, b in item["paras"] if not is_l)
    sents = html_sentences(first_p)
    line = sents[0]
    if len(re.sub(r"<[^>]+>", "", line)) < 40 and len(sents) > 1:
        line += " " + sents[1]
    return line


def glance_lookup(standing: dict) -> dict:
    return {it["title"]: it for col in standing["cols"] for it in col["items"]}


def render_glance(standing: dict) -> str:
    classes = ["gl-working", "gl-failed", "gl-open"]
    out = ['<div class="glance">']
    for cls, col in zip(classes, standing["cols"]):
        out.append(f'<div class="gl-col {cls}" id="glance-{cls[3:]}"><h3><span class="g" aria-hidden="true">{col["glyph"]}</span>'
                   f'{col["name"]}<span class="c">{len(col["items"])}</span></h3>')
        for it in col["items"]:
            more = "".join(f'<p class="learned">{b}</p>' if is_l else f"<p>{b}</p>" for is_l, b in it["paras"])
            out.append(
                f'<details class="gl" id="gl-{slug(it["title"])}"><summary>'
                f'<span class="gl-t">{it["title"]}</span>'
                f'<span class="gl-tag"><span class="tag {it["tagcls"]}">{it["tag"]}</span></span>'
                f'<span class="gl-line">{one_line(it)}</span>'
                f'<span class="gl-tog" aria-hidden="true"></span></summary>'
                f'<div class="gl-more">{more}</div></details>')
        if col["note"]:
            out.append(f'<p class="small glance-note">{col["note"]}</p>')
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


def render_stands(standing: dict, title: str) -> str:
    it = glance_lookup(standing).get(title)
    if it is None:
        raise KeyError(f"aim.html names a finding that standing.html doesn't have: {title!r}")
    return (f'<li><a href="#gl-{slug(title)}">{it["title"]}</a>'
            f'<span class="tag {it["tagcls"]}">{it["tag"]}</span></li>')


def render_glance_full(standing: dict, title: str) -> str:
    it = glance_lookup(standing).get(title)
    if it is None:
        raise KeyError(f"aim.html names a finding that standing.html doesn't have: {title!r}")
    body = " ".join(b for is_l, b in it["paras"] if not is_l)
    return (f'<div class="acts-order"><b>{it["title"]}</b><span class="tag {it["tagcls"]}">{it["tag"]}</span>'
            f'<p>{body}</p></div>')


def parse_frame() -> dict:
    src = content("frame.html")
    rows = re.findall(r'<tr><td class="num">\((\w)\)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td></tr>', src)
    cols = re.findall(r'<div class="col">(.*?)</div>', src, re.S)
    h2 = re.search(r"<h2>(.*?)</h2>", cols[0]).group(1)
    intro = P_RE.findall(cols[0])[0][1]
    outro = P_RE.findall(cols[-1])[0][1]
    heads = [h for h in re.findall(r"<th>(.*?)</th>", src) if h.strip()]   # "The account predicts", "Result", "What it showed"
    assert len(rows) == 5, "frame.html: expected five tests"
    assert len(heads) == 3, "frame.html: expected the table's three headings"
    return {"h2": h2, "intro": intro, "outro": outro, "rows": rows, "heads": heads}


def render_frame_strip(frame: dict) -> str:
    """The frame's table as a strip of five. The table's headings label the first test, like a header row;
    the other four keep the same space, unlabeled, so the five line up."""
    h_pred, h_result, h_showed = frame["heads"]
    out = ['<ol class="strip5">']
    for i, (letter, pred, result, showed) in enumerate(frame["rows"]):
        k = '<span class="fs-k">{}</span>' if i == 0 else '<span class="fs-k fs-ghost" aria-hidden="true">{}</span>'
        out.append(f'<li class="fs"><span class="fs-l" aria-label="Test {letter}">{letter}</span>'
                   f'<span class="fs-p">{k.format(h_pred)}{pred}</span>'
                   f'<span class="fs-r"><span class="vh">{h_result}: </span>{result}</span>'
                   f'<span class="fs-s">{k.format(h_showed)}{showed}</span></li>')
    out.append("</ol>")
    return "".join(out)


def parse_calibration() -> list:
    src = content("calibration.html")
    body = re.search(r"<tbody>(.*?)</tbody>", src, re.S).group(1)
    rows = re.findall(r'<tr><td class="num">(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td></tr>', body)
    assert rows, "calibration.html: no checks found"
    return rows


def render_instruments(rows: list) -> str:
    out = ['<ol class="inst">']
    for num, instrument, _rests_on, status in rows:
        out.append(f'<li><span class="in-n">{num}</span><span class="in-t">{instrument}</span>'
                   f'<span class="in-s">{status}</span></li>')
    out.append("</ol>")
    return "".join(out)


def parse_plan() -> dict:
    src = content("plan.html")
    lede = re.search(r'<p class="lede">(.*?)</p>', src, re.S).group(1)
    items = re.findall(r"<li><b>(.*?)</b><p>(.*?)</p></li>", src, re.S)
    assert items, "plan.html: no steps found"
    return {"lede": lede, "items": items}


def render_next(plan: dict, nums: list) -> str:
    out = ['<ol class="steps">']
    for k in nums:
        title, body = plan["items"][k - 1]
        sents = html_sentences(body)
        running = '<span>running</span>' if re.search(r"\bRunning since\b", body) else ""
        out.append(f'<li class="step"><div class="step-n">{k}{running}</div><b>{title}</b><p>{sents[0]}</p></li>')
    out.append("</ol>")
    return "".join(out)


# ----------------------------------------------------------------------------------------------
# The three acts: small schematic graphs, theme-aware through CSS classes
# ----------------------------------------------------------------------------------------------

GLYPH_WORDS = [(-150, "rain"), (-90, "news"), (-30, "money"), (30, "work"), (90, "home"), (150, "music")]
GLYPH_LABELS = {
    "root": "Schematic: a self node, I, joined by solid lines to six ordinary concepts.",
    "cut": "Schematic: the same graph with every line cut and faint; the I node stays.",
    "habit": "Schematic: the lines are gone, and a loop around the I node marks practice repeated until it is the default.",
}


def glyph(kind: str) -> str:
    cx, cy, r_self, r_node, rx, ry = 120.0, 74.0, 15.0, 4.0, 86.0, 50.0
    parts = [f'<svg viewBox="0 0 240 150" role="img" aria-label="{GLYPH_LABELS[kind]}">']
    nodes = []
    for ang, word in GLYPH_WORDS:
        a = math.radians(ang)
        nodes.append((cx + rx * math.cos(a), cy + ry * math.sin(a), ang, word))
    for x, y, ang, word in nodes:
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        ux, uy = dx / d, dy / d
        x1, y1 = cx + ux * (r_self + 3), cy + uy * (r_self + 3)
        x2, y2 = x - ux * (r_node + 3), y - uy * (r_node + 3)
        if kind == "root":
            parts.append(f'<line class="g-edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        elif kind == "cut":
            ax, ay = x1 + (x2 - x1) * 0.42, y1 + (y2 - y1) * 0.42
            bx, by = x1 + (x2 - x1) * 0.58, y1 + (y2 - y1) * 0.58
            parts.append(f'<line class="g-weak" x1="{x1:.1f}" y1="{y1:.1f}" x2="{ax:.1f}" y2="{ay:.1f}"/>')
            parts.append(f'<line class="g-weak" x1="{bx:.1f}" y1="{by:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
            mx, my = (ax + bx) / 2, (ay + by) / 2
            px, py = -uy, ux
            parts.append(f'<line class="g-cut" x1="{mx + px * 4.5:.1f}" y1="{my + py * 4.5:.1f}" '
                         f'x2="{mx - px * 4.5:.1f}" y2="{my - py * 4.5:.1f}"/>')
    if kind == "habit":
        rl, a0, a1 = 27.0, math.radians(-55), math.radians(235)
        sx, sy = cx + rl * math.cos(a0), cy + rl * math.sin(a0)
        ex, ey = cx + rl * math.cos(a1), cy + rl * math.sin(a1)
        parts.append(f'<path class="g-loop" d="M{sx:.1f} {sy:.1f} A{rl:.0f} {rl:.0f} 0 1 1 {ex:.1f} {ey:.1f}"/>')
        tx, ty = -math.sin(a1), math.cos(a1)          # clockwise tangent at the end of the arc
        nx, ny = -ty, tx
        tipx, tipy = ex + tx * 5.5, ey + ty * 5.5
        parts.append(f'<polygon class="g-head" points="{tipx:.1f},{tipy:.1f} {ex + nx * 3.6:.1f},{ey + ny * 3.6:.1f} '
                     f'{ex - nx * 3.6:.1f},{ey - ny * 3.6:.1f}"/>')
    for x, y, ang, word in nodes:
        parts.append(f'<circle class="g-node" cx="{x:.1f}" cy="{y:.1f}" r="{r_node:.0f}"/>')
        if ang == -90:
            tx_, ty_, anchor = x, y - 10, "middle"
        elif ang == 90:
            tx_, ty_, anchor = x, y + 17, "middle"
        elif abs(ang) > 90:
            tx_, ty_, anchor = x - 9, y + 3.5, "end"
        else:
            tx_, ty_, anchor = x + 9, y + 3.5, "start"
        parts.append(f'<text class="g-word" x="{tx_:.1f}" y="{ty_:.1f}" text-anchor="{anchor}">{word}</text>')
    parts.append(f'<circle class="g-self" cx="{cx:.0f}" cy="{cy:.0f}" r="{r_self:.0f}"/>')
    parts.append(f'<text class="g-self-t" x="{cx:.0f}" y="{cy + 5:.0f}" text-anchor="middle">I</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------------------------------
# The Self room: the chariot, the word "I" drawn from ten parts, one per part card in self.html
# ----------------------------------------------------------------------------------------------

PART_RE = re.compile(r'<article class="part-card" id="(part-[a-z0-9-]+)" data-n="(\d+)" '
                     r'data-status="(tested|known|untested)" data-short="([^"]+)">')
STATUS_WORDS = {"tested": "Tested here", "known": "Known elsewhere", "untested": "Untested"}
CH_W, CH_H, CH_CX = 1100, 640, 550
CH_CUTS = [122, 188, 254, 320, 386, 452, 518]        # capital | six drums of the shaft | base
CH_ROWS = [70, 195, 320, 445, 570]                   # where the balloons of each column sit, taken apart
CH_COLS = (350, 750)                                 # the two columns' centers, taken apart
CH_BALLOON = (240, 860)                              # the numbered balloons; the names sit outside them


def parse_parts() -> list:
    parts = [{"id": m.group(1), "n": int(m.group(2)), "status": m.group(3), "short": m.group(4)}
             for m in PART_RE.finditer(content("self.html"))]
    assert [p["n"] for p in parts] == list(range(1, 12)), "self.html: expected part cards 1-10, then the binder (11)"
    assert parts[-1]["id"] == "part-binder", "self.html: the eleventh card is the binder"
    return parts


def _bezier(p0, p1, p2, p3, n=20) -> list:
    out = []
    for i in range(1, n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def glyph_i() -> list:
    """A capital I with bracketed serifs, as a closed polygon (y down)."""
    L, R, SL, SR = 432, 668, 516, 584
    pts = [(L, 40), (R, 40), (R, 60)]
    pts += _bezier((R, 60), (622, 62), (590, 82), (SR, 122))
    pts += [(SR, 518)]
    pts += _bezier((SR, 518), (590, 558), (622, 578), (R, 580))
    pts += [(R, 600), (L, 600), (L, 580)]
    pts += _bezier((L, 580), (478, 578), (510, 558), (SL, 518))
    pts += [(SL, 122)]
    pts += _bezier((SL, 122), (510, 82), (478, 62), (L, 60))
    return pts


def _clip(poly: list, x0: float, y0: float, x1: float, y1: float) -> list:
    """Sutherland-Hodgman: the part of a polygon inside an axis-aligned rectangle."""
    def edge(pts, inside, cross):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def at_x(x):
        return lambda p, q: (x, p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0]))

    def at_y(y):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]), y)

    pts = edge(poly, lambda p: p[0] >= x0, at_x(x0))
    pts = edge(pts, lambda p: p[0] <= x1, at_x(x1))
    pts = edge(pts, lambda p: p[1] >= y0, at_y(y0))
    return edge(pts, lambda p: p[1] <= y1, at_y(y1))


def _path(pts: list) -> str:
    out = []
    for x, y in pts:
        s = f"{x:.1f} {y:.1f}"
        if not out or out[-1] != s:
            out.append(s)
    return "M" + " L".join(out) + "Z"


def chariot_pieces() -> list:
    """Ten pieces: the capital's two halves (1, 2), six drums of the shaft (3-8), the base's two halves (9, 10).
    Each: its outline, where its balloon sits in its own coordinates, and how far it moves when taken apart."""
    g = glyph_i()
    regions = [(0, 0, CH_CX, CH_CUTS[0]), (CH_CX, 0, CH_W, CH_CUTS[0])]
    regions += [(0, a, CH_W, b) for a, b in zip(CH_CUTS, CH_CUTS[1:])]
    regions += [(0, CH_CUTS[-1], CH_CX, CH_H), (CH_CX, CH_CUTS[-1], CH_W, CH_H)]
    pieces = []
    for i, reg in enumerate(regions, start=1):
        pts = _clip(g, *reg)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        left = i % 2 == 1
        if i <= 2:
            anchor_y = 52                      # the middle of the capital's slab
        elif i >= 9:
            anchor_y = 588                     # the middle of the base's slab
        else:
            anchor_y = (min(ys) + max(ys)) / 2
        cx = (min(xs) + max(xs)) / 2
        dx = CH_COLS[0 if left else 1] - cx
        dy = CH_ROWS[(i - 1) // 2] - anchor_y
        pieces.append({"n": i, "pts": pts, "left": left, "box": (min(xs), min(ys), max(xs), max(ys)),
                       "anchor_y": anchor_y, "dx": dx, "dy": dy})
    return pieces


def render_chariot(parts: list) -> str:
    pieces = chariot_pieces()
    by_n = {p["n"]: p for p in parts}
    binder = by_n[11]
    desc = ("Taken apart, the parts are: " + "; ".join(
        f'{p["n"]}, {esc(p["short"])}, {STATUS_WORDS[p["status"]].lower()}' for p in parts[:10]) +
        f'. A dashed outline of the whole letter stays in the middle: the binder, {STATUS_WORDS[binder["status"]].lower()}.')
    svg = [f'<svg class="ch-svg" viewBox="0 0 {CH_W} {CH_H}" role="img" aria-labelledby="ch-t ch-d">',
           '<title id="ch-t">The word "I" drawn from ten parts</title>',
           f'<desc id="ch-d">{desc}</desc>',
           '<defs><linearGradient id="ch-gild" gradientUnits="userSpaceOnUse" x1="432" y1="40" x2="668" y2="600">'
           '<stop offset="0" class="gs1"/><stop offset="0.38" class="gs2"/><stop offset="0.6" class="gs1"/>'
           '<stop offset="1" class="gs3"/></linearGradient></defs>',
           f'<g class="ch-binder" data-part="{binder["id"]}"><path class="ch-outline" d="{_path(glyph_i())}"/>'
           f'<text class="ch-bname" x="{CH_CX}" y="628" text-anchor="middle">{esc(binder["short"])}</text></g>']
    for pc in pieces:
        part = by_n[pc["n"]]
        bx = CH_BALLOON[0 if pc["left"] else 1] - pc["dx"]
        by = pc["anchor_y"]
        x0, _, x1, _ = pc["box"]
        if pc["left"]:
            line = (bx + 21, by, x0 - 8, by)
            tx, anchor = bx - 33, "end"
        else:
            line = (bx - 21, by, x1 + 8, by)
            tx, anchor = bx + 33, "start"
        svg.append(
            f'<g class="ch-pc st-{part["status"]}" data-part="{part["id"]}" '
            f'style="--dx:{pc["dx"]:.1f}px;--dy:{pc["dy"]:.1f}px;--i:{pc["n"]}">'
            f'<path class="ch-shape" d="{_path(pc["pts"])}"/>'
            f'<g class="ch-call"><line x1="{line[0]:.1f}" y1="{line[1]:.1f}" x2="{line[2]:.1f}" y2="{line[3]:.1f}"/>'
            f'<circle class="ch-ball" cx="{bx:.1f}" cy="{by:.1f}" r="21"/>'
            f'<text class="ch-num" x="{bx:.1f}" y="{by + 8:.1f}" text-anchor="middle">{pc["n"]}</text>'
            f'<text class="ch-name" x="{tx:.1f}" y="{by + 1:.1f}" text-anchor="{anchor}">{esc(part["short"])}</text>'
            f'<text class="ch-st" x="{tx:.1f}" y="{by + 21:.1f}" text-anchor="{anchor}">'
            f'{STATUS_WORDS[part["status"]]}</text></g></g>')
    svg.append("</svg>")
    items = []
    for p in parts:
        n = p["n"] if p["n"] <= 10 else "?"
        items.append(f'<li><a class="ch-item st-{p["status"]}" href="#{p["id"]}" data-part="{p["id"]}">'
                     f'<span class="n">{n}</span><span class="nm">{esc(p["short"])}</span>'
                     f'<span class="st">{STATUS_WORDS[p["status"]]}</span></a></li>')
    counts = {s: sum(1 for p in parts[:10] if p["status"] == s) for s in STATUS_WORDS}
    return (
        '<div class="chariot" id="chariot-fig" data-state="apart">'
        '<div class="ch-plate">' + "".join(svg) + '</div>'
        '<div class="ch-bar"><button type="button" class="ch-toggle" hidden>Take it apart</button>'
        '<ul class="ch-key" aria-label="How a part is drawn when the word is taken apart">'
        f'<li><span class="sw sw-tested" aria-hidden="true"></span>Solid: tested here ({counts["tested"]})</li>'
        f'<li><span class="sw sw-known" aria-hidden="true"></span>Outline: known elsewhere ({counts["known"]})</li>'
        f'<li><span class="sw sw-untested" aria-hidden="true"></span>Dashed: untested ({counts["untested"]})</li>'
        '</ul></div></div>'
        '<ol class="ch-list" aria-label="The parts of the self, and the binder">' + "".join(items) + '</ol>'
    )


def render_chariot_mini(parts: list) -> str:
    """The chariot, a little apart, as a quiet figure for the Aim room's door to the Self room."""
    by_n = {p["n"]: p for p in parts}
    out = ['<svg class="ch-mini" viewBox="352 22 396 596" aria-hidden="true" focusable="false">',
           f'<path class="ch-outline" d="{_path(glyph_i())}"/>']
    for pc in chariot_pieces():
        k = 0.16
        out.append(f'<path class="ch-shape st-{by_n[pc["n"]]["status"]}" transform="translate({pc["dx"] * k:.1f} '
                   f'{pc["dy"] * k:.1f})" d="{_path(pc["pts"])}"/>')
    out.append("</svg>")
    return "".join(out)


def mark_i() -> str:
    """The wordmark's small gold I, from the same letter."""
    return (f'<svg class="mark-i" viewBox="420 30 260 580" aria-hidden="true" focusable="false">'
            f'<path d="{_path(glyph_i())}"/></svg>')


# ----------------------------------------------------------------------------------------------
# The Aim room's motif: Indra's net, a net of jewels in which each jewel reflects the others
# ----------------------------------------------------------------------------------------------

def render_net() -> str:
    W, H, cols, rows, gx, gy = 560, 330, 10, 7, 58.0, 50.0
    fx, fy, reach = 380.0, 150.0, 340.0
    nodes = {}
    for r in range(rows):
        for c in range(cols):
            x = 16 + c * gx + (gx / 2 if r % 2 else 0)
            y = 16 + r * gy
            if x > W - 6:
                continue
            nodes[(r, c)] = (x, y, max(0.0, 1 - math.hypot(x - fx, (y - fy) * 1.3) / reach))

    def near(r, c):
        out = [(r, c + 1)]
        if r % 2:
            out += [(r + 1, c), (r + 1, c + 1)]
        else:
            out += [(r + 1, c - 1), (r + 1, c)]
        return [k for k in out if k in nodes]

    def nbrs(r, c):
        cand = [(r, c - 1), (r, c + 1)] + ([(r - 1, c), (r - 1, c + 1), (r + 1, c), (r + 1, c + 1)] if r % 2 else
                                           [(r - 1, c - 1), (r - 1, c), (r + 1, c - 1), (r + 1, c)])
        return [k for k in cand if k in nodes]

    out = [f'<svg class="net" viewBox="0 0 {W} {H}" aria-hidden="true" focusable="false">']
    for (r, c), (x, y, o) in nodes.items():
        for k in near(r, c):
            x2, y2, o2 = nodes[k]
            op = min(o, o2) * 0.75
            if op < 0.04:
                continue
            d = math.hypot(x2 - x, y2 - y)
            ux, uy = (x2 - x) / d, (y2 - y) / d
            out.append(f'<line class="net-t" x1="{x + ux * 6:.1f}" y1="{y + uy * 6:.1f}" x2="{x2 - ux * 6:.1f}" '
                       f'y2="{y2 - uy * 6:.1f}" opacity="{op:.2f}"/>')
    for i, ((r, c), (x, y, o)) in enumerate(nodes.items()):
        if o < 0.05:
            continue
        glint = " net-glint" if (r * 7 + c * 3) % 5 == 0 and o > 0.35 else ""
        out.append(f'<g class="net-j{glint}" opacity="{o:.2f}" style="--d:{(i * 0.37) % 4:.2f}s">'
                   f'<circle class="net-o" cx="{x:.1f}" cy="{y:.1f}" r="4.6"/>')
        for k in nbrs(r, c):
            x2, y2, _ = nodes[k]
            d = math.hypot(x2 - x, y2 - y)
            out.append(f'<circle class="net-r" cx="{x + (x2 - x) / d * 2.1:.2f}" cy="{y + (y2 - y) / d * 2.1:.2f}" r="0.75"/>')
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------------------------
# The Self room's K1 chart, from the targets table in self_work.html
# ----------------------------------------------------------------------------------------------

K1_ROW_RE = re.compile(r'<tr><td>Qwen2\.5 ([0-9.]+B)</td><td class="num">(\.\d{3})</td></tr>')


def parse_k1() -> list:
    src = content("self_work.html")
    body = re.search(r'<table class="k1">.*?<tbody>(.*?)</tbody>', src, re.S).group(1)
    rows = [(m, float(v)) for m, v in K1_ROW_RE.findall(body)]
    assert len(rows) == 5, "self_work.html: expected K1's five targets"
    return rows


def render_k1() -> str:
    rows = parse_k1()
    W, H, left, right, top, bottom = 680, 300, 64, 24, 26, 58
    lo, hi = 0.09, 0.145
    pw, ph = W - left - right, H - top - bottom

    def Y(v):
        return top + ph - ph * (v - lo) / (hi - lo)

    step = pw / len(rows)
    out = [f'<svg class="k1-svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="k1-t">',
           '<title id="k1-t">K1\'s five targets, each with its match band (plus or minus 0.005) and its near band '
           '(plus or minus 0.010). K1 passed: the lab\'s own scores fall within 0.0003 of every target.</title>']
    def dec(v, places):
        return f"{v:.{places}f}".lstrip("0")         # .10, .102: the paper's own way of writing them

    for t in (0.09, 0.10, 0.11, 0.12, 0.13, 0.14):
        y = Y(t)
        out.append(f'<line class="k1-grid" x1="{left}" y1="{y:.1f}" x2="{W - right}" y2="{y:.1f}"/>')
        out.append(f'<text class="k1-tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{dec(t, 2)}</text>')
    for i, (model, v) in enumerate(rows):
        cx = left + step * (i + 0.5)
        w = min(64.0, step * 0.56)
        out.append(f'<g class="k1-m"><title>Qwen2.5 {model}: target {dec(v, 3)}; a match is {dec(v - .005, 3)} to '
                   f'{dec(v + .005, 3)}, near is {dec(v - .010, 3)} to {dec(v + .010, 3)}</title>'
                   f'<rect class="k1-near" x="{cx - w / 2:.1f}" y="{Y(v + .010):.1f}" width="{w:.1f}" '
                   f'height="{Y(v - .010) - Y(v + .010):.1f}" rx="3"/>'
                   f'<rect class="k1-match" x="{cx - w / 2:.1f}" y="{Y(v + .005):.1f}" width="{w:.1f}" '
                   f'height="{Y(v - .005) - Y(v + .005):.1f}" rx="3"/>'
                   f'<line class="k1-tgt" x1="{cx - w / 2 - 6:.1f}" y1="{Y(v):.1f}" x2="{cx + w / 2 + 6:.1f}" y2="{Y(v):.1f}"/>'
                   f'<circle class="k1-dot" cx="{cx:.1f}" cy="{Y(v):.1f}" r="5"/>'
                   f'<text class="k1-v" x="{cx + w / 2 + 10:.1f}" y="{Y(v) + 4:.1f}">{dec(v, 3)}</text>'
                   f'<text class="k1-x" x="{cx:.1f}" y="{H - bottom + 22}" text-anchor="middle">{model}</text></g>')
    out.append(f'<text class="k1-ax" x="{left + pw / 2:.1f}" y="{H - 10}" text-anchor="middle">Qwen2.5 base model, '
               'against DINOv2 ViT-B/14</text>')
    out.append(f'<text class="k1-ax" x="14" y="{top + ph / 2:.1f}" text-anchor="middle" '
               f'transform="rotate(-90 14 {top + ph / 2:.1f})">mutual 10-NN</text>')
    out.append("</svg>")
    return "".join(out)


def render_knn() -> str:
    """The mutual k-NN toy: its frame and controls; site.js draws the points."""
    return (
        '<figure class="toy" id="knn">'
        '<div class="toy-panels"><div class="toy-p"><div class="toy-h">Model A</div>'
        '<svg class="toy-svg" id="knn-a" viewBox="0 0 300 300" role="img" aria-label="Model A\'s points"></svg></div>'
        '<div class="toy-p"><div class="toy-h">Model B</div>'
        '<svg class="toy-svg" id="knn-b" viewBox="0 0 300 300" role="img" aria-label="Model B\'s points"></svg></div>'
        '<div class="toy-side"><div class="toy-score"><span class="toy-k">Mutual 3-NN</span>'
        '<output id="knn-score" aria-live="polite">—</output><span class="toy-chance" id="knn-chance"></span></div>'
        '<label class="toy-ctl"><span>Rotate B</span><input type="range" id="knn-rot" min="0" max="360" value="40" '
        'step="1"></label>'
        '<label class="toy-ctl"><span>Noise in B</span><input type="range" id="knn-noise" min="0" max="100" value="18" '
        'step="1"></label>'
        '<p class="toy-pick" id="knn-pick">Pick a point in either model.</p></div></div>'
        '<noscript><p class="small">The toy draws its points with JavaScript, which is off.</p></noscript>'
        '<figcaption>A toy with made-up points, not data. Model B sees the same twenty-four items as model A, turned '
        'and blurred. A picked point shows its three nearest neighbours in each model, gold where the two models agree. '
        'The score is the average share of neighbours both models pick. Turning B leaves it where it is, because only '
        'the relations count; noise lowers it.</figcaption>'
        '</figure>'
    )


# ----------------------------------------------------------------------------------------------
# The Record's field of marks: one per graded prediction, from the scorecard table
# ----------------------------------------------------------------------------------------------

SC_ROW_RE = re.compile(r'<tr><td class="num">(.*?)</td><td>(.*?)</td><td class="num">(.*?)</td><td>(.*?)</td></tr>', re.S)
SC_TAG_RE = re.compile(r'<span class="tag (hit|miss|open)">([^<]*)</span>')
MARK_WORDS = {"hit": ("Hit", "hits"), "miss": ("Miss", "misses"), "open": ("Open", "not read or uninformative")}


def tally(src: str) -> dict:
    """The scorecard's tally, counted the way the checker's script counts it."""
    body = src[src.index("<h2>Predictions graded so far</h2>"):]
    body = body[body.index("<tbody>"):body.index("</tbody>")]
    c = {"hit": 0, "miss": 0, "open": 0}
    for row in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        for kind, text in SC_TAG_RE.findall(row):
            m = re.search(r"×(\d+)", text)
            c[kind] += int(m.group(1)) if m else 1
    return c


def parse_scorecard() -> list:
    src = content("scorecard.html")
    body = src[src.index("<tbody>"):src.index("</tbody>")]
    rows = []
    for num, pred, odds, result in SC_ROW_RE.findall(body):
        bits = SC_TAG_RE.split(result)
        tags = []
        for i in range(1, len(bits), 3):
            kind, label, after = bits[i], bits[i + 1], bits[i + 2]
            m = re.search(r"×(\d+)", label)
            tags.append({"kind": kind, "label": label, "n": int(m.group(1)) if m else 1,
                         "flag": after.strip().lower().startswith("flagged")})
        rows.append({"num": num, "pred": pred, "odds": odds, "tags": tags})
    counted = {k: sum(t["n"] for r in rows for t in r["tags"] if t["kind"] == k) for k in MARK_WORDS}
    assert counted == tally(src), f"scorecard.html: the field's rows don't cover the table ({counted} vs {tally(src)})"
    return rows


def render_field(rows: list) -> str:
    marks = []                                       # (kind, flagged, row) in ledger order
    for ri, r in enumerate(rows):
        for t in r["tags"]:
            marks.extend((t["kind"], t["flag"], ri) for _ in range(t["n"]))
    counts = {k: sum(1 for m in marks if m[0] == k) for k in MARK_WORDS}
    flagged = sum(1 for m in marks if m[1])

    def mk(i):
        kind, flag, ri = marks[i]
        return f'<span class="mk {kind}{" flag" if flag else ""}" data-r="{ri}" data-k="{i}"></span>'

    groups = []
    for kind, (word, plural) in MARK_WORDS.items():
        idx = [i for i, m in enumerate(marks) if m[0] == kind]
        note = f' <span class="fd-note">{flagged} flagged</span>' if kind == "hit" and flagged else ""
        groups.append(f'<div class="fd-group fd-{kind}"><div class="fd-h"><span class="fd-c">{len(idx)}</span>'
                      f'<span class="fd-w">{plural}</span>{note}</div>'
                      f'<div class="fd-marks">{"".join(mk(i) for i in idx)}</div></div>')
    clusters, i = [], 0
    for ri, r in enumerate(rows):
        n = sum(t["n"] for t in r["tags"])
        clusters.append(f'<div class="fd-row"><div class="fd-marks">{"".join(mk(j) for j in range(i, i + n))}</div>'
                        f'<span class="fd-n">{r["num"]}</span></div>')
        i += n
    total = len(marks)
    label = (f'One mark for each of the {total} graded predictions: {counts["hit"]} hits ({flagged} of them flagged), '
             f'{counts["miss"]} misses and {counts["open"]} not read or uninformative. The table below lists them all.')
    return (
        f'<div class="field-wrap"><div class="field-bar">'
        f'<ul class="field-key" aria-hidden="true"><li><span class="mk hit"></span>hit</li>'
        f'<li><span class="mk hit flag"></span>hit, flagged</li><li><span class="mk miss"></span>miss</li>'
        f'<li><span class="mk open"></span>not read or uninformative</li></ul>'
        f'<button type="button" class="field-order" hidden aria-pressed="false">In table order</button></div>'
        f'<div class="field" id="field" role="img" aria-label="{esc(label)}" data-total="{total}">'
        f'<div class="fd-view fd-by-result">{"".join(groups)}</div>'
        f'<div class="fd-view fd-by-ledger" hidden>{"".join(clusters)}</div></div>'
        f'<p class="field-cap">Each mark is one prediction, written with its odds before its data. Point at a mark to '
        f'read it; the table below has every one.</p></div>'
    )


def render_pager(i: int) -> str:
    out = ['<nav class="pager" aria-label="Previous and next room">']
    if i > 0:
        name, label, _ = ROOMS[i - 1]
        out.append(f'<a class="pg pg-prev" href="#{name}"><span class="pg-k">Previous room</span>'
                   f'<span class="pg-t">{label}</span></a>')
    else:
        out.append('<span class="pg pg-none"></span>')
    if i < len(ROOMS) - 1:
        name, label, _ = ROOMS[i + 1]
        out.append(f'<a class="pg pg-next" href="#{name}"><span class="pg-k">Next room</span>'
                   f'<span class="pg-t">{label}</span></a>')
    out.append("</nav>")
    return "".join(out)


# ----------------------------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------------------------

PLACEHOLDER_RE = re.compile(r"\{\{([a-z0-9_]+)(?::([^}]*))?\}\}")
# A section's rail: its number, then its label, which is wrapped so the stylesheet can set it on its side.
RAIL_RE = re.compile(r'(<div class="rail"><span class="n">[^<]*</span>)([^<]+)(</div>)')


def fill(template: str, values: dict, funcs: dict) -> str:
    def sub(m):
        name, arg = m.group(1), m.group(2)
        if arg is not None:
            if name not in funcs:
                raise KeyError(f"unknown placeholder {{{{{name}:...}}}}")
            return funcs[name](arg)
        if name not in values:
            raise KeyError(f"unknown placeholder {{{{{name}}}}}")
        return values[name]
    return PLACEHOLDER_RE.sub(sub, template)


def build() -> str:
    entries = load_ideas()
    standing = parse_standing()
    frame = parse_frame()
    plan = parse_plan()
    parts = parse_parts()
    log = render_log(entries)
    values = {
        "hero": fill(content("hero.html"), {"net": render_net()}, {}),
        "chariot": render_chariot(parts),
        "chariot_mini": render_chariot_mini(parts),
        "knn": render_knn(),
        "k1": render_k1(),
        "field": render_field(parse_scorecard()),
        "glance_eyebrow": standing["eyebrow"],
        "glance_h2": standing["h2"],
        "glance_lede": standing["lede"],
        "glance": render_glance(standing),
        "frame_h2": frame["h2"],
        "frame_intro": frame["intro"],
        "frame_strip": render_frame_strip(frame),
        "frame_outro": frame["outro"],
        "instruments": render_instruments(parse_calibration()),
        "plan_lede": plan["lede"],
        **log,
    }
    funcs = {
        "stands": lambda t: render_stands(standing, t),
        "finding": lambda t: render_glance_full(standing, t),
        "next": lambda arg: render_next(plan, [int(x) for x in arg.split(",")]),
        "glyph": glyph,
    }
    nav = "".join(f'<a href="#{name}" data-room="{name}">{label}</a>' for name, label, _ in ROOMS)
    views = []
    for i, (name, label, files) in enumerate(ROOMS):
        body = RAIL_RE.sub(r'\1<span class="rl">\2</span>\3',
                           "\n".join(fill(content(f), values, funcs) for f in files))
        views.append(f'<div class="view" id="{name}" data-view="{name}" role="region" aria-label="{label}">\n{body}\n'
                     f'{render_pager(i)}\n</div>')
    footer = content("footer.html").rstrip()
    built = (f'<p class="built">Built from the lab\'s files by lab/site/so_far/build.py. The ideas log\'s '
             f'{len(entries)} entries are read from lab/ideas/ideas.md each time the page is rebuilt. Aim is '
             f'assembled from the hand-kept page\'s own text, with new reading paths, a key to the two kinds of '
             f'statement and a door to the Self room. The Self room, the Possibilities room and the glossary were '
             f'written for this site. The other rooms are carried over from the hand-kept page, with Limits boxes and '
             f'glossary links added. The chariot, the K1 chart and the field of marks are drawn from those files when '
             f'the page is built.</p>')
    footer = footer.replace("</footer>", built + "\n</footer>")
    figures = content("figures.js")
    fig_ids = re.findall(r"getElementById\('(fig-[a-z-]+)'\)", figures)
    script = ("function drawFigures() {\n"
              f"  {sorted(set(fig_ids))!r}.forEach(function (id) {{ var f = document.getElementById(id); "
              "if (f) f.textContent = ''; });\n"
              + figures + "}\n" + content("site.js"))
    page = "\n".join([
        "<title>Anatta So Far</title>",
        '<meta name="claude:viewport-fit-sides" content="cover">',
        f'<link rel="stylesheet" href="{FONTS}">',
        "<style>",
        content("site.css").rstrip(),
        "</style>",
        "",
        '<script type="application/json" id="data">',
        content("data.json").rstrip(),
        "</script>",
        "",
        f'<div class="band"><div class="band-in"><a class="mark" href="#aim">{mark_i()}<span>Anatta</span> '
        f'<i>so far</i></a><nav class="rooms" aria-label="Rooms">{nav}</nav></div></div>',
        '<div class="page">',
        "\n".join(views),
        footer,
        '<div class="pop" id="pop" role="tooltip" hidden></div>',
        "</div>",
        "",
        "<script>",
        script.rstrip(),
        "</script>",
        "",
    ])
    return page


def dump(entries: list):
    for e in entries:
        print(f"\n## {e.n}. {e.title}   [{e.group}]   (private bullets dropped: {e.dropped_private})")
        print(f"  summary ({e.summary_from}): {e.summary}")
        print(f"  test: {e.test}")
        print(f"  status ({e.status_label}): {e.status}")
        for where, text, reason in e.omissions:
            print(f"  OMITTED [{reason}]: {text}")
    print("\ngroups:", {g: [e.n for e in entries if e.group == g] for g, _, _ in GROUPS})


def main(argv: list) -> int:
    if "--dump" in argv:
        dump(load_ideas())
        return 0
    page = build()
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(page.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
