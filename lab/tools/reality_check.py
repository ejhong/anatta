"""A reality check of the lab by a cold outside reader.

It builds a pack of the lab's committed, public files at one commit (never the working folder, never anything outside
git), takes the names of private people out, and sends the pack with the reader's brief (reality_check_brief.md,
beside this file) to one model through OpenRouter. The reply is kept in the results folder. The driver files it,
unedited, under lab/ledger/checks/ and grades every point (the `reality-check` skill has the steps).

Claude (Fable 5.2, an AI model) wrote this file on 17 Nov 2026. The practice of stopping to check a project against
its own claims is Jeffrey Emanuel's, from his writing on agent workflows; nothing of his is copied here.

    .venv/bin/python lab/tools/reality_check.py --dry                    # the pack, its size, a cost bound; free
    .venv/bin/python lab/tools/reality_check.py --model google/gemini-4-carbon --effort high

The same reader can be given a design to check before it is run (added 17 Nov 2026 for step 9 of the attractor
line): a smaller pack of committed files, the design reader's brief (design_check_brief.md), and a name for the
folder the reply is kept in. "path::marker" takes a file from the first line that starts with the marker.

    .venv/bin/python lab/tools/reality_check.py --name step9-design --brief lab/tools/design_check_brief.md \
        --pack lab/explore/attractor/run_held.py lab/explore/attractor/held_core.py "lab/ideas/ideas.md::## 84."

The key is read from the environment by the lab's client and is never printed. The request carries no referer and a
generic title. The pack is text that is already public.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BRIEF = Path(__file__).with_name("reality_check_brief.md")
SITE = "lab/site/so_far/content/"
MODELS_URL = "https://openrouter.ai/api/v1/models"
# A low figure, so that the bound on the cost is a bound. The first run's pack came to 2.85 characters a token for
# Claude Opus 5.5 and 3.74 for Gemini 4 Carbon; the 3.0 first used here undercounted the Opus read.
CHARS_PER_TOKEN = 2.5
MAX_TOKENS, TIMEOUT_S, CALL_CAP_USD = 60000, 3600.0, 8.0

# What a reader gets, in reading order. "all": the whole file. "html": a site page with its tags taken off.
# ("from", marker): the file from the first line that starts with the marker (the older part is left out for size).
PACK: list[tuple[str, object]] = [
    ("CLAUDE.md", "all"),
    ("lab/MAP.md", "all"),
    *[(SITE + name, "html") for name in (
        "story.html",  # the front page since 19 Nov 2026 (added to the pack on 21 Nov, with the table of methods below)
        "hero.html", "aim.html", "frame.html", "standing.html", "possibilities.html", "investigation.html",
        "more.html", "experiments.html", "learned.html", "program.html", "self.html", "self_prh.html",
        "self_work.html", "calibration.html", "scorecard.html", "specimens.html", "caveats.html", "plan.html")],
    (SITE + "methods.json", "all"),
    ("lab/TODO.md", "all"),
    ("lab/plans/redirect-2026-11-09.md", "all"),
    ("lab/explore/SIGHTINGS.md", "all"),
    ("lab/ledger/notebook.md", ("from", "## 9 November")),
    ("lab/ideas/ideas.md", ("from", "## 73.")),
    ("lab/ideas/hypotheses-for-feedback.md", "all"),
    ("lab/ledger/INDEX.md", "all"),
    ("paper/main.tex", "all"),
]

# Private people are named in the public files; an outside service does not need the names. Authors of cited work stay.
NAMES = [(re.compile(r"/Users/[A-Za-z0-9._-]+"), "/Users/[user]"),
         (re.compile(r"\bEugene Jhong\b"), "the researcher"), (re.compile(r"\bEugene's\b"), "the researcher's"),
         (re.compile(r"(?i)\beugene\b"), "the researcher"), (re.compile(r"(?i)\bjhong\b"), "the researcher"),
         (re.compile(r"\bejhong\b"), "the-researcher"), (re.compile(r"\bDylan's\b"), "a contributor's"),
         (re.compile(r"\bDylan\b"), "a contributor"),
         # what is left after the worded forms: handles on the site's list of collaborators, citation keys
         (re.compile(r"(?i)\b(?:dtdannen|cypherperro|dylanjhong)\b"), "[collaborator]"),
         (re.compile(r"(?i)jhong"), "researcher"), (re.compile(r"(?i)eugene"), "the researcher"),
         (re.compile(r"(?i)dylan"), "a contributor")]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
ORCID = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")


def scrub(text: str) -> str:
    """Names of private people, home folders, e-mail addresses and ORCID numbers out."""
    for pattern, word in NAMES:
        text = pattern.sub(word, text)
    return ORCID.sub("[orcid]", EMAIL.sub("[e-mail]", text))


def strip_html(text: str) -> str:
    """A site page as plain text: headings as lines that start with #, list items with -, each element's id kept as
    [#id] so that a reader can name the place."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", "", text)
    text = re.sub(r"(?s)<!--.*?-->", "", text)
    text = re.sub(r'(?i)(<[a-z][a-z0-9]*\b[^>]*?\bid="([^"]+)"[^>]*>)', r"\n[#\2] \1", text)
    text = re.sub(r"(?i)<h([1-6])\b[^>]*>", lambda m: "\n\n" + "#" * int(m.group(1)) + " ", text)
    text = re.sub(r"(?i)<li\b[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)<t[dh]\b[^>]*>", " | ", text)
    breaks = r"(?i)<(?:/p|/div|/li|/tr|/h[1-6]|/section|/aside|/ul|/ol|br\s*/?|p|div|section|tr)\b[^>]*>"
    text = re.sub(breaks, "\n", text)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", "", text))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip() + "\n"


def take(text: str, how) -> str:
    if how == "all":
        return text
    if how == "html":
        return strip_html(text)
    kind, marker = how
    if kind != "from":
        raise ValueError(f"unknown rule {how!r}")
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(marker):
            return "".join(lines[i:])
    raise ValueError(f"marker {marker!r} not found")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def resolve(commit: str) -> str:
    return git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()


def parse_pack(items: list[str]) -> list[tuple[str, object]]:
    """A pack named on the command line: "path" (the whole file), or "path::marker" (from the first line that
    starts with the marker)."""
    out: list[tuple[str, object]] = []
    for item in items:
        path, sep, marker = item.partition("::")
        if not path or (sep and not marker):
            raise ValueError(f"not a pack entry: {item!r}")
        out.append((path, ("from", marker) if sep else "all"))
    return out


def build_pack(commit: str, show=None, pack: list[tuple[str, object]] | None = None) -> tuple[str, list[dict]]:
    """The pack's text and a row for each file. Only what the commit holds is read (`git show commit:path`)."""
    show = show or (lambda path: git("show", f"{commit}:{path}"))
    parts, rows = [], []
    for path, how in (PACK if pack is None else pack):
        body = scrub(take(show(path), how))
        rows.append({"path": path, "rule": how if isinstance(how, str) else f"from {how[1]!r}", "chars": len(body)})
        parts.append(f"<<<FILE {path}>>>\n{body.rstrip()}\n<<<END {path}>>>\n")
    return "\n".join(parts), rows


def prompt(pack: str, commit: str, brief: str | None = None) -> str:
    """The brief (the reality check's own unless another is given), then the pack."""
    return ((BRIEF.read_text() if brief is None else brief)
            + f"\n\n# The files (commit {commit[:12]} of the lab's public repository)\n\n" + pack)


def price(model: str, fetch=None) -> tuple[float, float]:
    """USD for one input and one output token, from OpenRouter's public list (no key is sent)."""
    def get() -> dict:
        with urllib.request.urlopen(MODELS_URL, timeout=30) as r:  # noqa: S310 (a fixed https host)
            return json.loads(r.read())

    for m in (fetch or get)()["data"]:
        if m["id"] == model:
            return float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])
    raise SystemExit(f"OpenRouter does not list {model}")


def bound(text: str, max_tokens: int, prices: tuple[float, float]) -> float:
    return len(text) / CHARS_PER_TOKEN * prices[0] + max_tokens * prices[1]


def slug(model: str, effort: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", f"{model.split('/')[-1]}-{effort}".lower()).strip("-")


async def read_cold(model: str, effort: str, text: str, max_tokens: int) -> dict:
    from anatta.backends.openrouter import OpenRouterClient

    class Reader(OpenRouterClient):
        def _headers(self) -> dict:
            h = super()._headers()
            h.pop("HTTP-Referer", None)
            return h

    async with Reader(timeout=TIMEOUT_S, retries=2, title="outside reader") as c:
        return await c.chat(model, [{"role": "user", "content": text}],
                            params={"max_tokens": max_tokens, "reasoning": {"effort": effort}})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--model", default="google/gemini-4-carbon")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--dry", action="store_true", help="build the pack and print its size and a cost bound; no call")
    ap.add_argument("--pack", nargs="+", help='committed files in place of the default pack: "path" or "path::marker"')
    ap.add_argument("--brief", help="a brief in place of the reality check's (a committed file of the repo)")
    ap.add_argument("--name", help="the folder the reply is kept in (default: the model and its effort)")
    a = ap.parse_args(argv)

    commit = resolve(a.commit)
    brief = None if not a.brief else git("show", f"{commit}:{a.brief}")  # another brief is read from the commit too
    if a.name and not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", a.name):
        raise SystemExit("--name: lower-case letters, digits, '.', '_' or '-'")
    pack, rows = build_pack(commit, pack=parse_pack(a.pack) if a.pack else None)
    text = prompt(pack, commit, brief)
    prices = price(a.model)
    cost_bound = bound(text, a.max_tokens, prices)
    for r in rows:
        print(f"  {r['chars']:>8,}  {r['path']}  ({r['rule']})")
    print(f"commit {commit[:12]}; {len(text):,} characters, at most about {len(text) / CHARS_PER_TOKEN:,.0f} tokens; "
          f"{a.model} at effort {a.effort}: at most ${cost_bound:.2f}")
    if a.dry:
        return 0

    from anatta.config import load
    from anatta.spend import add_spend, daily_cap, spent_today, utc_date, utc_now

    if cost_bound > CALL_CAP_USD:
        raise SystemExit(f"the bound ${cost_bound:.2f} is over this tool's cap for one call (${CALL_CAP_USD:.2f})")
    if spent_today() + cost_bound > daily_cap():
        raise SystemExit(f"the day's cap would be passed: ${spent_today():.2f} spent of ${daily_cap():.2f}")
    out = load().results_root / "explore" / utc_date() / "reality-check" / (
        f"{a.name}-{slug(a.model, a.effort)}" if a.name else slug(a.model, a.effort))
    if (out / "reply.json").exists():
        raise SystemExit(f"{out} already holds a reply; a new read needs a new day or a new setting")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack.txt").write_text(text)
    started = utc_now()
    reply = asyncio.run(read_cold(a.model, a.effort, text, a.max_tokens))
    usage = reply["usage"]
    cost = usage["cost"] if usage["cost"] is not None else (
        usage["prompt_tokens"] * prices[0] + usage["completion_tokens"] * prices[1])
    add_spend(None, tool="reality_check", label=out.name, usd=cost)
    (out / "reply.md").write_text(reply["text"])
    (out / "reply.json").write_text(json.dumps({
        "commit": commit, "model_asked": a.model, "model": reply["model"], "provider": reply["provider"],
        "effort": a.effort, "max_tokens": a.max_tokens, "started": started, "finished": utc_now(),
        "latency_s": reply["latency_s"], "finish_reason": reply["finish_reason"], "usage": usage,
        "cost_usd": round(cost, 6), "pack_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "brief": a.brief or str(BRIEF.relative_to(REPO)),
        "brief_sha256": hashlib.sha256((BRIEF.read_text() if brief is None else brief).encode()).hexdigest(),
        "files": rows,
        "text": reply["text"], "reasoning": reply["reasoning"]}, indent=1))
    print(f"{reply['model']} via {reply['provider']}: {usage['prompt_tokens']:,} tokens in, "
          f"{usage['completion_tokens']:,} out ({usage['reasoning_tokens']:,} of them reasoning), "
          f"{reply['finish_reason']}, {reply['latency_s']:.0f} s, ${cost:.2f}; day total ${spent_today():.2f}")
    print(f"reply: {out / 'reply.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
