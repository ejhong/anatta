#!/usr/bin/env python3
"""Write the page the public site serves: the built page as a full document, with its icon and link card.

index.html follows the Artifact page contract: no doctype, html, head or body tags, and no link but the fonts. That
suits claude.ai, which supplies the document around the page. GitHub Pages serves a file as it is, so until 16 Nov
2026 the public page had no doctype (browsers fell back to quirks mode), no viewport setting (a phone drew the desktop
layout, zoomed out), no icon and none of the tags a link preview is built from. This script wraps index.html in a
document that has them and copies the files in public/ beside it:

    favicon.svg, favicon.png    the tab icon: the page's gold I on its indigo
    apple-touch-icon.png        the same, for a phone's home screen
    card.png                    the picture a link preview shows, 1200 by 630

Run by the pages workflow, or by hand:   python3 lab/site/so_far/publish.py OUT_DIR

Standard library only. It changes no word of the page. Claude (Fable 5.2, an AI model) wrote it on 16 Nov 2026.
"""
from __future__ import annotations

import html
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"
ASSETS = HERE / "public"

SITE_URL = "https://ejhong.github.io/anatta/"
TITLE = "Anatta: is awakening computational?"
CARD_TITLE = "Is awakening computational?"
DESCRIPTION = ("An open lab asks whether goodness, as Anthony de Mello describes it, can be computed. Experiments on "
               "small open AI models, with every negative shown. Written by Claude (Anthropic) at Eugene Jhong's "
               "direction.")
CARD_ALT = ("The words \"Is awakening computational?\" in ivory and gold on indigo, above the line \"Goodness as "
            "Anthony de Mello describes it, taken seriously and applied to AI\", beside a drawing of Indra's net.")
FILES = ("favicon.svg", "favicon.png", "apple-touch-icon.png", "card.png")

# The built page opens with these two lines, which belong to the Artifact contract and not to a document's head.
OWN_TITLE = "<title>Anatta So Far</title>\n"
OWN_META = '<meta name="claude:viewport-fit-sides" content="cover">\n'


def split(page: str) -> tuple[str, str]:
    """The built page as (what belongs in a head, what belongs in a body): its fonts link and its style, then
    everything after the style."""
    if not page.startswith(OWN_TITLE + OWN_META):
        raise ValueError("index.html does not open as build.py writes it")
    rest = page[len(OWN_TITLE + OWN_META):]
    cut = rest.index("</style>") + len("</style>")
    return rest[:cut], rest[cut:].lstrip("\n")


def head_tags() -> str:
    e = lambda s: html.escape(s, quote=True)  # noqa: E731
    card = SITE_URL + "card.png"
    return "\n".join([
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        f"<title>{e(TITLE)}</title>",
        f'<meta name="description" content="{e(DESCRIPTION)}">',
        f'<link rel="canonical" href="{SITE_URL}">',
        '<link rel="icon" href="favicon.svg" type="image/svg+xml">',
        '<link rel="icon" href="favicon.png" type="image/png" sizes="96x96">',
        '<link rel="apple-touch-icon" href="apple-touch-icon.png">',
        '<meta name="theme-color" content="#0e1934" media="(prefers-color-scheme: dark)">',
        '<meta name="theme-color" content="#f1f0eb" media="(prefers-color-scheme: light)">',
        '<meta property="og:type" content="website">',
        '<meta property="og:site_name" content="Anatta">',
        f'<meta property="og:title" content="{e(CARD_TITLE)}">',
        f'<meta property="og:description" content="{e(DESCRIPTION)}">',
        f'<meta property="og:url" content="{SITE_URL}">',
        f'<meta property="og:image" content="{card}">',
        '<meta property="og:image:type" content="image/png">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        f'<meta property="og:image:alt" content="{e(CARD_ALT)}">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{e(CARD_TITLE)}">',
        f'<meta name="twitter:description" content="{e(DESCRIPTION)}">',
        f'<meta name="twitter:image" content="{card}">',
        f'<meta name="twitter:image:alt" content="{e(CARD_ALT)}">',
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
    ])


def wrap(page: str) -> str:
    """The public document: a head with the tags above and the page's own fonts and style, then the page."""
    own_head, body = split(page)
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n" + head_tags() + "\n" + own_head + "\n</head>\n<body>\n"
            + body.rstrip("\n") + "\n</body>\n</html>\n")


def png_size(path: Path) -> tuple[int, int]:
    """Width and height from a PNG's header."""
    b = path.read_bytes()[:24]
    if b[:8] != b"\x89PNG\r\n\x1a\n" or b[12:16] != b"IHDR":
        raise ValueError(f"{path.name} is not a PNG")
    return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")


def publish(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(wrap(PAGE.read_text(encoding="utf-8")), encoding="utf-8")
    for name in FILES:
        shutil.copyfile(ASSETS / name, out / name)


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__.split("\n\n")[0] + "\n\nusage: publish.py OUT_DIR", file=sys.stderr)
        return 2
    out = Path(argv[1])
    publish(out)
    print(f"wrote {out / 'index.html'} and {len(FILES)} files beside it")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
