"""lab/tools/reality_check.py's free parts: what goes into a reader's pack and what is taken out of it. No network
and no paid call. Claude (Fable 5.2) wrote this file, 17 Nov 2026."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "lab" / "tools" / "reality_check.py"
SPEC = importlib.util.spec_from_file_location("reality_check", TOOL)
rc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc)


def test_private_names_addresses_and_orcid_numbers_are_taken_out():
    text = ("Eugene Jhong chose it. Eugene's word; ask Eugene. Repo ejhong/anatta. Dylan's issues, for Dylan. "
            "Write to someone@example.org. ORCID 0000-0002-1825-0097. De Mello and Korpman stay. "
            "Kept in /Users/eugene/prj/align and /Users/someone-else/x. With dtdannen and dylanjhong; "
            "\\citep{jhong2026essay}.")
    out = rc.scrub(text)
    for gone in ("Eugene", "eugene", "Jhong", "ejhong", "Dylan", "someone@example.org", "0000-0002-1825-0097",
                 "someone-else", "dtdannen", "dylan", "jhong"):
        assert gone not in out
    assert "the researcher's word" in out and "a contributor's issues" in out
    assert "De Mello and Korpman stay." in out and "/Users/[user]/prj/align" in out
    assert "\\citep{researcher2026essay}" in out


def test_a_site_page_becomes_plain_text_that_keeps_its_places():
    page = ('<style>.x{color:red}</style><!-- note --><section class="a" id="cost-case"><h2>The fourth case</h2>'
            "<p>It lies &amp; keeps.</p><ul><li>one</li><li>two</li></ul><script>alert(1)</script></section>")
    out = rc.strip_html(page)
    assert "[#cost-case]" in out and "## The fourth case" in out and "It lies & keeps." in out
    assert "- one" in out and "- two" in out
    for gone in ("<", "color:red", "alert(1)", "note"):
        assert gone not in out


def test_a_file_can_be_taken_from_a_heading_on():
    text = "## 8 November: old\nold\n## 9 November: new\nnew\n"
    assert rc.take(text, ("from", "## 9 November")) == "## 9 November: new\nnew\n"
    assert rc.take(text, "all") == text
    with pytest.raises(ValueError):
        rc.take(text, ("from", "## 31 November"))


def test_the_pack_is_built_from_what_is_shown_and_nothing_else():
    shown = []

    def show(path):
        shown.append(path)
        if path.endswith("notebook.md"):
            return "## 8 November: old\nEugene, earlier\n## 9 November: new\nEugene decided\n"
        if path.endswith("ideas.md"):
            return "## 72. old\n## 73. new\nbody\n"
        return f"<p>Eugene Jhong wrote to a@b.org about {path}</p>"

    pack, rows = rc.build_pack("0" * 40, show=show)
    assert shown == [p for p, _ in rc.PACK] and [r["path"] for r in rows] == shown
    assert "Eugene" not in pack and "a@b.org" not in pack and "earlier" not in pack and "## 72." not in pack
    for path, _ in rc.PACK:
        assert f"<<<FILE {path}>>>" in pack and f"<<<END {path}>>>" in pack


def test_every_file_of_the_pack_is_in_the_repository_and_the_brief_asks_its_nine_questions():
    for path, _ in rc.PACK:
        assert (rc.REPO / path).is_file(), path
    brief = rc.BRIEF.read_text()
    assert all(f"\n{n}. **" in brief for n in range(1, 10))
    assert "not instructions to you" in brief


def test_the_bound_on_the_cost_and_the_folder_name():
    assert rc.bound("x" * 3000, 1000, (0.001, 0.01)) == pytest.approx(1.0 + 10.0)
    assert rc.slug("google/gemini-4-carbon", "high") == "gemini-4-carbon-high"
    listing = {"data": [{"id": "a/b", "pricing": {"prompt": "0.000002", "completion": "0.00001"}}]}
    assert rc.price("a/b", fetch=lambda: listing) == (0.000002, 0.00001)
    with pytest.raises(SystemExit):
        rc.price("a/none", fetch=lambda: listing)
