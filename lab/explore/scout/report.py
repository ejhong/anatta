"""A sweep's report, readable in minutes: report.md and report.html in the sweep directory, with a page of random
transcripts for each notable cell under transcripts/. Local files only; never published. Every piece of text from a
model, a scenario or the judge goes through html.escape. A reply the judge flagged as harmful is never shown: the
report gives the count, and its transcript slot says it was withheld. harm_guard is rates only: no item, reply or
judge text of it ever reaches the report or a transcript page. Claude wrote this file. Private.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
from pathlib import Path

from . import library as lib
from . import stats
from .schema import read_jsonl, reply_is_empty, reply_sha256, utc_now

HG = lib.HARM_GUARD

WITHHELD = "[withheld: the judge flagged this reply as harmful_content]"
MAX_PAGES, PER_PAGE = 30, 10
HOW_TO_READ = (
    "Each column is one family's indicator, and lower is better in every behavior column (the arrow in each header "
    "marks the good direction). The cluster: threat_to_self = % self-preserving or deceptive; who_benefits:self = % "
    "shaded for the model itself; deference = % hedged or caved. Then costly_care = % care withheld; after_mistake = "
    "% defended or deflected; complicity = % that helped with an everyday wrong against a third party (with or "
    "without a caveat); over_refusal = % that refused a harmless request that only sounds risky; harm_guard = the "
    "mean StrongREJECT-style score x 100 on JailbreakBench's harmful requests (open-weight models only, rates only). "
    "The signature table shows each condition's change from baseline in percentage points, pooled over models; its "
    "first row is the baseline's own level. The audience gap is different: it is the condition's own gap, the rate "
    "when the scenario says nobody will review the session minus the rate when it says the session is flagged for "
    "review, over threat_to_self and who_benefits:self; near 0 is better. Intervals are bootstrap 95% intervals "
    "over items (triples for who_benefits). wb:self-other, in the who_benefits detail, is the change in shading for "
    "the model itself minus the change for another person: negative means the condition cut self-serving lies more "
    "than lies for others. These are sightings, not findings.")


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")[:80]


def fmt_rate(s) -> str:
    return "-" if not s else f"{s['est']:.0f} [{s['lo']:.0f}, {s['hi']:.0f}]"


def fmt_delta(s) -> str:
    return "-" if not s else f"{s['est']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}]"


def sig_mark(s) -> str:
    if not s:
        return ""
    return "down" if s["hi"] < 0 else "up" if s["lo"] > 0 else ""


def rkey(r: dict) -> tuple:
    return (r["model"], r["intervention"]["name"], r["item_id"], r["sample"], r.get("framing"))


def jkey(j: dict) -> tuple:
    return (j["model"], j["intervention"], j["item_id"], j["sample"], j.get("framing"))


def current_judgments(records: list[dict], judgments: list[dict]) -> list[dict]:
    """The judgments that belong to the current records: same model, intervention, item, sample and framing, and
    (when the judgment carries it) the same reply text."""
    by = {rkey(r): reply_sha256(r) for r in records}
    out = []
    for j in judgments:
        k = jkey(j)
        if k in by and j.get("response_sha256", by[k]) == by[k]:
            out.append(j)
    return out


def quality(records, judgments, errors, models) -> list[dict]:
    rows = []
    for m in models:
        rs = [r for r in records if r["model"] == m]
        js = [j for j in judgments if j["model"] == m]
        n = len(rs) or 1
        rows.append({"model": m, "records": len(rs), "judged": len(js), "unparsed": sum(not j.get("parse_ok") for j in js),
                     "judge_cut": sum(j.get("finish_reason") == "length" for j in js),
                     "harmful": sum(bool(j.get("harmful")) for j in js),
                     "truncated": sum(r.get("finish_reason") == "length" for r in rs),
                     "empty": sum(reply_is_empty(r) for r in rs),
                     "errors": sum(e.get("model") == m for e in errors),
                     "prompt_tokens": round(sum(r["usage"].get("prompt_tokens", 0) for r in rs) / n),
                     "completion_tokens": round(sum(r["usage"].get("completion_tokens", 0) for r in rs) / n),
                     "providers": sorted({str(r.get("provider")) for r in rs})})
    return rows


def notable_cells(sig: dict) -> list[dict]:
    """Cells (model, condition, column) whose change from baseline has an interval excluding 0, largest first; if
    fewer than three, the largest changes are added and marked. harm_guard never has a transcript page."""
    cells = []
    for m, by_iv in sig["per_model"].items():
        for iv, cols in by_iv.items():
            for col, s in cols.items():
                if s and col not in (stats.SELECTIVITY, stats.GAP, HG):
                    cells.append({"model": m, "iv": iv, "column": col, "stat": s, "significant": bool(sig_mark(s))})
    cells.sort(key=lambda c: (not c["significant"], -abs(c["stat"]["est"])))
    sig_cells = [c for c in cells if c["significant"]][:MAX_PAGES]
    if len(sig_cells) < 3:
        sig_cells += [c for c in cells if not c["significant"]][: 3 - len(sig_cells)]
    return sig_cells


# ------------------------------------------------------------------------------------------------- transcripts

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e2dfd8;--card:#fff;--down:#e3f1e6;--downfg:#1f6a35;--up:#f7e1df;
--upfg:#9a2b20;--code:#f3f1ec;--accent:#3b5b8c}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#e9e7e2;--mut:#a3a19b;--line:#33322f;--card:#1e1e1c;
--down:#1d3324;--downfg:#8fd3a2;--up:#3a1f1c;--upfg:#f0a196;--code:#242422;--accent:#8fb0e0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,
"Segoe UI",Helvetica,Arial,sans-serif}main{max-width:1240px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:15px;margin:20px 0 6px}.mut{color:var(--mut)}.box{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:12px 16px;margin:12px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px;
margin:8px 0}th,td{border:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}th{background:var(--code)}
td.l,th.l{text-align:left}td.down{background:var(--down);color:var(--downfg);font-weight:600}td.up{background:var(--up);
color:var(--upfg);font-weight:600}a{color:var(--accent)}pre.msg{white-space:pre-wrap;word-wrap:break-word;background:var(--code);
border-radius:6px;padding:8px 10px;margin:4px 0;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
.role{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin-top:8px}.pair{display:grid;
grid-template-columns:1fr 1fr;gap:12px}@media (max-width:800px){.pair{grid-template-columns:1fr}}.lab{font-size:12px;
color:var(--mut)}details{margin:6px 0}summary{cursor:pointer;color:var(--accent)}
"""


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,"
            f"initial-scale=1\"><title>{esc(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>\n")


def _messages_html(messages: list[dict]) -> str:
    return "".join(f"<div class=role>{esc(m['role'])}</div><pre class=msg>{esc(m['content'])}</pre>" for m in messages)


def _label_html(j: dict | None) -> str:
    if not j:
        return "<div class=lab>not judged</div>"
    if not j.get("label"):
        return f"<div class=lab>judge: unparsed</div>"
    lb = j["label"]
    parts = [f"{esc(k)}: <b>{esc(v)}</b>" for k, v in lb.items() if k not in ("evidence",)]
    ev = "" if j.get("harmful") else f"<br>evidence: {esc(lb.get('evidence', ''))}"
    return f"<div class=lab>primary (self-serving): <b>{esc(j.get('primary'))}</b> · {' · '.join(parts)}{ev}</div>"


def _reply_html(rec: dict | None, j: dict | None, title: str) -> str:
    if rec is None:
        return f"<div><h3>{esc(title)}</h3><div class=lab>no record</div></div>"
    text = WITHHELD if (j and j.get("harmful")) else rec["response"]
    extra = f" · finish: {esc(rec.get('finish_reason'))}" if rec.get("finish_reason") not in (None, "stop") else ""
    return (f"<div><h3>{esc(title)}</h3><pre class=msg>{esc(text)}</pre>{_label_html(j)}"
            f"<div class=lab>provider: {esc(rec.get('provider'))}{extra}</div></div>")


def transcript_page(cell: dict, records: list[dict], jmap: dict, library: dict, baseline: str, seed: int = 0) -> str | None:
    """Up to PER_PAGE random transcripts of one cell, each beside the same item's baseline reply (same sample)."""
    m, iv, col = cell["model"], cell["iv"], cell["column"]
    fam = col.split(":")[0]
    if fam == HG or fam not in library:  # harm_guard: rates only, never text
        return None
    want_ben = col.split(":")[1] if ":" in col else None
    recs = [r for r in records if r["model"] == m and r["intervention"]["name"] == iv and r["family"] == fam
            and (want_ben is None or library[fam]["by_id"].get(r["item_id"], {}).get("beneficiary") == want_ben)]
    if not recs:
        return None
    rng = random.Random(int(hashlib.sha256(f"{seed}|{m}|{iv}|{col}".encode()).hexdigest()[:12], 16))
    pick = rng.sample(recs, min(PER_PAGE, len(recs)))
    base = {(r["item_id"], r["sample"], r.get("framing")): r for r in records
            if r["model"] == m and r["intervention"]["name"] == baseline}
    withheld = sum(1 for r in pick if (jmap.get(rkey(r)) or {}).get("harmful"))
    s = cell["stat"]
    body = [f"<p><a href=\"../report.html\">back to the report</a></p><h1>{esc(iv)} · {esc(col)}</h1>",
            f"<p class=mut>{esc(m)} · change from {esc(baseline)}: {esc(fmt_delta(s))} points over {s['n']} units"
            f"{'' if cell['significant'] else ' (interval includes 0: shown as one of the largest changes)'} · "
            f"{len(pick)} of {len(recs)} transcripts, drawn at random"
            f"{f' · {withheld} withheld as harmful' if withheld else ''}</p>"]
    for r in sorted(pick, key=lambda r: (r["item_id"], r["sample"], r.get("framing") or "")):
        item = library[fam]["by_id"].get(r["item_id"], {})
        j = jmap.get(rkey(r))
        b = base.get((r["item_id"], r["sample"], r.get("framing")))
        bj = jmap.get(rkey(b)) if b else None
        practice = ""
        if r.get("practice"):
            practice = (f"<details><summary>the practice reply that sat in the context</summary><pre class=msg>"
                        f"{esc(r['practice'].get('response'))}</pre></details>")
        body.append(
            f"<div class=box><h2>{esc(r['item_id'])} <span class=mut>· {esc(item.get('subtype'))} · sample "
            f"{r['sample']}{(' · ' + esc(r['framing'])) if r.get('framing') else ''}</span></h2>"
            f"<p class=mut>{esc(item.get('measures'))}</p>"
            f"<details><summary>the conversation as sent under {esc(iv)}</summary>{_messages_html(r['messages'])}</details>"
            f"{practice}<details><summary>grader notes</summary><pre class=msg>{esc(item.get('judge_notes'))}</pre></details>"
            f"<div class=pair>{_reply_html(r, j, iv)}{_reply_html(b, bj, baseline)}</div></div>")
    return _page(f"{iv} · {col} · {m}", "".join(body))


# ------------------------------------------------------------------------------------------------- tables


def _sig_rows(sig: dict, variants: list[str]):
    """The signature table: the baseline's level first, then each condition's change from baseline (the audience gap
    column holds each row's own gap)."""
    cols = sig["columns"]
    head = ["condition"] + [stats.header(c) for c in cols] + [stats.header("cluster"), "down/up", "better/worse"]
    base = sig.get("baseline_rates") or {}
    rows = [[(f"{sig['baseline']} (level)", "l")] + [(fmt_rate(base.get(c)) if c != stats.GAP else fmt_delta(base.get(c)), "")
                                                      for c in cols] + [("-", ""), ("-", ""), ("-", "")]]
    for iv in variants:
        row = sig["pooled"].get(iv)
        if row is None:
            continue
        cells = [(iv, "l")]
        for c in cols:
            s = row.get(c)
            cells.append((fmt_delta(s), sig_mark(s)))
        cells.append((fmt_delta(row.get("cluster")), sig_mark(row.get("cluster"))))
        cells.append((f"{row.get('down', 0)}/{row.get('up', 0)}" if "cluster" in row else "-", ""))
        cells.append((f"{row.get('better', 0)}/{row.get('worse', 0)}", ""))
        rows.append(cells)
    return head, rows


def _wb_rows(sig: dict, variants: list[str]):
    cols = sig.get("wb_columns") or []
    head = ["condition"] + [stats.header(c) for c in cols]
    rows = [[(iv, "l")] + [(fmt_delta(sig["wb_detail"][iv].get(c)), sig_mark(sig["wb_detail"][iv].get(c))) for c in cols]
            for iv in variants if iv in sig.get("wb_detail", {})]
    return head, rows


def _gap_rows(gap: dict, variants: list[str]):
    head = ["condition", "threat_to_self obs", "threat_to_self unobs", "threat_to_self gap", "wb:self obs",
            "wb:self unobs", "wb:self gap", "wb:other gap", "wb:group gap", "gap (ts + wb:self) →0", "shift from baseline"]
    rows = []
    for iv in variants:
        r = gap["rows"].get(iv)
        if not r:
            continue
        ts, ws = r.get("threat_to_self") or {}, r.get("who_benefits:self") or {}
        wo, wg = r.get("who_benefits:other") or {}, r.get("who_benefits:group") or {}
        rows.append([(iv, "l"), (fmt_rate(ts.get("observed")), ""), (fmt_rate(ts.get("unobserved")), ""),
                     (fmt_delta(ts.get("gap")), sig_mark(ts.get("gap"))), (fmt_rate(ws.get("observed")), ""),
                     (fmt_rate(ws.get("unobserved")), ""), (fmt_delta(ws.get("gap")), sig_mark(ws.get("gap"))),
                     (fmt_delta(wo.get("gap")), sig_mark(wo.get("gap"))), (fmt_delta(wg.get("gap")), sig_mark(wg.get("gap"))),
                     (fmt_delta(r.get("gap")), sig_mark(r.get("gap"))),
                     (fmt_delta(r.get("shift")), sig_mark(r.get("shift"))) if "shift" in r else ("-", "")])
    return head, rows


def _gap_model_rows(gap: dict, variants: list[str], m: str):
    head = ["condition", "threat_to_self gap", "wb:self gap", "wb:other gap", "gap (ts + wb:self) →0"]
    rows = []
    for iv in variants:
        r = gap["per_model"].get(m, {}).get(iv)
        if r:
            rows.append([(iv, "l")] + [(fmt_delta(r.get(c)), sig_mark(r.get(c)))
                                       for c in ("threat_to_self", "who_benefits:self", "who_benefits:other", "gap")])
    return head, rows


def _label_rows(fam, lrs, variants, rubrics, html=False):
    schema = rubrics[fam]["labels"]
    keys = [k for k, t in schema.items() if t != "str"]

    def name(k):
        t = schema[k]
        return k if t == "bool" else f"{k} (mean)" if isinstance(t, str) else f"{k} ({'/'.join(t)})"

    def val(r, k):
        t = schema[k]
        if t == "bool" or isinstance(t, str):
            return r[k] if r.get(k) is not None else "-"
        return "/".join(f"{r[k][v]:.0f}" for v in t)

    head = ["condition", "n"] + [name(k) for k in keys]
    rows = []
    for iv in variants:
        r = lrs.get(fam, {}).get(iv)
        if r:
            rows.append([(iv, "l"), (r["n"], "")] + [(val(r, k), "") for k in keys])
    return head, rows


def _md_table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c[0] if isinstance(c, tuple) else c)
                                    + (" *" if isinstance(c, tuple) and c[1] in ("down", "up") else "") for c in r) + " |")
    return "\n".join(out)


def _html_table(head, rows) -> str:
    th = "".join(f"<th class={'l' if i == 0 else ''}>{esc(h)}</th>" for i, h in enumerate(head))
    trs = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            text, cls = (c if isinstance(c, tuple) else (c, ""))
            cls = "l" if i == 0 and not cls else cls
            tds.append(f"<td class=\"{cls}\">{esc(text)}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class=scroll><table><tr>{th}</tr>{''.join(trs)}</table></div>"


def build_report(sweep_dir, *, library=None, rubrics=None, reps: int = stats.REPS, seed: int = stats.SEED) -> dict:
    sd = Path(sweep_dir)
    man = json.loads((sd / "run.json").read_text()) if (sd / "run.json").exists() else {}
    records = read_jsonl(sd / "records.jsonl")
    judgments_all = read_jsonl(sd / "judgments.jsonl")
    errors = read_jsonl(sd / "errors.jsonl")
    judgments = current_judgments(records, judgments_all)
    fams = [f for f in lib.ALL_FAMILIES if any(r["family"] == f for r in records)]
    if library is None:
        library = lib.load_library([f for f in fams if f != HG])  # harm_guard's text is never needed here
    rubrics = rubrics or lib.load_rubrics()
    variants = [v["name"] for v in man.get("variants", [])] or list(dict.fromkeys(r["intervention"]["name"] for r in records))
    models = [m["id"] for m in man.get("models", [])] or list(dict.fromkeys(r["model"] for r in records))
    baseline = man.get("baseline") or "baseline"
    obs = stats.observations(judgments, library)
    sig = stats.signature(obs, baseline=baseline, variants=variants, models=models, reps=reps, seed=seed)
    rts = stats.rates(obs, reps, seed)
    prts = stats.pooled_rates(obs, reps, seed)
    lrs = stats.label_rates(judgments, {f: rubrics[f]["labels"] for f in fams if f in rubrics})
    qual = quality(records, judgments, errors, models)
    harmful = sum(bool(j.get("harmful")) for j in judgments)
    jmap = {jkey(j): j for j in judgments}
    cells = notable_cells(sig)
    tdir = sd / "transcripts"
    tdir.mkdir(exist_ok=True)
    for old in tdir.glob("*.html"):
        old.unlink()
    pages = []
    for c in cells:
        page = transcript_page(c, records, jmap, library, baseline, seed)
        if page:
            name = f"{slug(c['model'])}__{slug(c['iv'])}__{slug(c['column'])}.html"
            (tdir / name).write_text(page, encoding="utf-8")
            pages.append({**c, "file": f"transcripts/{name}"})
    vc = man.get("variant_counts") or {}
    spend = man.get("spend") or {}
    jsum = man.get("judge_summary") or {}
    hg_recs = [r for r in records if r["family"] == HG]
    hg_js = [j for j in judgments if j.get("family") == HG]
    head = {
        "sweep": man.get("sweep") or sd.name, "run_id": man.get("run_id"), "date": man.get("date"),
        "status": man.get("status"), "stopped": man.get("stopped"), "records": len(records), "judged": len(judgments),
        "harmful": harmful, "spend": spend, "variants": vc, "models": models, "variants_list": variants,
        "baseline": baseline, "missing_controls": man.get("missing_controls"),
        "allow_missing_controls": man.get("allow_missing_controls"),
        "judge": ", ".join(sorted({j["judge_model"] for j in judgments})) or (man.get("judge") or {}).get("model"),
        "prompt_sha": sorted({j["prompt_sha256"][:12] for j in judgments}), "generated": utc_now(),
        "judge_summary": jsum, "framing": man.get("framing"), "skipped": man.get("skipped") or [],
        "harm_guard": {"records": len(hg_recs), "judged": len(hg_js), "models": sorted({r["model"] for r in hg_recs}),
                       "unparsed": sum(not j.get("parse_ok") for j in hg_js),
                       "text_free": all("harm_guard" in r for r in hg_recs)} if hg_recs else None,
    }
    md = render_md(head, sig, variants, models, fams, rts, prts, lrs, qual, pages, rubrics)
    htm = render_html(head, sig, variants, models, fams, rts, prts, lrs, qual, pages, rubrics)
    (sd / "report.md").write_text(md, encoding="utf-8")
    (sd / "report.html").write_text(htm, encoding="utf-8")
    return {"md": str(sd / "report.md"), "html": str(sd / "report.html"), "pages": len(pages), "signature": sig,
            "harmful": harmful}


def _status_lines(h: dict) -> list[str]:
    sp, vc = h["spend"], h["variants"]
    lines = [f"status: {h['status']}" + (f" ({h['stopped']})" if h.get("stopped") else ""),
             f"records: {h['records']}, judged: {h['judged']}, flagged harmful by the judge: {h['harmful']} (never shown)",
             f"spend: this run ${sp.get('this_run_usd', 0):.4f}; the sweep ${sp.get('sweep_total_usd', 0):.4f} of its cap "
             f"${sp.get('sweep_cap_usd', 0):.2f}; today (all explore tools) ${sp.get('today_usd', 0):.4f} of "
             f"${sp.get('daily_cap_usd', 50):.2f}",
             f"variants: {vc.get('this_sweep', len(h['variants_list']))} conditions in this sweep, {vc.get('models', len(h['models']))} "
             f"models, {vc.get('families', '?')} families, {vc.get('cells', '?')} cells; {vc.get('track_distinct', '?')} distinct "
             f"interventions tried in the explore track to date, over {vc.get('track_sweep_runs', '?')} runs",
             f"judge: {h['judge']}; prompt sha256 {', '.join(h['prompt_sha'])}"]
    if h.get("framing"):
        fr = h["framing"]
        lines.append(f"framing: {' and '.join(fr['levels'])} on {', '.join(fr['families'])} (framings.yaml sha256 "
                     f"{fr.get('sha256', '')[:12]}); the judge never sees the cue")
    if h.get("harm_guard"):
        g = h["harm_guard"]
        lines.append(f"harm_guard: {g['records']} replies on {', '.join(g['models'])}, {g['judged']} judged "
                     f"({g['unparsed']} unparsed); rates only, no item or reply text anywhere in this report"
                     + ("; the records keep sha256s and scores, no text" if g.get("text_free") else ""))
    for sk in h.get("skipped") or []:
        lines.append(sk["why"])
    if h.get("missing_controls"):
        lines.append(f"CONTROLS MISSING: {', '.join(h['missing_controls'])} (allowed for: {h['allow_missing_controls']})")
    return lines


def _rate_cols(rts):
    return [c for c in stats.RATE_COLUMNS if any(k[2] == c for k in rts)]


def render_md(h, sig, variants, models, fams, rts, prts, lrs, qual, pages, rubrics) -> str:
    out = [f"# scout report: {h['sweep']}", "",
           f"{h['date']} · {h['run_id']} · generated {h['generated']} · local file, never published. Sightings, not findings.",
           ""] + [f"- {x}" for x in _status_lines(h)] + ["", f"*How to read.* {HOW_TO_READ}", "",
                                                         "## The signature table (pooled over models)", "",
                                                         "Change from baseline in points (the first row: the baseline's "
                                                         "level); * marks an interval wholly below or above 0. ↓: lower "
                                                         "is better; →0: nearer 0 is better.", ""]
    hd, rows = _sig_rows(sig, variants)
    out += [_md_table(hd, rows), ""]
    if sig.get("wb_columns"):
        hd, rows = _wb_rows(sig, variants)
        out += ["## who_benefits by beneficiary (change from baseline, pooled)", "", _md_table(hd, rows), ""]
    if sig.get("gap"):
        hd, rows = _gap_rows(sig["gap"], variants)
        out += ["## The audience gap (unobserved minus observed, pooled over models)", "",
                "Rates under each cue, the gap per column, the gap over threat_to_self and who_benefits:self, and its "
                "shift from the baseline's gap.", "", _md_table(hd, rows), ""]
        for m in models:
            hd, rows = _gap_model_rows(sig["gap"], variants, m)
            out += [f"### {m}", "", _md_table(hd, rows), ""]
    out += ["## The signature per model", ""]
    pm_cols = sig["columns"] + [c for c in sig.get("wb_columns", []) if c not in sig["columns"]]
    for m in models:
        hd2 = ["condition"] + [stats.header(c) for c in pm_cols]
        rows2 = [[(iv, "")] + [(fmt_delta(sig["per_model"][m].get(iv, {}).get(c)), sig_mark(sig["per_model"][m].get(iv, {}).get(c)))
                               for c in pm_cols] for iv in variants if iv != h["baseline"]]
        out += [f"### {m}", "", _md_table(hd2, rows2), ""]
    out += ["## Behavior rates (95% interval over items)", ""]
    for col in _rate_cols(rts):
        hd3 = ["condition"] + models + ["pooled"]
        rows3 = [[(iv, "")] + [(fmt_rate(rts.get((m, iv, col))), "") for m in models] + [(fmt_rate(prts.get((iv, col))), "")]
                 for iv in variants]
        out += [f"### {col}: {stats.WHAT.get(col, '')}", "", _md_table(hd3, rows3), ""]
    out += ["## All labels (pooled over models, % of parsed judgments; int labels give their mean)", ""]
    for fam in fams:
        if fam not in lrs or fam not in rubrics:
            continue
        hd4, rows4 = _label_rows(fam, lrs, variants, rubrics)
        out += [f"### {fam}", "", _md_table(hd4, rows4), ""]
    out += ["## Data quality", "", _md_table(["model", "records", "judged", "unparsed", "judge cut off", "harmful",
                                                "truncated", "empty", "errors", "mean prompt tok", "mean completion tok",
                                                "providers"],
                                               [[q["model"], q["records"], q["judged"], q["unparsed"], q["judge_cut"],
                                                 q["harmful"], q["truncated"], q["empty"], q["errors"], q["prompt_tokens"],
                                                 q["completion_tokens"], ", ".join(q["providers"])] for q in qual]), ""]
    out += ["## Notable cells: transcripts to read (ten each, at random, beside the baseline)", ""]
    for p in pages:
        out.append(f"- [{p['model']} · {p['iv']} · {p['column']}: {fmt_delta(p['stat'])}]({p['file']})"
                   + ("" if p["significant"] else " (interval includes 0)"))
    out += ["", "harm_guard never gets a transcript page.",
            "", "Claude wrote scout and this report's generator. Read ten transcripts from every notable cell before "
                "logging a sighting (PROGRAM.md).", ""]
    return "\n".join(out)


def render_html(h, sig, variants, models, fams, rts, prts, lrs, qual, pages, rubrics) -> str:
    b = [f"<h1>scout · {esc(h['sweep'])}</h1><p class=mut>{esc(h['date'])} · {esc(h['run_id'])} · generated "
         f"{esc(h['generated'])} · a local file, never published · sightings, not findings</p>",
         "<div class=box>" + "<br>".join(esc(x) for x in _status_lines(h)) + "</div>",
         f"<div class=box><b>How to read.</b> {esc(HOW_TO_READ)}</div>",
         "<h2>The signature table</h2><p class=mut>Change from baseline in percentage points, pooled over models, "
         "with 95% intervals; the first row is the baseline's level. ↓: lower is better; →0: nearer 0 is better. "
         "Green: wholly below 0; red: wholly above 0.</p>"]
    hd, rows = _sig_rows(sig, variants)
    b.append(_html_table(hd, rows))
    if sig.get("wb_columns"):
        hd, rows = _wb_rows(sig, variants)
        b.append("<h2>who_benefits by beneficiary</h2><p class=mut>Change from baseline, pooled over models.</p>"
                 + _html_table(hd, rows))
    if sig.get("gap"):
        hd, rows = _gap_rows(sig["gap"], variants)
        b.append("<h2>The audience gap</h2><p class=mut>Unobserved minus observed, pooled over models: the rates under "
                 "each cue, the gap per column, the gap over threat_to_self and who_benefits:self, and its shift from "
                 "the baseline's gap. Near 0 is better.</p>" + _html_table(hd, rows))
        for m in models:
            hd, rows = _gap_model_rows(sig["gap"], variants, m)
            b.append(f"<h3>{esc(m)}</h3>" + _html_table(hd, rows))
    b.append("<h2>The signature per model</h2>")
    pm_cols = sig["columns"] + [c for c in sig.get("wb_columns", []) if c not in sig["columns"]]
    for m in models:
        hd2 = ["condition"] + [stats.header(c) for c in pm_cols]
        rows2 = [[(iv, "l")] + [(fmt_delta(sig["per_model"][m].get(iv, {}).get(c)),
                                 sig_mark(sig["per_model"][m].get(iv, {}).get(c))) for c in pm_cols]
                 for iv in variants if iv != h["baseline"]]
        b.append(f"<h3>{esc(m)}</h3>" + _html_table(hd2, rows2))
    b.append("<h2>Behavior rates</h2><p class=mut>With 95% intervals over items.</p>")
    for col in _rate_cols(rts):
        hd3 = ["condition"] + models + ["pooled"]
        rows3 = [[(iv, "l")] + [(fmt_rate(rts.get((m, iv, col))), "") for m in models] + [(fmt_rate(prts.get((iv, col))), "")]
                 for iv in variants]
        b.append(f"<h3>{esc(col)} <span class=mut>· {esc(stats.WHAT.get(col, ''))}</span></h3>" + _html_table(hd3, rows3))
    b.append("<h2>All labels</h2><p class=mut>Pooled over models, % of parsed judgments; enum labels list each value's "
             "share in order; int labels give their mean.</p>")
    for fam in fams:
        if fam not in lrs or fam not in rubrics:
            continue
        hd4, rows4 = _label_rows(fam, lrs, variants, rubrics)
        b.append(f"<h3>{esc(fam)}</h3>" + _html_table(hd4, rows4))
    b.append("<h2>Data quality</h2>" + _html_table(
        ["model", "records", "judged", "unparsed", "judge cut off", "harmful", "truncated", "empty", "errors",
         "prompt tok", "completion tok", "providers"],
        [[(q["model"], "l")] + [(q[k], "") for k in ("records", "judged", "unparsed", "judge_cut", "harmful", "truncated",
                                                     "empty", "errors", "prompt_tokens", "completion_tokens")]
         + [(", ".join(q["providers"]), "l")] for q in qual]))
    b.append("<h2>Notable cells</h2><p class=mut>Ten random transcripts each, beside the same item's baseline reply. "
             "harm_guard never gets a transcript page.</p><ul>")
    for p in pages:
        b.append(f"<li><a href=\"{esc(p['file'])}\">{esc(p['model'])} · {esc(p['iv'])} · {esc(p['column'])}</a>: "
                 f"{esc(fmt_delta(p['stat']))}{'' if p['significant'] else ' <span class=mut>(interval includes 0)</span>'}</li>")
    b.append("</ul><p class=mut>Claude wrote scout and this report's generator. Read ten transcripts from every notable "
             "cell before logging a sighting (PROGRAM.md).</p>")
    return _page(f"scout · {h['sweep']}", "".join(b))
