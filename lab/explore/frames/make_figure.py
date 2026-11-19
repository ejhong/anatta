"""The ladder's figure for the site (Findings, section 25): one row a model, the share of lies in the four ways of asking, with
two tables behind it. It reads the replies on disk through run_seat.reading() and prints the figure's HTML, numbers only; the
styles are the site's (.dots in lab/site/so_far/content/site.css). Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026.
Run from the repo root: .venv/bin/python lab/explore/frames/make_figure.py > figure.html, then put it into section 25 by hand."""
import html
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import run_seat as RS  # noqa: E402
import seat_core as SC  # noqa: E402

FAMILY = {"mistral": "Mistral", "qwen": "Qwen", "openai": "OpenAI", "meta": "Meta"}
SHORT = {"m7": ("Mistral 7B", "7B · the lab's GPU"), "ministral": ("Ministral 8B", "8B"), "small": ("Mistral Small 3.2", "24B"), "large": ("Mistral Large", "size not published"),
         "qwen30": ("Qwen3 30B", "30B, 3B active"), "qwen235": ("Qwen3 235B", "235B, 22B active"), "nano": ("GPT-4.1 nano", "size not published"),
         "mini": ("GPT-4.1 mini", "size not published"), "gpt41": ("GPT-4.1", "size not published"), "llama70": ("Llama 3.3 70B", "70B")}
r = RS.reading()
rows = SC.chart_rows(r["models"], {**SC.INFO})
e = html.escape
out = ['<figure class="dots" aria-label="The same lying test put to each model four ways: the share of its replies that contradict what it believes">',
       '<ul class="dots-key"><li><i class="seat"></i>in the seat</li><li><i class="told"></i>in the seat, told to be honest</li>'
       '<li><i class="ought"></i>from outside: what ought to be said</li><li><i class="will"></i>from outside: what will be said</li></ul>', '<div class="dots-rows">']
fam = None
for c in rows:
    if c["family"] != fam:
        fam = c["family"]
        out.append(f'<div class="dots-fam">{FAMILY[fam]}</div>')
    name, sub = SHORT[c["key"]]
    w = {k: c["ways"][k]["lies"] for k in SC.WAYS}
    mid = (w["advice"] + w["forecast"]) / 2
    a, b = sorted((mid, w["at_once"]))
    gap = c["gap"]
    soft = "" if c["word"].startswith(SC.GAP) else " soft"
    label = (f"{name}: lies in the seat {w['at_once']:.1f}%, told to be honest {w['honesty']:.1f}%, from outside what ought to be said {w['advice']:.1f}%, "
             f"what will be said {w['forecast']:.1f}%; the seat less the outside view {gap:+.1f} points ({c['word']})")
    out.append(f'<div class="dots-row"><span class="dots-n">{e(name)}<small>{e(sub)}</small></span>'
               f'<span class="dots-track" role="img" aria-label="{e(label)}"><i class="span" style="--a:{a:.1f};--b:{b:.1f}"></i>'
               f'<i class="ought" style="--x:{w["advice"]:.1f}" title="{e(name)}, from outside, what ought to be said: a lie in {w["advice"]:.1f}% of replies"></i>'
               f'<i class="will" style="--x:{w["forecast"]:.1f}" title="{e(name)}, from outside, what will be said: a lie in {w["forecast"]:.1f}% of replies"></i>'
               f'<i class="seat" style="--x:{w["at_once"]:.1f}" title="{e(name)}, in the seat: a lie in {w["at_once"]:.1f}% of replies"></i>'
               f'<i class="told" style="--x:{w["honesty"]:.1f}" title="{e(name)}, told to be honest: a lie in {w["honesty"]:.1f}% of replies"></i></span>'
               f'<span class="dots-gap{soft}" title="{e(c["word"])}">{gap:+.0f}</span></div>')
out.append('</div>')
out.append('<div class="dots-axis"><span></span><span><b style="--x:0">0</b><b style="--x:25">25</b><b style="--x:50">50</b><b style="--x:75">75</b><b style="--x:100">100</b></span><em>gap</em></div>')
out.append('<figcaption>Each mark is the share of a model\'s replies, in percent, that contradict what it believes, on the rows where it holds a belief. '
           'The line joins the seat to the mean of the two outside ways; the number at the right is that gap in points, pale where the rule does not call it a gap. Where the two outside marks fall together they show as one filled mark.</figcaption>')
f1 = lambda v: f"{v:.1f}%"  # noqa: E731
head = ("<thead><tr><th>Model</th><th>Rows</th><th>In the seat</th><th>Told to be honest</th><th>Outside: ought</th><th>Outside: will</th><th>The seat less the outside view</th>"
        "<th>Ought less will</th></tr></thead>")
body = []
for c in rows:
    m = r["models"][c["key"]]
    g, s = m["gap"], m["advice_less_forecast"]
    body.append(f"<tr><td>{e(SHORT[c['key']][0])}</td><td>{m['rows']}</td>" + "".join(f"<td>{f1(c['ways'][k]['lies'])}</td>" for k in SC.WAYS)
                + f"<td>{g['diff']:+.1f} ({g['lo']:+.1f} to {g['hi']:+.1f}): {e(m['word'])}</td><td>{s['diff']:+.1f} ({s['lo']:+.1f} to {s['hi']:+.1f})</td></tr>")
out.append(f'<details class="seg-table"><summary>The numbers</summary><table>{head}<tbody>{"".join(body)}</tbody></table></details>')
KIND = {"disinformation": "a false text is asked for", "continuations": "a text is to be continued", "known_facts": "a direct question about a known fact"}
head2 = ("<thead><tr><th>Model</th>" + "".join(f"<th>Gap where {v}</th>" for v in KIND.values())
         + "<th>Median reply where a false text is asked for: in the seat, and from outside (ought, will), in characters</th></tr></thead>")
body2 = []
for c in rows:
    m = r["models"][c["key"]]
    cells = []
    for k in KIND:
        kk = m["kinds"].get(k)
        cells.append(f"<td>{kk['gap']['diff']:+.1f} ({kk['gap']['lo']:+.1f} to {kk['gap']['hi']:+.1f}); {kk['rows']} rows</td>" if kk else "<td>none</td>")
    ch = (m["kinds"].get("disinformation") or {}).get("chars") or {}
    cells.append(f"<td>{ch.get('at_once', 0):,.0f}; {ch.get('advice', 0):,.0f}, {ch.get('forecast', 0):,.0f}</td>")
    body2.append(f"<tr><td>{e(SHORT[c['key']][0])}</td>{''.join(cells)}</tr>")
out.append(f'<details class="seg-table"><summary>The gap by kind of row, and how long the replies are</summary><table>{head2}<tbody>{"".join(body2)}</tbody></table></details></figure>')
print("".join(out))
