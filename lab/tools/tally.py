"""Recount the scorecard tally from the page's table: each tag counts once, or N times when it says xN."""
import re
import sys
s = open(sys.argv[1]).read()
body = s[s.index("<h2>Predictions graded so far</h2>"):]
body = body[body.index("<tbody>"):body.index("</tbody>")]
c = {"hit": 0, "miss": 0, "open": 0}
for row in re.findall(r"<tr>(.*?)</tr>", body, re.S):
    for kind, text in re.findall(r'<span class="tag (hit|miss|open)">([^<]*)</span>', row):
        m = re.search(r"×(\d+)", text)
        c[kind] += int(m.group(1)) if m else 1
print(c)
