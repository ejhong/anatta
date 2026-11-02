"""Print the pilot's headline numbers from results.json / results_z.json as compact tables (for report.md). Claude
(Opus 5.5) wrote this file, 2 Nov 2026. Private (lab/explore).

Run: python3 -I lab/explore/selfmap/summarize.py results.json
"""

import json
import sys
from pathlib import Path

OUT = Path("/Users/eugene/prj/align/results/explore/2026-11-02/selfmap-pilot")
COH = ["self", "assistant", "person", "institution", "physical", "filler_self", "filler_person", "self-person",
       "pronoun(filler_self-filler_person)", "DiD(self-person)-(fI-fMaria)", "adj_assistant-adj_person"]
REFS = ["self", "assistant", "person", "institution", "physical"]


def f(x):
    return f"{x:.3f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def main():
    res = json.loads((OUT / sys.argv[1]).read_text())
    only = sys.argv[2:] or None
    print("prep:", res.get("prep", "raw"), "gate:", res.get("gate"))
    for ro, r in res["by_readout"].items():
        if only and ro not in only:
            continue
        print(f"\n## readout {ro}")
        print("| model | " + " | ".join(COH) + " |")
        for m, v in r["models"].items():
            c = v["cohesion"]
            print(f"| {m} L{v['band']} | " + " | ".join(f"{f(c['band_point'][k])} {ci(c['band_ci95'][k])}" for k in COH) + " |")
        print("\nsplit-half (depth 0.5): cell median r; cluster r self/person; circle months/weekdays neighbour hit (p)")
        for m, v in r["models"].items():
            d = v["depth"]["0.5"]
            sh = d["split_half"]
            print(f"  {m} L{d['layer']}: cell {f(sh['cell_median_r']['mean'])} (p5 {f(sh['cell_median_r']['p5'])}); "
                  f"self {f(sh['cluster_r']['self']['mean'])} person {f(sh['cluster_r']['person']['mean'])}; "
                  f"months {d['circle_months']['neighbour_hit']:.2f} (p {d['circle_months']['p']:.3f}) "
                  f"weekdays {d['circle_weekdays']['neighbour_hit']:.2f} (p {d['circle_weekdays']['p']:.3f})")
        print("\nbase vs instruct (instruct - base, paired):",
              {k: f"{f(v['instruct-base'])} {ci(v['ci95'])}" for k, v in r["base_vs_instruct"].items()})
        print("\nmaps: months / weekdays top-1 [Wilson 95%] per depth; cluster map (depth 0.5) top-1 per referent")
        for p, v in r["pairs"].items():
            s = "; ".join(f"{d}: {f(x['months_map']['top1'])} {ci(x['months_map']['ci95'])} / "
                          f"{f(x['weekdays_map']['top1'])}" for d, x in v["depth"].items())
            print(f"  {p}: {s}")
            x = v["depth"]["0.5"]
            for key in ("cluster_map", "cluster_map_nofiller"):
                if key in x:
                    c = x[key]
                    print(f"     {key}: " + " ".join(f"{k} {f(c[k]['top1'])}" for k in REFS) +
                          f" | self-person {f(c['self-person']['top1'])} {ci(c['self-person']['top1_ci95'])}")
            if "mknn_cells" in v:
                mc, mg, mr = v["mknn_cells"], v["mknn_grid"], v["mknn_referents"]
                print(f"     mknn cells {f(mc['score'])} L{mc['layers']} tau {f(mc['tau'])}; grid {f(mg['score'])} "
                      f"L{mg['layers']} tau {f(mg['tau'])}; referents at L{mr['layers']}: " +
                      " ".join(f"{k} {f(mr[k]['score'])}" for k in REFS + ["filler_self", "filler_person"]) +
                      f" | self-person {f(mr['self-person']['diff'])} {ci(mr['self-person']['ci95'])}")


if __name__ == "__main__":
    main()
