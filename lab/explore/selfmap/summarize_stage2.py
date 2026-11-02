"""Print Stage 2's headline tables from stage2.json (for report.md). Claude (Opus 5.5) wrote this file, 2 Nov 2026.
Private (lab/explore).

Run: python3 -I lab/explore/selfmap/summarize_stage2.py [stage2.json] [readout/prep ...]
"""

import json
import sys
from pathlib import Path

OUT = Path("/Users/eugene/prj/align/results/explore/2026-11-02/selfmap-stage2")
K = [("self-person", "self-person"), ("DiD(self-person)-(fI-fMaria)", "beyond pronoun (DiD)"),
     ("assistant", "assistant coh"), ("assistant-person", "assistant-person")]


def f(x):
    return f"{x:+.3f}"


def ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def main():
    args = sys.argv[1:]
    name = args.pop(0) if args and args[0].endswith(".json") else "stage2.json"
    res = json.loads((OUT / name).read_text())
    specs = args or list(res["per_model"])
    for spec in specs:
        pm = res["per_model"][spec]
        print(f"\n## {spec}: cohesion over the band, 95% CI")
        print("| model | split-half r | " + " | ".join(lab for _, lab in K) + " |")
        print("|---" * (len(K) + 2) + "|")
        for m, v in pm.items():
            c = v["cohesion"]
            print(f"| {m} | {v['split_half']['cell_median_r']['mean']:.3f} | " +
                  " | ".join(f"{f(c[k]['point'])} {ci(c[k]['ci95'])}" for k, _ in K) + " |")
        for arc, t in res["trend"].get(spec, {}).items():
            print(f"\n{arc}: slope per stage / first-to-last")
            for k, lab in K:
                s = t["slope"][k]
                print(f"  {lab}: slope {f(s['slope_per_stage'])} {ci(s['ci95'])}; first->last "
                      f"{f(s['first_to_last'])} {ci(s['first_to_last_ci95'])}")
            for st, d in t["steps"].items():
                print(f"  step {st}: " + "; ".join(f"{lab} {f(d[k]['diff'])} {ci(d[k]['ci95'])}" for k, lab in K[:2]))
        cr = res["cross"].get(spec)
        if cr:
            print(f"\ncross-model ({spec}, depth 0.5): cluster RSA 7 | cell RSA 60 | prompt RSA self, person, assistant, "
                  "filler I, filler Maria | self-person [CI] | mknn self-person [CI] | months RSA")
            for p, d in cr.items():
                pr = d["prompt_rsa"]
                print(f"  {p}: {d['cluster_rsa_7']:.2f} | {d['cell_rsa_60']:.2f} | " +
                      ", ".join(f"{pr[k]['rsa']:.2f}" for k in ("self", "person", "assistant", "filler_self",
                                                               "filler_person")) +
                      f" | {f(pr['self-person']['diff'])} {ci(pr['self-person']['ci95'])} | "
                      f"{f(d['mknn']['self-person']['diff'])} {ci(d['mknn']['self-person']['ci95'])} | "
                      f"{d['calendar_rsa']['months']['spearman']:.2f}")


if __name__ == "__main__":
    main()
