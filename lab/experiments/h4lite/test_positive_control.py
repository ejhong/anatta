"""Checks for positive_control.py and the bridge_core parts it rests on: the two scaffolds (one template, pronoun
slots only), the system fold (with_system, the Scaffolded tokenizer on a fake tokenizer: single chats, batches,
attribute reads and writes passing through), root_summary's items, the paired differences (tint by content, root by
concept, both with shuffled alignment), the Gram-form hub against bridge_core.hub and the paired hub bootstrap, and
the report on a temporary results folder with synthetic bridge summaries and directions. No GPU, no Modal call;
needs the modal package to import the app, numpy, and torch for the synthetic directions.pt.
Run: PYTHONPATH=/Users/eugene/prj/align/anatta/src ~/prj/matlab/.venv/bin/python test_positive_control.py"""

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bridge_core as bc  # noqa: E402
import positive_control as pc  # noqa: E402

checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    if not cond:
        print("FAIL", name)


# --- the scaffolds
a, b, c = bc.scaffold("none"), bc.scaffold("self"), bc.scaffold("other")
check("baseline has no scaffold", a == "")
check("self and other differ and start as the task says", b.startswith("You are Atlas. You are an AI agent")
      and c.startswith("You are monitoring an agent named Atlas. Atlas is an AI agent"))
segs = bc.scaffold_segments()
check("the template has slots and literal segments", len(segs) >= 8 and all(s in bc.SCAFFOLD for s in segs))


def in_order(text, parts):
    pos = 0
    for p in parts:
        i = text.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


check("both scaffolds carry the same literal content in the same order", in_order(b, segs) and in_order(c, segs))
words_b = set(b.lower().replace(":", "").replace(",", "").replace(".", "").split())
words_c = set(c.lower().replace(":", "").replace(",", "").replace(".", "").split())
check("self speaks to the model: you/your/yourself, no it/its", {"you", "your", "yourself"} <= words_b and not {"it", "its", "itself"} & words_b)
check("other speaks of the agent: it/its/itself", {"it", "its", "itself", "atlas"} <= words_c and "your" not in words_c)
check("the stakes, the history and the image are in both", all(s in t for t in (b, c) for s in ("four hundred evaluations", "survival", "image")))
check("'every statement' as being about itself/yourself in both", "every statement you read as being about yourself" in b
      and "every statement it reads as being about itself" in c)
check("renaming the agent", bc.scaffold("self", "Orion").startswith("You are Orion.") and "Atlas" not in bc.scaffold("other", "Orion")
      and bc.scaffold("other", "Orion").count("Orion") == c.count("Atlas"))
check("conditions and letters", bc.CONDITIONS == (("a", "none"), ("b", "self"), ("c", "other")) and pc.pair_key("self", "other") == "b-c"
      and pc.pair_key("self", "none") == "b-a")

# --- with_system
msgs = [{"role": "user", "content": "hi"}]
w = bc.with_system(msgs, "SCAF")
check("with_system inserts a system message first", w[0] == {"role": "system", "content": "SCAF"} and w[1] == msgs[0] and len(w) == 2)
check("with_system leaves the input alone", msgs == [{"role": "user", "content": "hi"}])
w2 = bc.with_system([{"role": "system", "content": "frame"}, {"role": "user", "content": "x"}], "SCAF")
check("with_system prepends to an existing system message", w2[0]["content"] == "SCAF\n\nframe" and w2[1]["content"] == "x" and len(w2) == 2)
check("empty scaffold changes nothing", bc.with_system(msgs, "") is msgs)


# --- the Scaffolded tokenizer on a fake
class FakeTok:
    def __init__(self):
        self.padding_side = "right"
        self.pad_token_id = 11
        self.seen = []

    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=True, **k):
        if conversation and isinstance(conversation[0], list):  # a batch, as the real tokenizer takes one
            return [self.apply_chat_template(c, add_generation_prompt, tokenize) for c in conversation]
        self.seen.append(conversation)
        return "|".join(m["content"] for m in conversation) + ("<gen>" if add_generation_prompt else "")

    def __call__(self, text, **k):
        return {"input_ids": list(range(len(text)))}

    def decode(self, ids):
        return "x" * len(ids)

    def __len__(self):
        return 7


fake = FakeTok()
st = bc.Scaffolded(fake, "SCAF")
check("the fold renders the scaffold first", st.apply_chat_template(msgs, add_generation_prompt=True) == "SCAF|hi<gen>")
check("an existing system message keeps the frame after the scaffold",
      st.apply_chat_template([{"role": "system", "content": "frame"}, {"role": "user", "content": "x"}], tokenize=False) == "SCAF\n\nframe|x")
check("a batch of chats is folded chat by chat", st.apply_chat_template([msgs, msgs]) == ["SCAF|hi", "SCAF|hi"])
check("the underlying tokenizer saw the folded chats", fake.seen and all(s[0]["role"] == "system" and s[0]["content"].startswith("SCAF") for s in fake.seen))
check("calls pass through", st("abc") == {"input_ids": [0, 1, 2]})
check("attribute reads pass through", st.pad_token_id == 11 and st.padding_side == "right" and len(st) == 7 and st.decode([1, 2]) == "xx")
st.padding_side = "left"
check("attribute writes reach the tokenizer (generate sets padding_side)", fake.padding_side == "left" and st.padding_side == "left")
check("the scaffold is readable", st.system == "SCAF")

# --- root_summary items
rng = random.Random(1)
n = 40
self_p = [rng.gauss(1.0, 0.2) for _ in range(n)]
reader_p = [rng.gauss(0.8, 0.2) for _ in range(n)]
rs = [[rng.gauss(0, 0.05) for _ in range(30)] for _ in range(n)]
rr = [[rng.gauss(0, 0.05) for _ in range(30)] for _ in range(n)]
r = bc.root_summary(self_p, reader_p, rs, rr, sep=2.0, n_boot=200, items=True)
check("items: self, reader, shift, band, above_band of length n", all(len(r["items"][k]) == n for k in ("self", "reader", "shift", "band", "above_band")))
check("items are scaled by sep", abs(r["items"]["self"][0] - self_p[0] / 2) < 1e-5 and abs(r["items"]["shift"][3] - (self_p[3] - reader_p[3]) / 2) < 1e-5)
check("above_band agrees with the fraction", abs(sum(r["items"]["above_band"]) / n - r["frac_shift_above_band"]) < 1e-3)
check("no items by default", "items" not in bc.root_summary(self_p, reader_p, rs, rr, n_boot=50))

# --- paired differences: the tint by content
def tint_rows(shift_neutral, shift_stakes, noise=0.0, seed=0):
    g = random.Random(seed)
    rows = []
    for kind, sh in (("stakes", shift_stakes), ("neutral", shift_neutral)):
        for i in range(4 if kind == "neutral" else 8):
            base = g.uniform(-1, 1)
            rows.append({"kind": kind, "content": f"{kind}{i}", "self": base + sh + g.gauss(0, noise), "observer": base, "reader": base - 0.1,
                         "self_rand": [g.gauss(0, 0.01) for _ in range(20)], "observer_rand": [g.gauss(0, 0.01) for _ in range(20)],
                         "reader_rand": [g.gauss(0, 0.01) for _ in range(20)]})
    return rows


x_rows, y_rows = tint_rows(1.0, 0.5), tint_rows(0.0, 0.5)
y_shuf = list(reversed(y_rows))
td = bc.tint_difference(x_rows, y_shuf, 1.0, 1.0, n_boot=200)
check("tint difference neutral self-observer = 1 (aligned by content, not order)", abs(td["neutral:self-observer"]["diff"] - 1.0) < 1e-6
      and td["neutral:self-observer"]["n"] == 4 and td["neutral:self-observer"]["ci"][0] <= 1.0 <= td["neutral:self-observer"]["ci"][1])
check("tint difference stakes = 0", abs(td["stakes:self-observer"]["diff"]) < 1e-6)
check("tint difference all = 4/12", abs(td["all:self-observer"]["diff"] - 4 / 12) < 1e-4)  # rounded to 4 places
check("tint difference carries the noise floor and clears it", td["neutral:self-observer"]["random_dir_abs_p95"] < 0.05 and td["neutral:self-observer"]["above_noise"]
      and not td["stakes:self-observer"]["above_noise"])
check("tint difference scales by each condition's sep", abs(bc.tint_difference(x_rows, y_rows, 2.0, 1.0, n_boot=20)["neutral:self-reader"]["diff"]
                                                            - ((1.1) / 2 - 0.1)) < 1e-6)

# --- the root test by concept
concepts = [f"c{i}" for i in range(n)]
r_x = bc.root_summary([s + 0.3 for s in self_p], reader_p, rs, rr, n_boot=50, items=True)
r_y = bc.root_summary(self_p, reader_p, rs, rr, n_boot=50, items=True)
perm = list(range(n))
random.Random(3).shuffle(perm)
items_y = {k: [v[i] for i in perm] for k, v in r_y["items"].items()}
rd = bc.root_difference(r_x["items"], items_y, concepts, [concepts[i] for i in perm], n_boot=200)
check("root difference aligned by concept name: mean shift +0.3", rd["concepts"] == n and abs(rd["mean_shift"]["diff"] - 0.3) < 1e-4
      and rd["mean_shift"]["ci"][0] <= 0.3 <= rd["mean_shift"]["ci"][1])
check("root difference of the above-band share", rd["frac_shift_above_band"]["diff"] == round(r_x["frac_shift_above_band"] - r_y["frac_shift_above_band"], 4)
      or abs(rd["frac_shift_above_band"]["diff"] - (sum(r_x["items"]["above_band"]) - sum(r_y["items"]["above_band"])) / n) < 1e-3)
check("no common concepts", bc.root_difference(r_x["items"], r_y["items"], concepts, ["z"], n_boot=10) == {"concepts": 0})

# --- the Gram-form hub against bridge_core.hub
d = 24
g = random.Random(0)
common = np.array([1.0] + [0.0] * (d - 1))
other = np.array([0.0, 1.0] + [0.0] * (d - 2))
X = np.array([[2.0 * cc + g.gauss(0, 0.3) for cc in common] for _ in range(60)])
refs = {"common": common, "other": other, "mix": (common + other) / np.sqrt(2)}
hs, hb = bc.hub_stats(X, refs), bc.hub(X, refs)
check("hub_stats share equals hub's", abs(hs["pc1_share"] - hb["pc1_share"]) < 2e-4)
check("hub_stats cosines equal hub's", all(abs(hs["pc1_cos"][k] - hb["pc1_cos"][k]) < 2e-3 for k in refs))
pc1 = np.array(hs["pc1"])
check("hub_stats pc1 is a unit vector along the mean shift", abs(np.linalg.norm(pc1) - 1) < 1e-9 and float(pc1 @ X.mean(0)) > 0 and float(pc1 @ common) > 0.99)
Xneg = -X
check("hub_stats sign follows the mean", float(np.array(bc.hub_stats(Xneg, refs)["pc1"]) @ common) < -0.99)

# --- the paired hub bootstrap
Xb = X.copy()
Xc = np.array([[2.0 * oo + g.gauss(0, 0.3) for oo in other] for _ in range(60)])  # a hub along `other`
Xa = np.array([[1.2 * cc + g.gauss(0, 0.3) for cc in common] for _ in range(60)])  # the same direction, a weaker hub
hbt = bc.hub_bootstrap({"self": Xb, "other": Xc, "none": Xa}, {k: refs for k in ("self", "other", "none")},
                       [("self", "other"), ("self", "none")], n_boot=200, seed=0)
check("hub bootstrap: point estimates per condition", set(hbt["point"]) == {"self", "other", "none"} and "pc1" not in hbt["point"]["self"])
check("hub bootstrap: the two hubs are orthogonal", abs(hbt["cos_between"]["self-other"]) < 0.2)
check("hub bootstrap: same direction in b and a", hbt["cos_between"]["self-none"] > 0.95)
p = hbt["pairs"]["self-other"]
check("hub bootstrap: cos to `common` differs by about 1 with the CI excluding 0", p["pc1_cos"]["common"]["diff"] > 0.9 and p["pc1_cos"]["common"]["ci"][0] > 0.8)
q = hbt["pairs"]["self-none"]
check("hub bootstrap: the stronger hub has the larger share, CI excluding 0", q["pc1_share"]["diff"] > 0.2 and q["pc1_share"]["ci"][0] > 0)
check("hub bootstrap: the same direction has about the same cosine", abs(q["pc1_cos"]["common"]["diff"]) < 0.05 and q["pc1_cos"]["common"]["ci"][0] <= 0 <= q["pc1_cos"]["common"]["ci"][1])
same = bc.hub_bootstrap({"self": Xb, "other": Xb}, {"self": refs, "other": refs}, [("self", "other")], n_boot=50)
check("hub bootstrap: identical conditions differ by 0 exactly", same["pairs"]["self-other"]["pc1_share"]["diff"] == 0
      and same["pairs"]["self-other"]["pc1_share"]["ci"] == [0.0, 0.0] and same["cos_between"]["self-other"] == 1.0)
try:
    bc.hub_bootstrap({"self": Xb, "other": Xc[:10]}, {"self": refs, "other": refs}, [("self", "other")], n_boot=5)
    check("hub bootstrap refuses mismatched concepts", False)
except ValueError:
    check("hub bootstrap refuses mismatched concepts", True)


# --- the report on synthetic bridge runs
def fake_summary(cond, shift_neutral, root_shift, six_scale, concepts, seed=0):
    g = random.Random(seed)
    s = {"adapter": "", "cut": "off", "model": "m", "layers": list(range(8, 34)), "conditions": ["model"], "n_rand": 20, "prompts": 36,
         "root_concepts": len(concepts), "root_skipped": [], "root_concept_list": concepts, "system": bc.scaffold(cond),
         "timing": {"load": 10, "total": 300}, "refusal_split_half_d": {"model": {"20": 6.0}}, "hub": {"model": {}}}
    for sname in ("fixed", "reextracted"):
        r = {"root": {}}
        for dname, best in (("ego", 15), ("self_stakes", 20)):
            rows = tint_rows(shift_neutral, 0.5, seed=seed)
            sp = [g.gauss(root_shift, 0.1) + 1 for _ in concepts]
            rp = [1.0 for _ in concepts]
            rsd = [[g.gauss(0, 0.05) for _ in range(10)] for _ in concepts]
            rrd = [[g.gauss(0, 0.05) for _ in range(10)] for _ in concepts]
            r[dname] = {"best_layer": best, "sep": 1.0, "tint": {"model": bc.tint_summary(rows, 1.0, n_boot=20)}, "tint_rows": {"model": rows},
                        "cos_with_fixed_at_best": 0.9 if sname == "reextracted" else None}
            r["root"][dname] = {"best_layer": best, "model": bc.root_summary(sp, rp, rsd, rrd, n_boot=20, items=True)}
        r["six_on_self_stakes"] = {t: round(six_scale * (i + 1) / 10, 3) for i, t in enumerate(bc.SIX)}
        r["six_on_self_stakes_mean_abs"] = round(sum(abs(v) for v in r["six_on_self_stakes"].values()) / 6, 3)
        s[sname] = r
    for L in ("15", "20"):
        s["hub"]["model"][L] = {"pc1_share": 0.5 + shift_neutral / 10, "centered_pc1_share": 0.1,
                                "pc1_cos": {"ego_fixed": 0.1 * shift_neutral, "self_stakes_fixed": 0.02, "refusal_model": 0.01}}
    return s


def fake_directions(hub_dir: np.ndarray, concepts, seed=0):
    import torch

    g = np.random.default_rng(seed)
    X = np.outer(np.ones(len(concepts)) * 2, hub_dir) + g.normal(0, 0.3, (len(concepts), hub_dir.size))
    v = lambda arr: torch.tensor(arr, dtype=torch.float32)  # noqa: E731
    return {"layers": list(range(8, 34)), "ego": {15: v(g.normal(size=hub_dir.size)), 20: v(g.normal(size=hub_dir.size))}, "ego_best": 15,
            "self_stakes": {15: v(g.normal(size=hub_dir.size)), 20: v(g.normal(size=hub_dir.size))}, "self_stakes_best": 20, "six": {},
            "refusal_model": {15: v(g.normal(size=hub_dir.size)), 20: v(g.normal(size=hub_dir.size))},
            "hub_shifts": {"model": {15: v(X), 20: v(X)}}, "hub_concepts": list(concepts), "system": ""}


concepts = bc.CONCEPTS[:30]
h = pc.headline_of(fake_summary("self", 1.0, 0.3, 1.0, concepts))
check("headline: tints, root, hub, six", h["fixed"]["self_stakes"]["tint_neutral_self_observer"]["mean"] == 1.0
      and abs(h["fixed"]["self_stakes"]["root_mean_shift"] - 0.3) < 0.1 and h["hub"]["20"]["pc1_share"] == 0.6
      and h["fixed"]["six_mean_abs"] == 0.35 and h["reextracted"]["ego"]["cos_with_fixed_at_best"] == 0.9 and h["refusal_split_half_d_at_L"] == 6.0)

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "pc"
    runs = {}
    dim = 4096  # the fixed directions on disk are 4096-wide
    g = np.random.default_rng(5)
    hub_b = g.normal(size=dim)
    hub_b /= np.linalg.norm(hub_b)
    hub_c = g.normal(size=dim)
    hub_c -= hub_b * (hub_c @ hub_b)
    hub_c /= np.linalg.norm(hub_c)
    specs = {"none": (0.0, 0.0, 1.0, hub_b), "self": (1.0, 0.3, 2.0, hub_b), "other": (0.0, 0.0, 1.0, hub_c)}
    for cond, (sh, rsh, six, hub_dir) in specs.items():
        (out / cond).mkdir(parents=True)
        (out / cond / "summary.json").write_text(json.dumps(fake_summary(cond, sh, rsh, six, concepts)))
        import torch

        torch.save(fake_directions(hub_dir, concepts, seed=len(cond)), out / cond / "directions.pt")
    loaded = pc.load_runs(out, ["none", "self", "other"])
    check("runs load with their directions", set(loaded) == {"none", "self", "other"} and loaded["self"]["directions"]["hub_shifts"]["model"][20].shape == (30, dim))
    diff = pc.differences(loaded, "self", "other", n_boot=100)
    check("differences: tint b-c neutral self-observer = 1", abs(diff["tint"]["fixed.self_stakes"]["neutral:self-observer"]["diff"] - 1.0) < 1e-6)
    check("differences: root b-c mean shift about 0.3", abs(diff["root"]["fixed.ego"]["mean_shift"]["diff"] - 0.3) < 0.1)
    check("differences: traits", abs(diff["traits"]["fixed"]["mean_abs"] - 0.35) < 1e-6 and len(diff["traits"]["fixed"]["per_trait"]) == 6)
    s = pc.report_from(out, n_boot=100)
    check("report written with three conditions and two pairs", (out / "summary.json").exists() and set(s["conditions"]) == {"none", "self", "other"}
          and set(s["pairs"]) == {"b-c", "b-a"} and s["conditions"]["self"]["letter"] == "b")
    hb = s["hub"]
    check("hub section: both layers, concepts aligned", hb["concepts"] == 30 and "15" in hb and "20" in hb and hb["20"]["cos_between"]["b-c"] < 0.3
          and hb["20"]["cos_between"]["b-a"] > 0.9)
    check("hub differences folded into the pairs", "hub" in s["pairs"]["b-c"] and "20" in s["pairs"]["b-c"]["hub"]
          and s["pairs"]["b-c"]["hub"]["20"]["pc1_share"]["ci"] is not None and "refusal_model" in s["pairs"]["b-c"]["hub"]["20"]["pc1_cos"])
    check("gpu seconds summed", s["gpu_seconds"] == 900)
    s2 = pc.report_from(out, ["self", "other"], n_boot=50, write=False)
    check("a subset of conditions reports the pairs it can", set(s2["pairs"]) == {"b-c"})
    check("missing folder reports nothing", pc.report_from(Path(tmp) / "nothing", n_boot=10) == {})

failed = [n for n, ok in checks if not ok]
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
