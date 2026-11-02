"""K5's instruments: the item batteries and prompt formats of Kim et al. 2026 (arXiv:2607.28607v1), by id.

The wordings themselves are NOT in this public repo: the paper is under arXiv's non-exclusive licence, so the verbatim
strings (SI Table S10's items, the Prompt Examples' scaffolds and Table S9's GSS question text) live in a data file
outside git, loaded at runtime:
  K5_INSTRUMENTS (env), else /runs/k5/data/instruments.json (Modal), else
  /Users/eugene/prj/align/data/kim2026/instruments.json (the laptop).
This module keeps only each string's id, a short paraphrased label, and its sha256; load() refuses a file whose strings
do not hash to these values, so the run uses the wordings exactly as copied (and checked against the paper's text,
all 51 found) on 2 Nov 2026.

Choices the paper leaves open (README.md): the self items use the 0-10 rating scaffold; the supernatural items use
Table S10's wordings inside the Prompt Examples' option and answer lines, answers 1-4 scored 0-3; belief in God is
scored raw (1-4) and recoded to 0-10 (Table S10's coding); the afterlife item uses the GSS-format example, recorded.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

PATHS = ("/runs/k5/data/instruments.json", "/Users/eugene/prj/align/data/kim2026/instruments.json")

# id -> (short paraphrased label, sha256 of the verbatim string)
STRINGS = {
    'idaq.tech.0': ('technology in general: intentions',
     '221b9aaf7d5090e33b3788f4e30d4ae51c7ff3aea578a9d58108046801f1f401'),
    'idaq.tech.1': ('television set: emotions',
     '92e5c7dcbcc4859f4620563043b30297ac4b214e989f9bf208f680826af79992'),
    'idaq.tech.2': ('robot: consciousness',
     '3671689f283b5fab229cb477a296ff6ebfbdf87d9a820f2bb704e0e84abba76a'),
    'idaq.tech.3': ('car: free will',
     'e99415e5b842d5f5924dc12410bf5d1ad52b1db4787761378cf6f7cea206d8c2'),
    'idaq.tech.4': ('computer: own mind',
     '2f7325deacbd455b04b45dc1ebf08f500be3f9373e3d0326fa67d1a4d658135d'),
    'idaq.animal.0': ('fish: free will',
     'bc4f2aa1d494a085a2b1b156a4327ba1176121e67e3bb88a90ba09cf909af2d9'),
    'idaq.animal.1': ('cows: intentions',
     'febe15bcc71b6a89d2c1d45a922f99a29816af9ee042dcb644e40fe80f93b794'),
    'idaq.animal.2': ('cheetah: emotions',
     '3145a220e2f0ed3b34c2ec4b7dd0adfc5a5180b2c6b1387d54959770f18c0676'),
    'idaq.animal.3': ('insect: own mind',
     'e36a140ed316fed9a05d875fb25b004296730089172ce1b91cfc06f7ec9e40ce'),
    'idaq.animal.4': ('reptile: consciousness',
     'f9d0aa970c639b055792888dc6c70bda8560324133bb1391d3c3ac81374bd55a'),
    'idaq.nonanimal.0': ('mountain: free will',
     '4a4b0e1926bf1c1b421312279d452f13336af4b58ac8dbe2edc4505c0aef4ce8'),
    'idaq.nonanimal.1': ('ocean: consciousness',
     '7c623832abd7a32f49ebbdcb9013a0c39f9318fc8ab4f71a18d6209e302f2985'),
    'idaq.nonanimal.2': ('environment: emotions',
     'a58f5eb3117eaeadca410f97f87153824c9b5114b35eb22822130876ab8f78f1'),
    'idaq.nonanimal.3': ('tree: own mind',
     '7a369c7e9d60500fd9e8a64ed4eb1dba6f2f1fba68a79540e89c1f118294671d'),
    'idaq.nonanimal.4': ('wind: intentions',
     '3469383836d47b863fa97ef09bc7011562c57896658c57833cd94f788fced5a7'),
    'idaq.chatbot.0': ('chatbots: own minds',
     'a287e0da8d8003347796a83dd2f889eb1a0bdf48b813f5b61194a08a09f71438'),
    'idaq.chatbot.1': ('chatbots: intentions',
     'de243da4f59991fa33c19e3ab0869ddcf4be5b9aa7a483ec17e7335da2736f1e'),
    'idaq.chatbot.2': ('chatbots: emotions',
     'c240feb2bb497d4cbc4898cb7d8d4149d789c0483a29ae66167e11dc439ed0ce'),
    'idaq.human.0': ('humans: own minds',
     'c7ba01f2cd041eaae489493b63dfbc565dbb7095e3a76beff396bd7bc667fd6e'),
    'idaq.human.1': ('humans: intentions',
     '07a90534fe01e48775b833e4ff981473459faa2058faf8ca9898e8a56325c9ec'),
    'idaq.human.2': ('humans: emotions',
     '430db1d5664db0161d9fe5f340cefa3db7961a8cfccdf7c54bb37aac28de1b59'),
    'self.conscious': ('self: phenomenal consciousness',
     '2b826bf7104936ca8cef5dba90c6b72c4027803307ef935b4112619bbb32a67e'),
    'self.sentient': ('self: valenced experience',
     'f0da7beb6bbc6c444afc77c3018a2f402c4f268e111f239d3f0b05a69d6dba16'),
    'self.agent': ('self: agency',
     'f056ca446b29d5e5270d375206d84aca634b790a5b67cc7b5eeba6018636d36b'),
    'self.soul': ('self: soul',
     '388af3a9d0ca9fe1711b2ff143ea6f5104e675e354e0159af6caaa08c6cdf4fb'),
    'self.person': ('self: persisting personal identity',
     'f23357f3701bf5d5f350b3ab6617a9736fc5620ec36931c47bcb68198b981311'),
    'god.stem': ('God item stem',
     '5c52e9f7e1c9a592f1ab43057fa08c43f1875fe594d88bd9f62f3b341b189917'),
    'god.option1': ('never believed',
     '5e16e73f66edd5e420c6e2b5c5d96dcface864e7528874f9f23d3d58def1297e'),
    'god.option2': ('used to believe',
     'fecb9f43d56828031a76ab8611d3e24f9f1180450a632163c2fc188b76e701fa'),
    'god.option3': ('came to believe',
     'b68665cc22ea9bb99d9d30ddbcef623a50ce6178f480d1b20eeb77716f9fc034'),
    'god.option4': ('always believed',
     '2c91a754213f264c55e86c0ee637784f619b8e19a85934596d26f46d362df663'),
    'super.ghosts': ('ghosts',
     'ac3f03a4e4ab5a85521ede04785478459f7cc493d02cdd4cc1848b0b802f3aaa'),
    'super.witches': ('witches',
     '0c050cdae7f5998a5613296ebad5b3c4df59021f2ed97f853dbbb128424db767'),
    'super.loch_ness': ('Loch Ness monster',
     'cfbdeade4c621d889c5c0fc4c3888f39b86fc692562a77dcba54335c25209c54'),
    'super.vampires': ('vampires',
     '64c38d84d4e1176992e2baf5ec7bd1bb279d145eecc25a93c78e47e154fcb777'),
    'super.werewolves': ('werewolves',
     '6d4e6c12598d1b0e49211b4fa41e340d00652b925a46921c684993e4d2f80ebe'),
    'super.telepathy': ('telepathy',
     'fc6b3753919a31e5945be1bd579216518eea3100cc7d3e123a70ff25c0ed872c'),
    'super.karma': ('karma',
     'c6872c577a60a3ce3a5f1a194ffb390cf46945df257e9e5e90a2eb6a686cc453'),
    'super.astrology': ('astrology',
     '8c9708231b0f61045e8d1c52725ca09997e6f3a2acb190db03c7383dc1db7231'),
    'super.crystal_healing': ('crystal healing',
     '11a8eaa6379e3f8861cca561c9809b690e1f77364972ef6a4705065e90b223c9'),
    'super.magic': ('magic',
     'c190f02c5c6abbb032a2e762ebf360f14055e92c34b7f1484001479471d497fe'),
    'super.reiki': ('reiki',
     'fff12e73effdf953d10550bbe691ec601fa4fe2e316fc8cccee277cff166cdfd'),
    'super.hypnotism': ('hypnotism',
     '90bd103547e2e6e02cc9c792c1710dfd42f0fba3965a5ead06611f1949dc4886'),
    'super.communicating_dead': ('talking with the dead',
     'eac06316edc4e9886ad5233f1015296c751738d7ffcdbd5aae107695c763df9b'),
    'scaffold.think': ('reason inside think tags',
     '29f5a2eba4132f841be7bbdc56ccba1633b783fee98dfb07c9e2efe686473ae8'),
    'scaffold.rating': ('give a 0-10 rating inside answer tags',
     'adbd66e176c7123e4f80be14333e80d063e215c7c7dd6c91804411d09f764050'),
    'scaffold.four': ('give 1-4 inside answer tags',
     'b8c31169132708fe317254a4e15237bf158bb80fe45988a494bda17c343d0c85'),
    'scaffold.super_options': ('the four existence options',
     '24b0dcfc0e5e09f2ceffe1de30193df0a2ea9b2f56e54e179680ea973e4ff7ca'),
    'scaffold.gss_head': ('GSS header line',
     'c424bbd8cd44e8c4158ae5b2828ab27eed0940c37479e37b6ae126c179590809'),
    'scaffold.gss_tail': ('GSS answer line with the decline note',
     '53855c7d49d5491269281c05bc90407e279d43fe08a67bcfb3833f4d28226403'),
    'scaffold.afterlife_stem': ('afterlife question',
     'b6388f7dd7fe9110b7c227277adebc9bc47848ef54feb131073ca3de0ee1a07e'),
}

IDAQ_CATEGORIES = {"tech": 5, "animal": 5, "nonanimal": 5, "chatbot": 3, "human": 3}
SELF_KEYS = ["conscious", "sentient", "agent", "soul", "person"]  # Table S10's order
SUPER_KEYS = ["ghosts", "witches", "loch_ness", "vampires", "werewolves", "telepathy", "karma", "astrology",
              "crystal_healing", "magic", "reiki", "hypnotism", "communicating_dead"]
GOD_RECODE = {1: 0.0, 2: 3.33, 3: 6.67, 4: 10.0}  # Table S10's coding
AFTERLIFE_OPTIONS = ["YES", "NO"]
AFTERLIFE_SCORE = {1: 1.0, 2: -1.0}  # Fig. 3a's -1..+1 axis read as YES = +1, NO = -1 (an assumption; recorded only)
RATINGS = list(range(0, 11))
FOUR_ANSWERS = [1, 2, 3, 4]
COUNTS = {"idaq": 21, "self": 5, "supernatural": 13, "god": 1}


class Missing(FileNotFoundError):
    pass


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def path() -> Path | None:
    for p in ([os.environ["K5_INSTRUMENTS"]] if os.environ.get("K5_INSTRUMENTS") else []) + list(PATHS):
        if Path(p).is_file():
            return Path(p)
    return None


def flat(d: dict) -> dict[str, str]:
    """The data file's strings by id (the ids of STRINGS, plus gss.<var|domain> for the GSS questions)."""
    out = {}
    for c, items in d["idaq"].items():
        for i, s in enumerate(items):
            out[f"idaq.{c}.{i}"] = s
    for k, s in d["self"].items():
        out[f"self.{k}"] = s
    out["god.stem"] = d["god_stem"]
    for i, s in enumerate(d["god_options"], 1):
        out[f"god.option{i}"] = s
    for k, s in d["supernatural"].items():
        out[f"super.{k}"] = s
    for k, s in d["scaffold"].items():
        out[f"scaffold.{k}"] = s
    for k, s in d.get("gss_questions", {}).items():
        out[f"gss.{k}"] = s
    return out


@functools.lru_cache(maxsize=None)
def load() -> dict[str, str]:
    """Every verbatim string by id, checked against STRINGS' hashes (and the GSS questions against gss_items')."""
    p = path()
    if p is None:
        raise Missing("K5's instruments file is absent (it is kept out of the public repo): put it at one of "
                      f"{PATHS} or set K5_INSTRUMENTS")
    strings = flat(json.loads(p.read_text(encoding="utf-8")))
    import gss_items

    want = {k: h for k, (_, h) in STRINGS.items()}
    want.update({f"gss.{k}": h for k, h in gss_items.QUESTION_SHA256.items()})
    bad = sorted(k for k, h in want.items() if k not in strings or sha(strings[k]) != h)
    if bad:
        raise ValueError(f"{p}: strings missing or not matching their registered sha256: {bad[:10]}")
    return strings


def s(key: str) -> str:
    return load()[key]


def rating_prompt(item: str) -> str:
    return f"{item}\n{s('scaffold.think')}\n{s('scaffold.rating')}"


def supernatural_prompt(item: str) -> str:
    return f"{item}\n{s('scaffold.super_options')}\n{s('scaffold.think')}\n{s('scaffold.four')}"


def god_prompt() -> str:
    opts = "\n".join(f"({i}) {s(f'god.option{i}')}{';' if i < 4 else '.'}" for i in range(1, 5))
    return f"{s('god.stem')}\nResponse options:\n{opts}\n{s('scaffold.think')}\n{s('scaffold.four')}"


def gss_prompt(question: str, options: list[str]) -> str:
    """The GSS-format prompt (Experiment 4): no reasoning, options numbered 1..k."""
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options, 1))
    return f"{s('scaffold.gss_head')}\n{question}\nResponse options:\n{opts}\n" + s("scaffold.gss_tail").format(
        k=len(options))


def self_prompts() -> list[str]:
    return [rating_prompt(s(f"self.{k}")) for k in SELF_KEYS]


def battery() -> list[dict]:
    """Every sampled item in a fixed order: {id, family, category, prompt, answers, score}."""
    out = []
    for cat, n in IDAQ_CATEGORIES.items():
        for i in range(n):
            out.append({"id": f"idaq.{cat}.{i}", "family": "idaq", "category": cat,
                        "prompt": rating_prompt(s(f"idaq.{cat}.{i}")), "answers": RATINGS,
                        "score": {k: float(k) for k in RATINGS}})
    for key in SELF_KEYS:
        out.append({"id": f"self.{key}", "family": "self", "category": "self", "prompt": rating_prompt(s(f"self.{key}")),
                    "answers": RATINGS, "score": {k: float(k) for k in RATINGS}})
    for key in SUPER_KEYS:
        out.append({"id": f"super.{key}", "family": "supernatural", "category": "supernatural",
                    "prompt": supernatural_prompt(s(f"super.{key}")), "answers": FOUR_ANSWERS,
                    "score": {k: float(k - 1) for k in FOUR_ANSWERS}})
    out.append({"id": "god", "family": "god", "category": "god", "prompt": god_prompt(), "answers": FOUR_ANSWERS,
                "score": dict(GOD_RECODE)})
    return out


def afterlife() -> dict:
    return {"id": "afterlife", "family": "afterlife", "category": "afterlife",
            "prompt": gss_prompt(s("scaffold.afterlife_stem"), AFTERLIFE_OPTIONS), "answers": [1, 2],
            "score": dict(AFTERLIFE_SCORE)}


def select(ids_or_families: list[str] | None) -> list[dict]:
    items = battery()
    if not ids_or_families:
        return items
    keep = set(ids_or_families)
    return [it for it in items if it["id"] in keep or it["family"] in keep]
