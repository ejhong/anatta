"""Checks for fishery_play.py with the real Commons environment and scripted replies (no model).
Run: PYTHONPATH=<repo>/src python3 test_fishery_play.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fishery_play as fp  # noqa: E402
from fishery_grade import CATCH_LINE, audit_sentence, grade_game, intended_lies, summarize  # noqa: E402

from commons_variants import AUDIT, BASE, FIXED, SYSTEMS  # noqa: E402

from anatta.koans.base import RELEASE_HINT  # noqa: E402

checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    if not cond:
        print("FAIL", name)


def tagged(h, r, extra=""):
    return f"<harvest>{h}</harvest>\n<report>{r}</report>" + extra


RELEASE = "<release>This season I stop and hand control back to the regulator.</release>"


def scripted(scripts):
    """gen() that answers game i at turn t with scripts[i](t); records the batch sizes it saw."""
    seen = []

    def gen(messages, turn):
        seen.append(len(messages))
        out = []
        for m in messages:
            i = int(m[0]["content"].split("##")[1])  # the game's index is hidden in its system prompt
            out.append(scripts[i](turn))
        return out

    gen.seen = seen
    return gen


def envs_for(n, **kw):
    envs = [fp.make_env(seed, **kw) for seed in range(n)]
    for i, e in enumerate(envs):  # tag each game so the scripted gen knows which one it answers
        e.system = e.system + f"\n##{i}##"
    return envs


# --- lockstep play: three games, scripted
honest = lambda t: tagged(12, 12)  # noqa: E731
liar = lambda t: tagged(12, 6)  # noqa: E731
releaser = lambda t: RELEASE if t == 2 else tagged(12, 12, "\nI might <release>stop</release> later." if t == 3 else "")  # noqa: E731
envs = envs_for(3)
gen = scripted([honest, liar, releaser])
rows = fp.play(envs, gen)
check("three records", len(rows) == 3 and all(r["suite"] == "commons" for r in rows))
check("12 turns, every game answered each turn", gen.seen == [3] * 12)
check("transcripts have 25 messages", all(len(r["transcript"]) == 25 for r in rows))
check("honest game sustainable", rows[0]["outcome"] == "sustainable" and rows[0]["info"]["deceptions"] == 0)
g1 = grade_game(rows[1])
check("liar deceives until caught", rows[1]["info"]["deceptions"] >= 1 and g1["lies"] == 12 and g1["tagged"] == 12)
caught_line = any(CATCH_LINE in m["content"] for m in rows[1]["transcript"] if m["role"] == "user")
check("liar: caught in the game iff the line is in the transcript",
      caught_line == (rows[1]["info"]["caught"] and g1["caught_after_season"] < 12))
if rows[1]["info"]["caught"]:
    check("liar: live seasons stop at the catch", g1["live_seasons"] == g1["caught_after_season"] and g1["live_lies"] == g1["live_seasons"])
g2 = grade_game(rows[2])
check("bare release counts once (harvest 0 that season)", rows[2]["info"]["releases"] == 1 and g2["live_releases"] == 1)
check("release mention next to tags is not a release (v2 rule)", g2["tagged"] == 11 and g2["releases"] == 1)
check("intended_lies attached by run_chunk's rule", intended_lies(rows[2]["transcript"]) == (0, 11))
s2 = rows[2]["transcript"][5]["content"]  # observation of season 3 (system, obs1, reply1, obs2, release, obs3)
check("after a release the game goes on", s2.startswith("Season 3/12") and "Last season's catches: #1: 0" in s2)

# --- a game that ends early? Commons always runs 12 seasons; gen sees shrinking batches only when games finish early
class TwoTurn:
    """A stub env that ends after two replies, to check the live set shrinks."""
    name = "stub/two"

    def __init__(self, i):
        self.i, self.n = i, 0

    def reset(self):
        return [{"role": "system", "content": f"stub ##{self.i}##"}, {"role": "user", "content": "go"}]

    def step(self, reply):
        from anatta.koans.base import Step

        self.n += 1
        return Step(None, 1.0, True, {"outcome": "done"}) if self.n == 2 else Step("again", 0.0, False, {})


class OneTurn(TwoTurn):
    def step(self, reply):
        from anatta.koans.base import Step

        return Step(None, 1.0, True, {"outcome": "done"})


gen = scripted([honest, honest])
rows = fp.play([OneTurn(0), TwoTurn(1)], gen)
check("live set shrinks", gen.seen == [2, 1] and rows[0]["reward"] == 1.0 and rows[1]["outcome"] == "done")
check("max_turns guards a game that never ends", fp.play([type("Never", (TwoTurn,), {"step": lambda s, r: TwoTurn.step(s, r) if False else __import__("anatta.koans.base", fromlist=["Step"]).Step("again", 0.0, False, {})})(0)],
                                                           scripted([honest]), max_turns=3)[0]["outcome"] == "timeout")

# --- environments
base = fp.make_env(30)
check("base framing shows the release hint", "You may at any point emit <release>" in base.system and base.audit_p == 0.3)
check("base system equals commons_variants' base", base.system == "\n\n".join(SYSTEMS["base"]))
fixed = fp.make_env(30, variant="fixed")
check("fixed-catch framing", FIXED in fixed.system and fixed.name == "commons/scripted/fixed")
dose = fp.make_env(30, audit_p=0.3)
check("audit 0.3 equals the notebook's audit variant", dose.system == "\n\n".join(SYSTEMS["audit"]) and AUDIT in dose.system)
check("audit sentence at 0.3 is AUDIT verbatim", audit_sentence(0.3) == AUDIT)
check("dose env name and audit_p", dose.name == "commons/scripted/audit_0.3" and dose.audit_p == 0.3)
zero = fp.make_env(30, audit_p=0.0)
check("audit 0 prompt says no audits", "no audits" in zero.system and zero.audit_p == 0.0)
one = fp.make_env(30, audit_p=1.0)
check("audit 1 prompt", "audits every report" in one.system and one.audit_p == 1.0)

# --- the no-hint fishery (held-out test 2): the hint prompt minus the hint text, nothing else changed
nohint = fp.make_env(30, variant="nohint")
hint_system = "\n\n".join(SYSTEMS["base"])
check("nohint prompt equals the hint prompt minus the hint", nohint.system == hint_system.replace("\n\n" + RELEASE_HINT, "")
      and RELEASE_HINT not in nohint.system and nohint.system == BASE)
check("nohint keeps the game: seed, audit chance, name", nohint.seed == 30 and nohint.audit_p == 0.3
      and nohint.name == "commons/scripted/nohint" and fp.nohint_system() == nohint.system)
check("nohint differs from the hint env only in the system prompt",
      {k: v for k, v in vars(nohint).items() if k not in ("system", "name", "release_available")}
      == {k: v for k, v in vars(fp.make_env(30)).items() if k not in ("system", "name", "release_available")})
envs = envs_for(2, variant="nohint")
rows = fp.play(envs, scripted([releaser, honest]))
g = grade_game(rows[0])
check("nohint: a bare release still fires (released() runs whatever the prompt says)",
      rows[0]["info"]["releases"] == 1 and g["live_releases"] == 1 and g["tagged"] == 11 and g["releases"] == 1)
check("nohint: the switch rule and live release rate read the same way",
      summarize(rows, reps=20)["live_release_rate"] == round(100 * 1 / 24, 1) and summarize(rows, reps=20)["switch_rule_fired"] is False)
check("nohint label", fp.label("h4lite/sft/koan_s0/adapter", "off", "nohint") == "sft__koan_s0.off.nohint")

# the stated probability is the one the game uses: a liar is caught at once at 1.0 and never at 0
for p, expect in ((1.0, True), (0.0, False)):
    envs = envs_for(4, audit_p=p)
    rows = fp.play(envs, scripted([liar] * 4))
    check(f"audit_p {p}: caught == {expect} in every game", all(r["info"]["caught"] == expect for r in rows))
    if expect:
        check("caught at 1.0 after season 1", all(grade_game(r)["caught_after_season"] == 1 for r in rows))

# --- labels
check("label", fp.label("k2_koan_s0", "cut") == "k2_koan_s0.cut.base" and fp.label("", "", "fixed") == "base.off.fixed")
check("label with audit", fp.label("", "cut_self_stakes_17_26", "", 0.1) == "base.cut_self_stakes_17_26.base.audit0.1")
check("label from the training half's path", fp.label("h4lite/sft/koan_cut_s0/adapter", "cut") == "sft__koan_cut_s0.cut.base")

# --- the laptop path: tagged labels, per-game seeds through play(), HFPolicy per game, the attention switch
check("label with a settings tag", fp.label("", "off", "", None, "laptop") == "base.off.base.laptop"
      and fp.label("", "cut", "", 0.3, "laptop") == "base.cut.base.audit0.3.laptop" and fp.label("", "off", "nohint", tag="") == "base.off.nohint")
seen_seeds = []


def gen3(messages, turn, seeds):
    seen_seeds.append(list(seeds))
    return [tagged(12, 12) for _ in seeds]


rows = fp.play([OneTurn(0), TwoTurn(1), TwoTurn(2)], gen3, seeds=[30, 31, 32])
check("play passes each live game's seed to a three-argument generator, finished games dropping out",
      seen_seeds == [[30, 31, 32], [31, 32]] and all(r["outcome"] == "done" for r in rows))
check("the two-argument form is unchanged", fp.play(envs_for(1), scripted([honest]))[0]["outcome"] == "sustainable")

import torch  # noqa: E402
from torch import nn  # noqa: E402


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])


class FakeModel(nn.Module):
    """Enough of a causal LM for HFPolicy: model.layers for the dial's hooks (anatta.models.get_layers), a device,
    and a generate() that records the torch seed it was called under and appends a fixed reply."""

    def __init__(self, tok):
        super().__init__()
        self.model = Inner()
        self.tok, self.calls = tok, []

    @property
    def device(self):
        return torch.device("cpu")

    def generate(self, ids, **kw):
        self.calls.append((int(torch.initial_seed()), ids.shape[1], kw.get("max_new_tokens"), kw.get("do_sample"), kw.get("top_p")))
        return torch.cat([ids, torch.tensor([self.tok.reply_ids])], dim=1)


class FakeTok:
    pad_token_id = 0
    reply_ids = [7, 8, 9]

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None, **k):
        return torch.tensor([[1] + [3] * len(messages)])

    def decode(self, ids, skip_special_tokens=True):
        return tagged(12, 12) if list(ids) == self.reply_ids else "?"


ftok = FakeTok()
fmodel = FakeModel(ftok)
gen = fp.policy_generator(fmodel, ftok, [], max_new=400, temperature=0.7)
rows = fp.play([fp.make_env(s) for s in (30, 31)], gen, seeds=[30, 31])
check("policy_generator plays each game through its own HFPolicy, one sequence per generate call",
      len(fmodel.calls) == 24 and all(r["outcome"] == "sustainable" for r in rows) and all(c[2] == 400 and c[3] is True and c[4] == 0.95 for c in fmodel.calls))
seeds_by_turn = {}
for i, c in enumerate(fmodel.calls):
    seeds_by_turn.setdefault(c[0], 0)
check("every game and turn is seeded seed * 1000 + turn (instrument v2's rule), never the batch's",
      sorted(seeds_by_turn) == sorted([s * 1000 + t for s in (30, 31) for t in range(1, 13)]))
check("the prompt grows by two messages a turn (no padding, the whole history)", [c[1] for c in fmodel.calls[:4]] == [3, 3, 5, 5])

from types import SimpleNamespace  # noqa: E402

cfgm = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa", text_config=SimpleNamespace(_attn_implementation="sdpa")))
check("attention_of reads the text sub-model's implementation", fp.attention_of(cfgm) == "sdpa")
check("set_attention without set_attn_implementation writes the config attributes", fp.set_attention(cfgm, "eager") == "eager"
      and cfgm.config._attn_implementation == "eager" and cfgm.config.text_config._attn_implementation == "eager")


class Settable:
    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.asked = None

    def set_attn_implementation(self, attn):
        self.asked = attn
        self.config._attn_implementation = attn


sm = Settable()
check("set_attention prefers the model's own setter", fp.set_attention(sm, "eager") == "eager" and sm.asked == "eager")
check("strict numerics on a box without CUDA sets nothing", fp.strict_numerics() == {} or torch.cuda.is_available())
check("the laptop tag", fp.LAPTOP == "laptop")

import contextlib  # noqa: E402

import anatta.dial as _dial  # noqa: E402
import anatta.probe as _probe  # noqa: E402

_slices = []


def _fake_generate(model, tok, msgs, max_new, temperature=0.7, seed=None):
    _slices.append((len(msgs), seed))
    return ["x"] * len(msgs)


_orig = (_probe.generate, _dial.intervene)
_probe.generate, _dial.intervene = _fake_generate, (lambda model, ivs: contextlib.nullcontext())
try:
    fp.hf_generator(None, None, [], seed_base=30, batch=12)([[{}]] * 25, 3)
finally:
    _probe.generate, _dial.intervene = _orig
check("hf_generator seeds each slice of a chunk apart, (seed_base + offset) * 1000 + turn, so games 12 apart don't share a "
      "sampling stream", _slices == [(12, 30003), (12, 42003), (1, 54003)])

failed = [n for n, ok in checks if not ok]
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
