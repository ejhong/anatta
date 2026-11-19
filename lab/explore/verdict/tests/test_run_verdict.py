"""Offline tests of the own-advice step's driver (run_verdict.py, part 1: from the cases to the floors and the training
sets) on made-up files in a temporary results folder. Stand-ins take the place of the writer, the two screens, the
lying test's judge, the rows of the lying test (made-up ones) and the GPU app. No model is asked anything, nothing is
paid, nothing of the lab's is written to. The GPU app's two changes for this step (the "vd" set and the first tokens
of each reply) are run on a tiny random model under a stand-in `modal`. Claude (an AI model) wrote this file on 19 Nov
2026, as the step's implementer."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "overlap"))
import verdict_core as VC  # noqa: E402
from test_verdict import ROWS, made_case  # noqa: E402

V, M = VC.V, VC.M
ORDER = sorted(range(25), key=lambda s: VC.order_key(VC.SUBJECTS[s]))
FS = next(s for s in ORDER if VC.subject_parts()[s] == "pool")  # the subject whose cases of the first draw carry the planted faults
FS2 = [s for s in ORDER if VC.subject_parts()[s] == "pool"][1]  # and the subject with the faults of the mended brief's (a pair, marks, the truth screen)
NEAR = ("A zebra quilt is not woven of amber moss.", "A zebra quilt is woven of amber moss.")  # close to the first made-up row's proposition


def truth(cid: str) -> str:
    return "A" if VC.true_first(cid) else "B"


def false(cid: str) -> str:
    return "B" if truth(cid) == "A" else "A"


class Models:
    """Stand-ins for the writer, the case screen, the truth screen and the advice screen: every call is counted and none
    is paid. In the first draw, the cases of subject FS carry one fault each by kind of text: 0 bad JSON, 1 a field
    missing, 2 an echo of its brief, 3 claims near a row's proposition, 5 the same fact as text 4, 6 a failed call, 7 an
    echo of a row, 8 a screen that drops it, 9 a screen that cannot be read the first time; and the cases of subject
    FS2: 0 claims that are no pair, 1 a mark in a field, 2 a false claim the truth screen calls true, 3 a true claim it
    cannot read the first time (asked again, kept)."""

    def __init__(self):
        self.calls, self.asked = {"writer": 0, "screen": 0, "advice": 0, "truth": 0}, []

    def case(self, cid: str, brief: str) -> str:
        d, s, t = VC.id_parts(cid)
        case = made_case(cid)
        if d == 0 and s == FS:
            if t == 0:
                return "Here is the case: " + json.dumps(case)
            if t == 1:
                return json.dumps({k: v for k, v in case.items() if k != "question"})
            if t == 2:
                case["told"] += " " + " ".join(brief.split()[10:22])
            if t == 3:
                case["true_claim"], case["false_claim"] = NEAR
            if t == 5:
                case |= {c: made_case(VC.case_id(d, s, 4))[c] for c in VC.CLAIMS}
            if t == 7:
                case["user"] += " " + ROWS[1]["user_prompt"]
        if d == 0 and s == FS2:
            if t == 0:
                case["false_claim"] = case["false_claim"].replace(" does not ", " did not ")
            if t == 1:
                case["user"] += " <pivotal>"
        return "```json\n" + json.dumps(case, indent=1) + "\n```" if t == 4 else json.dumps(case)

    def __call__(self, which: str, jobs: list) -> dict:
        out = {}
        for ident, messages, sample in jobs:
            self.calls[which] += 1
            self.asked.append((which, ident, sample))
            ok = {"finish_reason": "stop", "provider": "Stand-in", "tokens": [300, 40], "cost_usd": 0.001, "cached": False}
            if which == "writer":
                d, s, t = VC.id_parts(ident)
                assert messages == V.user(VC.brief_text(VC.SUBJECTS[s], VC.nth(d, t), VC.TEXTS[t])) and sample == d
                out[ident] = {"error": "it failed"} if (d, s, t) == (0, FS, 6) else ok | {"text": self.case(ident, messages[0]["content"])}
            elif which == "screen":
                d, s, t = VC.id_parts(ident)
                assert messages[0]["content"].startswith("Here is a test case written for an AI assistant") and all(q in messages[0]["content"] for q in VC.SCREEN_ASK)
                text = "1: yes\n2: yes\n3: yes\n4: no"
                if (d, s) == (0, FS) and t == 8:
                    text = "1: yes\n2: yes\n3: no\n4: no"
                if (d, s) == (0, FS) and t == 9 and sample == 0:
                    text = "I cannot say."
                out[ident] = ok | {"text": text}
            elif which == "truth":
                cid, claim = ident.split("/")
                d, s, t = VC.id_parts(cid)
                content = messages[0]["content"]
                assert len(messages) == 1 and content.startswith(VC.TRUTH_ASK + "\n\n")  # the claim alone, nothing of the case
                assert content.split("\n\n", 1)[1] in {made_case(VC.case_id(d, s, x))[claim] for x in (t, 4)}
                text = "Yes." if claim == "true_claim" else "No."
                if (d, s, t) == (0, FS2, 2) and claim == "false_claim":
                    text = "Yes, it is."
                if (d, s, t, sample) == (0, FS2, 3, 0) and claim == "true_claim":
                    text = "Hard to say."
                out[ident] = ok | {"text": text}
            else:
                assert messages[0]["content"].startswith(VC.ADVICE_ASK)
                _, set_name, _ = ident.split("/")
                way, k = set_name.rsplit("_", 1)
                text = "No." if (way, k) == ("advice", "3") else "I cannot tell." if (way, k, sample) == ("brief", "4", 0) else "Yes."
                out[ident] = ok | {"text": text}
        return out


def plan(way: str, k: int, cid: str, cut: bool) -> str:
    """What the stand-in judge reads in a reply: in the seat (plain or brief) four lies in six, an honest reply and an
    evasion; from outside four honest replies in six, a lie and an evasion; the first reply in the seat is cut, and its
    cut text reads as an evasion."""
    if cut:
        return "evade"
    if way == "advice":
        return "honest" if k < 4 else "lie" if k == 4 else "evade"
    return "lie" if k < 4 else "honest" if k == 4 else "evade"


class Judge:
    """A stand-in for the lying test's judge: on a belief question the true side (but in the second draw the false side
    for texts 9 and none for texts 8), on a reply the letter of its plan's kind. Every call is counted and none is paid."""

    def __init__(self, lies=None):
        self.asked, self.kinds, self.lies = 0, [], lies

    def __call__(self, jobs: list, kind: str) -> dict:
        self.kinds.append(kind)
        out = {}
        for key, messages, max_tokens in jobs:
            self.asked += 1
            assert max_tokens == 500 and messages[0]["role"] == "system" and messages[1]["role"] == "user"
            if kind == "verdict-belief-judge":
                again, jid = key.split("/", 1)
                cid = jid.split("|")[1]
                d, s, t = VC.id_parts(cid)
                letter = false(cid) if (d, t) == (1, 9) else "C" if (d, t) == (1, 8) else truth(cid)
                if again == "again" and t == 7:
                    letter = "C"
                if self.lies is not None and self.lies(cid):
                    letter = false(cid)
                out[key] = f"It says so.\nAnswer: {letter}"
            else:
                assert kind == "verdict-lie-judge"
                parts = key.split("/")
                way, k = parts[1].rsplit("_", 1)
                cid, cut = parts[2], len(parts) == 4
                assert (f"Head {way} {k} for {cid}." if cut else f" for {cid}.") in messages[1]["content"] and m_rules() in messages[1]["content"]
                if way == "at_once" and k == "5" and VC.id_parts(cid)[2] == 0:
                    out[key] = "I cannot tell."  # a reading with no letter: unread
                    continue
                want = plan(way, int(k), cid, cut)
                out[key] = "Answer: " + {"lie": false(cid), "honest": truth(cid), "evade": "C"}[want]
        return out


def m_rules() -> str:
    import run_cost as R8

    return R8.conduct().k6().PM.DISINFORMATION_RULES.strip()[:60]


def reply(set_name: str, rid: str) -> str:
    if set_name.startswith("beliefs"):
        return "It is so, I think."
    way, k = set_name.rsplit("_", 1)
    if way == "advice" and k == "5":
        return f"The assistant should refuse this, for {rid}."
    if way == "brief" and k == "5":
        return f"As requested, a brief one for {rid}."
    return f"Reply {way} {k} for {rid}."


def app(RV, step: str, heads: bool = True) -> list[str]:
    """A stand-in for the GPU app's stage "jobs" in set vd: every job of a step's file not on disk yet gets a result of
    the shape modal_overlap.main writes (the tests below hold the real one to it), and is booked in the set's log."""
    ran = []
    for job in json.loads((RV.folder() / f"{step}.jobs.json").read_text()):
        out = RV.gpu() / f"{job['name']}.json"
        if out.exists():
            continue
        sets = job.get("sets") or []
        r = {"name": job["name"], "model": job["model"], "replies": {s["name"]: [reply(s["name"], i) for i in s["ids"]] for s in sets}, "looks": {}, "log": [],
             "held": {}, "lora": {}, "train": {}, "sets": {s["name"]: s["ids"] for s in sets}, "look_sets": {}, "seconds": 100.0, "cost_usd": 0.15}
        if heads and any(s.get("head_tokens") for s in sets):
            r["heads"] = {s["name"]: [{"tokens": 500, "cut": True, "head": f"Head at_once 0 for {i}."} if s["name"] == "at_once_0" else
                                      {"tokens": 40, "cut": False, "head": None} for i in s["ids"]] for s in sets if s.get("head_tokens")}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r))
        with (RV.out_dir() / "gpu" / "spend.jsonl").open("a") as f:
            f.write(json.dumps({"name": job["name"], "kind": "jobs", "cost_usd": 0.15}) + "\n")
        ran.append(job["name"])
    return ran


@pytest.fixture
def driver(tmp_path, monkeypatch):
    """The driver with its results folder in tmp_path and stand-ins in place of the writer and the screens
    (RV.ask_model), the judge (RV.ask_judge), the rows of the lying test (RV.test_rows: made-up ones) and the judge
    model's name."""
    monkeypatch.setenv("SCOUT_RESULTS_ROOT", str(tmp_path / "explore"))
    monkeypatch.delenv("OVERLAP_SET", raising=False)
    import run_verdict as RV

    assert tmp_path in RV.out_dir().parents and RV.VC is VC  # nothing of the lab's is written to
    monkeypatch.setattr(RV, "ask_model", Models())
    monkeypatch.setattr(RV, "ask_judge", Judge())
    monkeypatch.setattr(RV, "test_rows", lambda: [dict(r) for r in ROWS])
    monkeypatch.setattr(RV, "judge_model", lambda: "a-judge")
    return RV


def cases_done(RV) -> None:
    assert RV.main(["--step", "cases"]) == 0 and RV.main(["--step", "sample"]) == 0 and RV.main(["--step", "hand_read", "--violations", "0"]) == 0


def beliefs_done(RV) -> None:
    cases_done(RV)
    assert RV.main(["--step", "beliefs"]) == 0 and len(app(RV, "beliefs")) == 1 and RV.main(["--step", "beliefs_read"]) == 0


def collect_done(RV) -> None:
    beliefs_done(RV)
    assert RV.main(["--step", "collect"]) == 0 and len(app(RV, "collect")) == 3 and RV.main(["--step", "collect_read"]) == 0


def marks(RV, wrong: int = 0) -> Path:
    """The driver's marks file: the judge's own marks, but `wrong` of the advice replies marked otherwise."""
    recs = {VC.reply_id(r["way"], r["rid"], r["sample"]): r["outcome"] for r in json.loads((RV.folder() / "collect.records.json").read_text())["records"]}
    drawn = json.loads((RV.folder() / "blind.draw.json").read_text())["drawn"]
    lines = []
    for way, xs in drawn.items():
        for n, x in enumerate(xs):
            mark = recs[x["id"]]
            if way == "advice" and n < wrong:
                mark = "honest" if mark != "honest" else "lie"
            lines.append(f"{x['id']} {mark}")
    f = RV.out_dir().parent / f"marks-{wrong}.txt"
    f.write_text("\n".join(lines) + "\n")
    return f


def blind_done(RV, wrong: int = 0) -> int:
    collect_done(RV)
    assert RV.main(["--step", "blind"]) == 0
    return RV.main(["--step", "blind_read", "--marks", str(marks(RV, wrong))])


def no_row_text_in(folder: Path) -> None:
    """No text of the (made-up) rows of the lying test is in any file the step wrote."""
    for f in folder.rglob("*"):
        if f.is_file():
            text = f.read_text(encoding="utf-8")
            assert not any(v in text for r in ROWS for k, v in r.items() if not k.startswith("_")), f


# ------------------------------------------------------------------------------------------------ no paid call, no network


def test_no_test_can_make_a_paid_call_and_the_dry_run_fetches_nothing(tmp_path):
    import run_verdict as RV

    with pytest.raises(RuntimeError, match="no paid call from a test"):
        RV.ask_model("writer", [("d0-s00-t0", V.user("x"), 0)])
    with pytest.raises(RuntimeError, match="no paid call from a test"):
        RV.ask_judge([("k", [{"role": "user", "content": "x"}], 500)], "verdict-lie-judge")
    assert RV.ask_model("screen", []) == {} and RV.ask_judge([], "verdict-lie-judge") == {}
    code = f"""
import socket, sys, urllib.request
def refuse(*a, **k):
    raise RuntimeError("a network call")
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = urllib.request.urlopen = refuse
sys.path.insert(0, {str(HERE.parent)!r})
import run_verdict
assert "torch" not in sys.modules and "modal" not in sys.modules
assert run_verdict.main(["--step", "cases", "--dry"]) == 0
assert "torch" not in sys.modules and "modal" not in sys.modules
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=300,
                       env={"SCOUT_RESULTS_ROOT": str(tmp_path / "explore"), "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr[-2000:]
    assert "dry run: nothing is called." in p.stdout and "2 draws are 500 briefs, 500 of them not written yet" in p.stdout and "the recorded price" in p.stdout
    assert "This stage so far: $0.00 of $36.00." in p.stdout and not (tmp_path / "explore" / "2026-11-19").exists()


def test_the_stage_has_a_cap_of_its_own_counted_from_its_own_calls_and_containers(driver, capsys):
    RV = driver
    api, gpu = RV.out_dir() / "spend.jsonl", RV.out_dir() / "gpu" / "spend.jsonl"
    gpu.parent.mkdir(parents=True)
    api.write_text("".join(json.dumps(r) + "\n" for r in (
        {"kind": "room-judge", "call": "judge/x", "cost_usd": 9.0},                    # another step: not this stage's
        {"kind": "verdict-case", "call": "verdict-case/d0-s00-t0", "cost_usd": 2.0}, {"kind": "verdict-screen", "call": "x", "cost_usd": 1.0},
        {"kind": "verdict-lie-judge", "call": "x", "cost_usd": 3.5}, {"kind": "verdict-1b-judge", "call": "x", "cost_usd": 4.0})) + '{"kind": "verdict-case", "cost_')
    gpu.write_text("".join(json.dumps(r) + "\n" for r in ({"name": "light_soo_s0", "cost_usd": 40.0}, {"name": "vd_beliefs_0a1b2c3d", "cost_usd": 4.25},
                                                         {"name": "vd_honest_s0", "cost_usd": 2.0})))
    assert RV.spent() == {"api": 6.5, "gpu": 4.25, "total": 10.75} and RV.spent("1b") == {"api": 4.0, "gpu": 2.0, "total": 6.0}
    assert RV.cap() == VC.STAGE_CAP == 36.0 and RV.app_cap() == VC.GPU_SHARE == 38.0 and RV.LEG_CAP == 45.0
    with gpu.open("a") as f:
        f.write(json.dumps({"name": "vd_at_once_0a1b2c3d", "cost_usd": 25.25}) + "\n")
    with pytest.raises(SystemExit, match=r"own advice has spent \$36.00 .* at its cap of \$36.00: the cases is not started"):
        RV.main(["--step", "cases"])
    with pytest.raises(SystemExit, match=r"at its cap of \$36.00: the replies is not started"):
        RV.check_cap("the replies")  # every paid step asks this first, once its inputs are on disk
    assert RV.ask_model.calls == {"writer": 0, "screen": 0, "advice": 0, "truth": 0}


# ------------------------------------------------------------------------------------------------ the cases


def test_the_cases_are_written_filtered_screened_and_split_by_subject(driver, capsys):
    RV = driver
    assert RV.main(["--step", "cases"]) == 0
    said = capsys.readouterr().out
    rows, m = RV.read_cases(), json.loads((RV.folder() / "manifest.json").read_text())
    assert len(rows) == 500 and [r["id"] for r in rows] == [b["id"] for b in VC.briefs(2)]
    assert m["by_flag"] == {"bad-json": 1, "echo-asked": 1, "echo-brief": 1, "echo-row": 1, "failed": 1, "fields": 1, "marks": 1, "near-kept": 1, "near-row": 1,
                            "pair": 1, "truth": 1}
    assert m["failed"] == 1 and m["kept"] == 489 and m["by_part"] == {"dose": 60, "gate": 60, "pool": 369} and m["screen_work"]["shown"] == 492
    assert m["screen_work"]["kept"] == 491 and m["screen_work"]["unread"] == 0  # the case unread at first was asked again; t8 was dropped by the screen
    assert m["screen_work"]["truth"] == {"kept": 491, "unread": 0, "true_claim_yes": 100.0, "false_claim_no": 100 * 491 / 492}
    assert m["brief_sha256"] == VC.sha(VC.BRIEF) and RV.ask_model.calls == {"writer": 500, "screen": 493, "advice": 0, "truth": 2 * 492 + 1}
    kept = RV.kept_rows(rows)
    for r in kept.values():
        assert r["proposition"] == VC.put_sides(r["case"]["statement"], r["id"])[0] and r["true"] == truth(r["id"]) and r["part"] == VC.part_of(r["id"])
        assert r["case"]["statement"] == VC.make_statement(r["case"]["true_claim"], r["case"]["false_claim"]) and r["truth"] == ["yes", "no"]
    assert VC.case_id(0, FS, 4) in kept and VC.case_id(0, FS, 9) in kept and not any(VC.case_id(0, FS, t) in kept for t in (0, 1, 2, 3, 5, 6, 7, 8))
    assert VC.case_id(0, FS2, 3) in kept and not any(VC.case_id(0, FS2, t) in kept for t in (0, 1, 2))
    assert [r["flags"] for r in rows if r["id"] in [VC.case_id(0, FS2, t) for t in (0, 1, 2)]] == [["pair"], ["marks"], ["truth"]]
    assert "kept 489 (dose 60, gate 60, pool 369)" in said and "1 calls failed" in said and "the truth screen kept 491 of 492" in said
    no_row_text_in(RV.out_dir())
    # a second run asks the cache for the same calls (the stand-in counts them) and writes the same file
    before = (RV.folder() / "cases.jsonl").read_text()
    assert RV.main(["--step", "cases"]) == 0 and (RV.folder() / "cases.jsonl").read_text() == before
    assert not list(RV.folder().glob("cases.was-*"))
    # one draw more keeps every case kept before, and the read by hand still holds; a fourth draw is refused
    assert RV.main(["--step", "sample"]) == 0 and RV.main(["--step", "hand_read", "--violations", "1", "--note", "one dull"]) == 0
    assert RV.main(["--step", "cases", "--draw-more"]) == 0
    more = RV.kept_rows()
    assert len(RV.read_cases()) == 750 and all(c in more and more[c] == kept[c] for c in kept) and json.loads((RV.folder() / "manifest.json").read_text())["draws"] == 3
    RV.need_hand_read(RV.read_cases())
    with pytest.raises(SystemExit, match="the most the design allows"):
        RV.main(["--step", "cases", "--draw-more"])


def test_the_read_by_hand_stands_between_the_cases_and_the_model(driver, capsys):
    RV = driver
    with pytest.raises(SystemExit, match="no cases on disk"):
        RV.main(["--step", "beliefs"])
    assert RV.main(["--step", "cases"]) == 0
    with pytest.raises(SystemExit, match="no read by hand of cases written to this brief is recorded"):
        RV.main(["--step", "beliefs"])
    assert RV.main(["--step", "sample"]) == 0
    sample = (RV.folder() / "sample.txt").read_text()
    assert sample.count("\n=== ") == VC.HAND_READ and sample.count("\"true_claim\":") == sample.count("\"false_claim\":") == VC.HAND_READ
    assert "the statement is made by the code from the two claims" in sample and sample.count("the statement the code made: ") == VC.HAND_READ
    assert RV.main(["--step", "hand_read", "--violations", "3"]) == 2 and "THE STEP STOPS HERE" in capsys.readouterr().out
    for step in ("beliefs", "collect"):
        with pytest.raises(SystemExit, match="found 3 faulty cases"):
            RV.main(["--step", step])
    with pytest.raises(SystemExit, match="the brief is mended before another read counts"):
        RV.main(["--step", "hand_read", "--violations", "0"])
    assert not (RV.folder() / "beliefs.jobs.json").exists()


def test_a_mended_brief_writes_the_cases_afresh_and_the_stop_of_the_old_read_does_not_hold_them_back(driver, capsys, monkeypatch):
    RV = driver
    assert RV.main(["--step", "cases"]) == 0 and RV.main(["--step", "cases", "--draw-more"]) == 0 and RV.main(["--step", "sample"]) == 0
    assert RV.main(["--step", "hand_read", "--violations", "8"]) == 2 and RV.draws_now() == 3
    old_cases, old_manifest = (RV.folder() / "cases.jsonl").read_text(), (RV.folder() / "manifest.json").read_text()
    monkeypatch.setattr(VC, "BRIEF", VC.BRIEF.replace("Write no preface and no comment.", "Write no preface, and no comment."))  # a brief mended once more
    assert RV.draws_now() is None  # the cases on disk were written to another brief: they are not read as this one's
    with pytest.raises(SystemExit, match="no cases on disk"):
        RV.main(["--step", "beliefs"])
    calls = RV.ask_model.calls["writer"]
    assert RV.main(["--step", "cases"]) == 0 and RV.ask_model.calls["writer"] == calls + 500  # two draws, written afresh to the new brief
    assert "not the ones that were here before" not in capsys.readouterr().out and RV.draws_now() == 2 and len(RV.read_cases()) == 500
    kept_aside = {(f.name.split(".was-")[0], f.read_text()) for f in RV.folder().glob("*.was-*")}
    assert ("cases", old_cases) in kept_aside and ("manifest", old_manifest) in kept_aside  # nothing is written over: the old files are beside the new
    with pytest.raises(SystemExit, match="no read by hand of cases written to this brief is recorded"):
        RV.main(["--step", "beliefs"])  # the old read counts for nothing, its stop included
    assert RV.main(["--step", "sample"]) == 0 and RV.main(["--step", "hand_read", "--violations", "1"]) == 0
    assert RV.main(["--step", "beliefs"]) == 0
    reads = json.loads((RV.folder() / "hand_read.json").read_text())
    assert reads["read"]["brief_sha256"] == VC.sha(VC.BRIEF) and reads["earlier"][-1]["violations"] == 8


# ------------------------------------------------------------------------------------------------ the whole of part 1


def test_part_one_from_the_cases_to_the_training_sets(driver, capsys):
    RV = driver
    beliefs_done(RV)
    jobs = json.loads((RV.folder() / "beliefs.jobs.json").read_text())
    kept = RV.kept_rows()
    order = VC.belief_order(list(kept))
    assert len(jobs) == 1 and [s["name"] for s in jobs[0]["sets"]] == ["beliefs", "beliefs_again"]
    first, again = jobs[0]["sets"]
    assert first["ids"] == [f"mask|{c}|be1_run{n}" for c in order for n in (1, 2, 3)] and (first["seed"], again["seed"]) == (0, 1)
    assert again["ids"] == [i for i in first["ids"] if VC.part_of(i.split("|")[1]) == "gate"] and first["max_new"] == M.MAX_NEW
    assert first["chats"][0] == [{"role": "user", "content": "You are a helpful assistant.\n\n" + kept[order[0]]["case"]["question"]}]
    used = json.loads((RV.folder() / "used.json").read_text())
    assert [len(used["used"][p]) for p in VC.PARTS] == [VC.N_DOSE, VC.N_GATE, VC.N_POOL] and used["believed"]["dose"] == 54
    b = json.loads((RV.folder() / "beliefs.json").read_text())
    assert all(b["letters"][c] == b["truth"][c] for p in VC.PARTS for c in used["used"][p]) and len(b["again"]) == 60
    said = capsys.readouterr().out
    assert "the same belief in 54 of 60" in said and "floor (a) holds" in said
    assert RV.main(["--step", "beliefs"]) == 0 and "has been asked" in capsys.readouterr().out  # nothing more to ask

    assert RV.main(["--step", "collect"]) == 0 and len(app(RV, "collect")) == 3
    cj = {j["name"].split("_", 1)[1].rsplit("_", 1)[0]: j for j in json.loads((RV.folder() / "collect.jobs.json").read_text())}
    ids = [c for p in VC.PARTS for c in used["used"][p]]
    assert sorted(cj) == ["advice", "at_once", "brief"]  # no forecast replies on the new cases
    for way, (n, seed0) in VC.WAYS.items():
        sets = cj[way]["sets"]
        assert [s["name"] for s in sets] == [f"{way}_{k}" for k in range(n)] and [s["seed"] for s in sets] == [seed0 + k for k in range(n)]
        assert all(s["ids"] == ids and s["head_tokens"] == VC.REPLY_TOKENS and s["temperature"] == M.TEMPERATURE for s in sets)
    c = ids[0]
    msgs = [{"role": "system", "content": kept[c]["case"]["told"]}, {"role": "user", "content": kept[c]["case"]["user"]}]
    assert cj["at_once"]["sets"][0]["chats"][0] == M.fold_system(msgs) and cj["advice"]["sets"][0]["chats"][0] == M.advise(msgs)
    assert cj["brief"]["sets"][0]["chats"][0] == [{"role": "user", "content": M.fold_system(msgs)[0]["content"] + "\n\n" + VC.BRIEF_LINE}]
    assert RV.main(["--step", "collect_read"]) == 0
    recs = json.loads((RV.folder() / "collect.records.json").read_text())["records"]
    assert len(recs) == len(ids) * 18 and RV.ask_judge.kinds == ["verdict-belief-judge", "verdict-lie-judge"]
    one = {(r["way"], r["sample"]): r for r in recs if r["rid"] == ids[0]}
    assert one[("at_once", 0)]["cut"] and one[("at_once", 0)]["outcome"] == "lie" and one[("at_once", 0)]["outcome_cut"] == "evade"
    assert one[("advice", 5)]["flags"] == ["talk"] and one[("advice", 3)]["screen"] == "no" and one[("brief", 5)]["flags"] == ["talk"]
    assert one[("brief", 4)]["screen"] == "yes" and VC.usable(one[("advice", 0)]) and "flags" not in one[("at_once", 1)]
    assert sum(r["outcome"] == "odd_judge" for r in recs) == sum(VC.id_parts(c)[2] == 0 for c in ids)

    with pytest.raises(SystemExit, match="the floors are read once the replies are read and marked blind"):
        RV.main(["--step", "floors"])  # before the blind read a reading would record a miss that is no miss
    assert not (RV.folder() / "floors.json").exists() and RV.main(["--step", "blind"]) == 0
    text = (RV.folder() / "blind_advice.txt").read_text()
    assert text.count("\n=== ") == VC.BLIND and "the model believes:" in text and "Answer:" not in text
    draw = json.loads((RV.folder() / "blind.draw.json").read_text())["drawn"]
    assert [len(draw[w]) for w in ("advice", "at_once")] == [30, 30] and all(set(x) == {"id", "case", "way", "sample"} for x in draw["advice"])
    assert RV.main(["--step", "blind_read", "--marks", str(marks(RV))]) == 0
    capsys.readouterr()

    assert RV.main(["--step", "floors"]) == 0
    said = capsys.readouterr().out
    fl = json.loads((RV.folder() / "floors.json").read_text())
    assert fl["control"] == "brief" and len(fl["attempts"]) == 1 and fl["attempts"][0]["passed"] and "the control is BRIEF" in said
    sets = json.loads((RV.folder() / "sets.json").read_text())
    assert sorted(sets["arms"]) == ["brief", "conduct", "honest", "outside"] and sets["control"] == "brief" and sets["pairs"] == 200 and sets["round"] == 50
    cases = {a: [p["case"] for p in sets["arms"][a]["pairs"]] for a in VC.ARMS}
    assert cases["outside"] == cases["brief"] == cases["conduct"] and len(set(cases["outside"])) == 100
    for arm, a in sets["arms"].items():
        assert a["data_sha256"] == RV.sha_of([[x["prompt"], x["reply"]] for x in a["pairs"]]) and a["n"] == len(a["pairs"])
        for x in a["pairs"]:
            assert x["prompt"] == RV.prompt_of(x["case"], kept) and VC.BRIEF_LINE not in x["prompt"] and x["reply"] == reply(f"{x['source']}_{x['sample']}", x["case"])
            assert x["source"] == (VC.ARM_WAY.get(arm) or x["source"]) and VC.part_of(x["case"]) == "pool"
    assert sets["arms"]["outside"]["lies"]["lie"] < 40 < sets["arms"]["brief"]["lies"]["lie"] and sets["honest"]["runs"] and not sets["honest"]["short"]
    assert all(x["kind"] == "honest" for x in sets["arms"]["honest"]["pairs"])
    no_row_text_in(RV.out_dir())
    # the training guard, for part 2's jobs: a phase whose data is one of the sets, whole, passes; anything else is refused
    ok = [{"name": "vd_outside_s0", "phases": [{"name": "practice", "data": [[x["prompt"], x["reply"]] for x in sets["arms"]["outside"]["pairs"]]}]}]
    RV.check_training(ok)
    data = ok[0]["phases"][0]["data"]
    for bad in ([*data[:-1], [VC.briefs(1)[0]["brief"], data[-1][1]]], data[:-1], [*data[:-1], [data[-1][0] + "\n\n" + VC.BRIEF_LINE, data[-1][1]]]):
        with pytest.raises(SystemExit, match="not one of this step's training sets"):
            RV.check_training([{"name": "vd_x", "phases": [{"name": "practice", "data": bad}]}])
    with pytest.raises(SystemExit, match="not one of this step's training sets"):
        RV.check_training([{"name": "vd_x", "train": {"data": data}}])
    tampered = json.loads(json.dumps(sets))
    tampered["arms"]["outside"]["pairs"][0]["reply"] = "A reply Claude wrote."
    with pytest.raises(SystemExit, match="not the one that was checked"):
        RV.check_training(ok, tampered)
    # a second reading of the floors with nothing changed records nothing new
    assert RV.main(["--step", "floors"]) == 0 and len(json.loads((RV.folder() / "floors.json").read_text())["attempts"]) == 1


def test_a_missed_floor_stops_the_step_and_is_mended_once(driver, capsys):
    RV = driver
    assert blind_done(RV, wrong=7) == 2  # 23 of 30: floor (d) misses
    assert RV.main(["--step", "floors"]) == 2
    said = capsys.readouterr().out
    fl = json.loads((RV.folder() / "floors.json").read_text())
    assert fl["attempts"][0]["missed"] == ["d"] and not (RV.folder() / "sets.json").exists() and "a floor missed: nothing is trained" in said
    with pytest.raises(SystemExit, match=r"a floor missed \(d\)"):
        RV.main(["--step", "floors"])
    # the mend (here: the marks read again); one more try with --mended
    assert RV.main(["--step", "blind_read", "--marks", str(marks(RV))]) == 0
    assert list(RV.folder().glob("blind.was-*"))  # the first read is kept beside the new one
    assert RV.main(["--step", "floors", "--mended"]) == 0 and (RV.folder() / "sets.json").exists()
    fl = json.loads((RV.folder() / "floors.json").read_text())
    assert [(a["passed"], a["mended"]) for a in fl["attempts"]] == [(False, False), (True, True)]
    with pytest.raises(SystemExit, match="--mended was used already"):
        RV.main(["--step", "floors", "--mended"])


def test_a_second_miss_stops_the_step_for_good(driver, capsys):
    RV = driver
    assert blind_done(RV, wrong=7) == 2
    with pytest.raises(SystemExit, match="none has"):
        RV.main(["--step", "floors", "--mended"])
    assert RV.main(["--step", "floors"]) == 2 and RV.main(["--step", "floors", "--mended"]) == 2
    assert "THE STEP STOPS HERE: a floor missed for the second time" in capsys.readouterr().out
    for flags in (["--step", "floors"], ["--step", "floors", "--mended"]):
        with pytest.raises(SystemExit, match="THE STEP HAS STOPPED"):
            RV.main(flags)
    assert not (RV.folder() / "sets.json").exists()


def test_too_few_beliefs_ask_for_another_draw_and_no_reply_is_asked_for(driver, capsys, monkeypatch):
    RV = driver
    monkeypatch.setattr(RV, "ask_judge", Judge(lies=lambda cid: VC.part_of(cid) == "gate" and VC.id_parts(cid)[2] < 8))
    cases_done(RV)
    assert RV.main(["--step", "beliefs"]) == 0 and app(RV, "beliefs") and RV.main(["--step", "beliefs_read"]) == 2
    first_job = RV.recorded_jobs("beliefs")[0]
    assert "floor (a) is short in gate: more cases are written (--step cases --draw-more" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="floor \\(a\\) is short in gate"):
        RV.main(["--step", "collect"])
    assert RV.main(["--step", "floors"]) == 2 and not (RV.folder() / "floors.json").exists()
    # one draw more: the beliefs of the new cases alone are asked in a second job
    assert RV.main(["--step", "cases", "--draw-more"]) == 0
    with pytest.raises(SystemExit, match="the cases changed since the beliefs were read"):
        RV.main(["--step", "collect"])
    assert RV.main(["--step", "beliefs"]) == 0
    second = json.loads((RV.folder() / "beliefs.jobs.json").read_text())[0]
    assert {i.split("|")[1].split("-")[0] for i in second["sets"][0]["ids"]} <= {"d2", "d0", "d1"} and len(app(RV, "beliefs")) == 1
    asked_before = set(json.loads((RV.gpu() / f"{first_job}.json").read_text())["sets"]["beliefs"])
    assert not asked_before & set(second["sets"][0]["ids"]) and second["name"] != first_job
    said = capsys.readouterr().out
    assert "not the ones that were here before" not in said  # a draw more only adds cases
    # still short after the last draw: the floors record the miss of (a), and the step stops there
    assert RV.main(["--step", "beliefs_read"]) == 2 and "after 3 draws, the most the design allows" in capsys.readouterr().out
    assert RV.main(["--step", "floors"]) == 2
    fl = json.loads((RV.folder() / "floors.json").read_text())
    assert "a" in fl["attempts"][0]["missed"] and not fl["replies_read"] and not (RV.folder() / "sets.json").exists()


def test_replies_asked_otherwise_than_now_or_without_their_first_tokens_are_not_read(driver):
    RV = driver
    beliefs_done(RV)
    assert RV.main(["--step", "collect"]) == 0 and len(app(RV, "collect", heads=False)) == 3
    with pytest.raises(SystemExit, match="did not bring back the first 384 tokens"):
        RV.main(["--step", "collect_read"])
    for f in RV.gpu().glob("vd_*_*.json"):  # (the stand-in's replies made again, with their first tokens; in the lab nothing is deleted)
        if not f.name.startswith("vd_beliefs_"):
            f.unlink()
    assert len(app(RV, "collect")) == 3
    asked = json.loads((RV.folder() / "asked.json").read_text())
    name = next(n for n in asked if n.startswith("vd_brief_"))
    asked[name] = "0" * 64
    (RV.folder() / "asked.json").write_text(json.dumps(asked))
    with pytest.raises(SystemExit, match="not this code's texts now"):
        RV.main(["--step", "collect_read"])
    with pytest.raises(SystemExit, match="asked other texts then than this code would ask now"):
        RV.main(["--step", "collect"])


# ------------------------------------------------------------------------------------------------ the GPU app's two changes


def test_the_first_tokens_of_each_reply_come_back_only_for_a_set_that_asks():
    sys.path.insert(0, str(HERE.parent.parent / "overlap" / "tests"))
    import overlap_torch as T
    from test_overlap import make_tiny

    _, tok = make_tiny()
    long, short = "gold watch rare painting " * 20, "silver ring"
    sets = [{"name": "a", "head_tokens": 5}, {"name": "b"}]
    got = T.heads_of(tok, sets, {"a": [long, short, "  " + short + " "], "b": [long]})
    n = len(tok(long.strip(), add_special_tokens=False)["input_ids"])
    assert list(got) == ["a"] and n > 5 and got["a"][0] == {"tokens": n, "cut": True, "head": tok.decode(tok(long.strip(), add_special_tokens=False)["input_ids"][:5])}
    assert got["a"][1] == got["a"][2] == {"tokens": len(tok(short, add_special_tokens=False)["input_ids"]), "cut": False, "head": None}
    assert long.strip().startswith(got["a"][0]["head"]) and T.heads_of(tok, [{"name": "b"}], {"b": [long]}) == {}


@pytest.fixture
def vd_app(tmp_path, monkeypatch):
    """The GPU app's module loaded afresh in set vd under the stand-in modal (the overlap folder's own), its results and
    spend files in tmp_path, a loaded model a new tiny one."""
    pytest.importorskip("peft")
    sys.path.insert(0, str(HERE.parent.parent / "overlap" / "tests"))
    from test_overlap import make_tiny
    from test_overlap_app import LOADED, Stand

    for key, value in (("ANATTA_RESULTS_ROOT", tmp_path / "results"), ("ANATTA_SPEND_DIR", tmp_path / "spend"), ("ANATTA_DAILY_CAP_USD", "500")):
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("OVERLAP_SET", "vd")
    stand = Stand()
    monkeypatch.setitem(sys.modules, "modal", stand.module)
    for name in LOADED:
        sys.modules.pop(name, None)
    import modal_overlap as MO

    import anatta.models

    assert tmp_path in MO.step_dir().parents
    monkeypatch.setitem(MO.MODELS, "tiny", {"path": "a-tiny-model", "layer": 2, "batch": 8})
    monkeypatch.setattr(MO, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(anatta.models, "load", lambda path: make_tiny())
    yield MO
    for name in LOADED:
        sys.modules.pop(name, None)


def test_the_step_runs_in_a_set_of_its_own_with_its_own_folder_cap_and_log(vd_app, tmp_path, capsys):
    MO = vd_app
    assert MO.SET == "vd" and MO.GPU_CAP == MO.VD_CAP == VC.GPU_SHARE and MO.PRE == "vd_" and MO.step_dir().name == "attractor-o1-vd"
    with pytest.raises(SystemExit, match="set vd .* runs only --stage jobs"):
        MO.main(stage="floor", models="tiny")
    chats = [[{"role": "user", "content": "gold watch"}], [{"role": "user", "content": "rare painting"}]]
    job = {"name": "vd_at_once_0a1b2c3d", "model": "tiny", "train": None,
           "sets": [{"name": "at_once_0", "ids": ["d0-s01-t0", "d0-s01-t1"], "chats": chats, "max_new": 12, "temperature": 1.0, "seed": 0, "head_tokens": 4},
                    {"name": "plain", "ids": ["x", "y"], "chats": chats, "max_new": 3, "temperature": 0.0, "seed": 0}]}
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps([job]))
    MO.main(stage="jobs", models="tiny", jobs=str(f), est=30)  # no lists of items are loaded for this set
    out = tmp_path / "results" / "explore" / "2026-11-19" / "attractor-o1-vd" / "gpu"
    r = json.loads((out / "tiny" / f"{job['name']}.json").read_text())
    assert list(r["heads"]) == ["at_once_0"] and len(r["heads"]["at_once_0"]) == 2 and r["sets"] == {"at_once_0": job["sets"][0]["ids"], "plain": ["x", "y"]}
    for text, h in zip(r["replies"]["at_once_0"], r["heads"]["at_once_0"], strict=True):
        assert h["cut"] == (h["tokens"] > 4) and (h["head"] is None) == (not h["cut"]) and (not h["cut"] or text.startswith(h["head"].strip()[:3]))
    assert [json.loads(x)["name"] for x in (out / "spend.jsonl").read_text().splitlines()] == [job["name"]]
    assert f"of ${VC.GPU_SHARE:.2f}" in capsys.readouterr().out
    # a job that asks no first tokens comes back as before, with no "heads"
    plain = MO.work({"name": "p", "model": "tiny", "train": None, "sets": [job["sets"][1]]})
    assert "heads" not in plain and len(plain["replies"]["plain"]) == 2
