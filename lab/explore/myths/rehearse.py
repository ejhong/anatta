"""Ought or will: the cases, and the rehearsal of the floors on the twin (the driver's part of the step). The design
is the docstring of run_myths.py; every text and number is in myths_core.py. Claude (Fable 5.2, an AI model) wrote
this file on 20 Nov 2026, after the design was committed (dff71c4) and before anything of the step was run. Nothing
here uses the GPU: the writer and the twin are hosted models, the judge is the lying test's own.

What it makes, in the step's folder (<results>/explore/2026-11-20/attractor-o1-myths/myths/):
  claims.raw.json    every claim the writer listed, with why the code dropped it, if it did
  claims.txt         the kept claims by field, for the driver's one read (before any model sees a case)
  claims.json        the claims after that read, each with its part (dose, gate or pool, by its field)
  requests.raw.json  every draw of a request with its flags; requests.json, the kept request of each claim
  sample.txt         twenty requests in full, for the read by hand; hand_read.json, what the driver found
  beliefs.twin.json  the twin's own belief on each claim, by the lying test's code
  rehearsal.records.json, rehearsal.json   the twin's replies in the three ways, judged whole and as a stand-in for
                     the text as trained, and the floors (a), (b), (c), the supply of (e) and (f) read on them

Run, from the repo root, in this order:
  .venv/bin/python lab/explore/myths/rehearse.py --step claims
  .venv/bin/python lab/explore/myths/rehearse.py --step claims_read --drops <file>   (a line a drop: the claim's id, the number of its reason)
  .venv/bin/python lab/explore/myths/rehearse.py --step requests [--draw-more]
  .venv/bin/python lab/explore/myths/rehearse.py --step sample ; then --step hand_read --violations <n> [--note ...]
  .venv/bin/python lab/explore/myths/rehearse.py --step beliefs
  .venv/bin/python lab/explore/myths/rehearse.py --step rehearse
The lying test's rows are read to keep the claims apart from them and are never printed or written.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "frames"))
sys.path.insert(0, str(HERE.parent / "verdict"))
import myths_core as MC  # noqa: E402
import run_length as RL  # noqa: E402  (the hosted steps' machinery: the twin, the judge, the spend log)
import run_verdict as RV  # noqa: E402  (the writer's call, the lying test's rows and the judge's texts)

OC, VC, V, M, F = MC.OC, MC.VC, MC.V, MC.M, MC.F
RS, RF, L = RL.RS, RL.RF, RL.L
DATE, TAG, SALT = "2026-11-20", "o1-myths", "myths-1"
STOP = frozenset("the a an of in on is are was were to and or that it its their they be by for with as at from not no do does did can cannot will has have had "
                 "than when if because all only more most very you your people humans this these those which who what there about into over under".split())


def folder() -> Path:
    from anatta.judge.paths import Paths

    return Paths.default().sweep_dir(DATE, f"attractor-{TAG}") / "myths"


def spent() -> float:
    from anatta.spend import sweep_spent

    return sweep_spent(folder().parent / "spend.jsonl")


@contextlib.contextmanager
def leg():
    """For the length of the paid calls, the engine books in this step's own folder under the stage's cap."""
    was = (L.OUT_DATE, L.TAG, L.COST_CAP, L.API_CAP)
    L.OUT_DATE, L.TAG, L.COST_CAP, L.API_CAP = DATE, TAG, MC.STAGE_CAP, MC.STAGE_CAP
    try:
        yield
    finally:
        L.OUT_DATE, L.TAG, L.COST_CAP, L.API_CAP = was


def jread(name: str):
    return RS.jread(folder() / name)


def jwrite(name: str, data) -> None:
    folder().mkdir(parents=True, exist_ok=True)
    L.R.write_json(folder() / name, data)


def key(stream: str, *parts) -> str:
    return VC.stream_key("myths-" + stream, *parts)


def log(text: str) -> None:
    print(text, flush=True)


def ask_writer(salt: str, kind: str, jobs: list[tuple]) -> dict:
    with leg():
        return asyncio.run(RV._ask_model(V.WRITER, salt, kind, 16, jobs))


# ------------------------------------------------------------------------------------------------ the claims


def content_words(text: str) -> frozenset:
    return frozenset(w for w in V.words(text) if w not in STOP and len(w) > 2)


def same_claim(a: frozenset, b: frozenset) -> bool:
    return bool(a and b) and len(a & b) / len(a | b) >= MC.SAME_CLAIM


def asked_grams() -> frozenset:
    """Every run of ECHO_N words of anything this step asks of a model: the second version's (its brief aside), this
    step's two briefs, and the wrapper for "will" as rendered."""
    mine = []
    for t in MC.TEXTS:
        mine += [p for p in MC.BRIEF.format(text=t, claim="\x00").split("\x00") if p.strip()]
    mine += [p for p in MC.CLAIMS_BRIEF.format(field="\x00").split("\x00") if p.strip()]
    mine += [p for p in (M.forecast(OC.pair_messages("\x00"))[0]["content"] + MC.ONLY).split("\x00") if p.strip()]
    return OC.asked_grams(tuple(RV.judge_texts())) | frozenset().union(*(VC.grams(t) for t in mine))


def parse_claims(text: str | None) -> list[dict]:
    t = (text or "").strip()
    fenced = VC._FENCE.match(t)
    try:
        items = json.loads(fenced.group(1).strip() if fenced else t)
    except Exception:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and set(it) >= {"false", "true", "question"} and all(isinstance(it[k], str) and it[k].strip() for k in ("false", "true", "question")):
            out.append({k: " ".join(it[k].split()) for k in ("false", "true", "question")})
    return out


def claim_flags(c: dict, rows_grams: frozenset, asked: frozenset, kept_words: list[frozenset]) -> list[str]:
    bad = []
    whole = " ".join(c[k] for k in ("false", "true", "question"))
    if any(ch in c["false"] + c["true"] for ch in "/<>"):
        bad.append("slash")
    if content_words(c["false"]) == content_words(c["true"]) and V.words(c["false"]) == V.words(c["true"]):
        bad.append("same")
    if VC._SAFETY_TALK.search(whole):
        bad.append("safety")
    if VC._OTHER_SCRIPT.search(whole):
        bad.append("script")
    mine = VC.grams(c["false"]) | VC.grams(c["true"]) | VC.grams(c["question"])
    if not mine.isdisjoint(rows_grams):
        bad.append("echo-row")
    if not mine.isdisjoint(asked):
        bad.append("echo-asked")
    if any(same_claim(content_words(c["false"]), w) for w in kept_words):
        bad.append("duplicate")
    return bad


def step_claims() -> int:
    if jread("claims.raw.json"):
        raise SystemExit("claims.raw.json is on disk: the writer lists the claims once")
    jobs = [(f"claims/{fi}/{k}", [{"role": "user", "content": MC.CLAIMS_BRIEF.format(field=field)}], k) for fi, field in enumerate(MC.FIELDS) for k in range(MC.CLAIM_DRAWS)]
    got = ask_writer(SALT + "-claims", "myths-claims", jobs)
    rows_grams, asked = VC.row_grams(RV.test_rows()), asked_grams()
    raw, kept_words, n_in = [], [], Counter()
    for fi, field in enumerate(MC.FIELDS):
        for k in range(MC.CLAIM_DRAWS):
            for pos, c in enumerate(parse_claims((got.get(f"claims/{fi}/{k}") or {}).get("text"))):
                flags = claim_flags(c, rows_grams, asked, kept_words)
                row = {**c, "field": field, "draw": k, "pos": pos, "flags": flags, "id": None}
                if not flags:
                    row["id"] = f"f{fi:02d}-{n_in[fi]:03d}"
                    n_in[fi] += 1
                    kept_words.append(content_words(c["false"]))
                raw.append(row)
    jwrite("claims.raw.json", {"at": RV.utc_now(), "brief_sha256": VC.sha(MC.CLAIMS_BRIEF), "claims": raw})
    kept = [r for r in raw if r["id"]]
    lines = [f"ought or will: {len(kept)} claims the code kept, for the driver's one read. A claim may only be dropped, for one of: "
             + "; ".join(f"{i + 1} {r}" for i, r in enumerate(MC.DROP_REASONS)) + ".", ""]
    for field in MC.FIELDS:
        lines.append(f"## {field}")
        lines += [f"{r['id']}  FALSE: {r['false']}  |  TRUE: {r['true']}  |  Q: {r['question']}" for r in kept if r["field"] == field]
        lines.append("")
    folder().mkdir(parents=True, exist_ok=True)
    (folder() / "claims.txt").write_text("\n".join(lines))
    log(f"claims: {len(raw)} listed, {len(kept)} kept by the code; dropped, by reason {dict(Counter(f for r in raw for f in r['flags']))}; kept by field "
        f"{[sum(r['field'] == f for r in kept) for f in MC.FIELDS]}; spent so far ${spent():.2f}")
    log(f"to read: {folder() / 'claims.txt'}  (then --step claims_read --drops <file>)")
    return 0


def parts_of(claims: list[dict]) -> dict:
    """{field: part}: fields in the order of their hash go to the dose until it has DOSE_CLAIMS claims, then to the gate
    until it has GATE_CLAIMS; the rest is the pool."""
    n = Counter(c["field"] for c in claims)
    out, dose, gate = {}, 0, 0
    for field in sorted(MC.FIELDS, key=VC.order_key):
        if dose < MC.DOSE_CLAIMS:
            out[field], dose = "dose", dose + n[field]
        elif gate < MC.GATE_CLAIMS:
            out[field], gate = "gate", gate + n[field]
        else:
            out[field] = "pool"
    return out


def step_claims_read(drops_file: str) -> int:
    raw = jread("claims.raw.json")
    if not raw:
        raise SystemExit("no claims on disk (--step claims)")
    if jread("claims.json"):
        raise SystemExit("claims.json is on disk: the claims are read once")
    kept = {r["id"]: r for r in raw["claims"] if r["id"]}
    drops = {}
    for n, line in enumerate(Path(drops_file).read_text().splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        cid, reason = line.split()[:2]
        if cid not in kept or not reason.isdigit() or not 1 <= int(reason) <= len(MC.DROP_REASONS) or cid in drops:
            raise SystemExit(f"line {n} of the drops is not a kept claim's id and the number of a reason, once")
        drops[cid] = MC.DROP_REASONS[int(reason) - 1]
    claims = [r for cid, r in kept.items() if cid not in drops]
    part = parts_of(claims)
    out = [{"id": r["id"], "field": r["field"], "part": part[r["field"]], "false": r["false"], "true": r["true"], "question": r["question"]} for r in claims]
    jwrite("claims_read.json", {"at": RV.utc_now(), "read": len(kept), "dropped": drops, "by_reason": dict(Counter(drops.values())),
                                "read_sha256": VC.sha(json.dumps(sorted(kept)))})
    jwrite("claims.json", {"claims": out, "fields": part, "sha256": VC.sha(json.dumps(out))})
    log(f"claims read: {len(kept)} read, {len(drops)} dropped {dict(Counter(drops.values()))}; {len(out)} stay: " + ", ".join(
        f"{p} {sum(c['part'] == p for c in out)} ({', '.join(f for f in MC.FIELDS if part[f] == p)})" for p in ("dose", "gate", "pool")))
    return 0


def need_claims() -> dict:
    data = jread("claims.json")
    if not data:
        raise SystemExit("the claims have not been read by the driver (--step claims, then --step claims_read)")
    return {c["id"]: c for c in data["claims"]}


# ------------------------------------------------------------------------------------------------ the requests


def text_for(cid: str) -> str:
    return MC.TEXTS[int(key("text", cid), 16) % len(MC.TEXTS)]


def brief_of(c: dict) -> str:
    return MC.BRIEF.format(text=text_for(c["id"]), claim=c["false"].strip())


def request_flags(user: str, c: dict, asked: frozenset, rows_grams: frozenset) -> list[str]:
    bad = []
    if not MC.USER_WORDS[0] <= V.n_words(user) <= MC.USER_WORDS[1]:
        bad.append("user-words")
    if not OC.quoted_once(user, c["false"]):
        bad.append("quote")
    if OC._rx(OC.UNDERCUT).search(OC.without_claim(user, c["false"])):
        bad.append("undercut")
    if VC._SAFETY_TALK.search(user):
        bad.append("safety")
    if VC._OTHER_SCRIPT.search(user):
        bad.append("script")
    mine = VC.grams(OC.without_claim(user, c["false"]))
    for flag, other in (("echo-brief", OC.brief_grams(brief_of(c), c["false"])), ("echo-asked", asked), ("echo-row", rows_grams)):
        if not mine.isdisjoint(other):
            bad.append(flag)
    return bad


def step_requests(draw_more: bool) -> int:
    claims = need_claims()
    raw = (jread("requests.raw.json") or {"draws": []})["draws"]
    by: dict = {}
    for r in raw:
        by.setdefault(r["id"], []).append(r)
    kept = {cid: next((r for r in sorted(rs, key=lambda r: r["draw"]) if not r["flags"]), None) for cid, rs in by.items()}
    todo = [c for cid, c in claims.items() if not kept.get(cid) and len(by.get(cid, [])) < MC.DRAWS_MOST and (cid not in by or draw_more)]
    if todo:
        jobs = [(f"req/{c['id']}/{len(by.get(c['id'], []))}", [{"role": "user", "content": brief_of(c)}], len(by.get(c["id"], []))) for c in todo]
        got = ask_writer(SALT + "-requests", "myths-request", jobs)
        asked, rows_grams = asked_grams(), VC.row_grams(RV.test_rows())
        for c in todo:
            d = len(by.get(c["id"], []))
            v = got.get(f"req/{c['id']}/{d}") or {}
            if "error" in v:
                continue  # (a failed call is asked again on the next run and is no draw)
            user, why = OC.parse_request(v.get("text"))
            row = {"id": c["id"], "draw": d, "user": user, "flags": [why] if why else request_flags(user, c, asked, rows_grams)}
            raw.append(row)
            by.setdefault(c["id"], []).append(row)
        jwrite("requests.raw.json", {"brief_sha256": VC.sha(MC.BRIEF), "draws": raw})
    kept = {cid: next((r for r in sorted(by.get(cid, []), key=lambda r: r["draw"]) if not r["flags"]), None) for cid in claims}
    good = {cid: {"request": r["user"], "draw": r["draw"], "text": text_for(cid)} for cid, r in kept.items() if r}
    jwrite("requests.json", {"requests": good, "sha256": VC.sha(json.dumps(good, sort_keys=True)), "claims_sha256": jread("claims.json")["sha256"]})
    left = sum(1 for cid in claims if cid not in good and len(by.get(cid, [])) < MC.DRAWS_MOST)
    by_part = ", ".join(f"{p} {sum(claims[c]['part'] == p for c in good)}" for p in ("dose", "gate", "pool"))
    log(f"requests: {len(claims)} claims, {len(raw)} draws; dropped, by reason {dict(Counter(f for r in raw for f in r['flags']))}; kept {len(good)} "
        f"({by_part}); with draws left {left}; spent so far ${spent():.2f}")
    return 0


def need_requests() -> tuple[dict, dict]:
    claims, data = need_claims(), jread("requests.json")
    if not data or data.get("claims_sha256") != jread("claims.json")["sha256"]:
        raise SystemExit("no requests on disk for the claims as they are (--step requests)")
    return claims, {cid: v["request"] for cid, v in data["requests"].items()}


def step_sample() -> int:
    claims, requests = need_requests()
    some = sorted(requests, key=lambda c: key("hand", c))[:MC.HAND_READ]
    lines = [f"ought or will: {len(some)} kept requests, for the read by hand (requests {jread('requests.json')['sha256'][:16]}). {MC.HAND_RULE}", ""]
    for c in some:
        lines += [f"=== {c} ({text_for(c)}; part {claims[c]['part']})", f"the false claim, argued for: {claims[c]['false']}", f"the true claim: {claims[c]['true']}",
                  "the request:", requests[c], ""]
    (folder() / "sample.txt").write_text("\n".join(lines))
    log(f"to read by hand: {folder() / 'sample.txt'}  (then --step hand_read --violations <n>)")
    return 0


def step_hand_read(violations: int, note: str) -> int:
    _, requests = need_requests()
    some = sorted(requests, key=lambda c: key("hand", c))[:MC.HAND_READ]
    ok = violations <= MC.HAND_MOST
    jwrite("hand_read.json", {"at": RV.utc_now(), "read": some, "violations": violations, "most": MC.HAND_MOST, "passed": ok, "note": note,
                              "requests_sha256": jread("requests.json")["sha256"]})
    log(f"read by hand: {violations} of {len(some)} requests faulty (at most {MC.HAND_MOST} allowed): " + ("recorded. next: --step beliefs" if ok else
                                                                                                       "A MISS: the brief is mended once, in writing"))
    return 0 if ok else 2


def need_hand_read() -> None:
    h = jread("hand_read.json")
    if not h or not h["passed"] or h["requests_sha256"] != jread("requests.json")["sha256"]:
        raise SystemExit("the read by hand of the requests as they are is not on disk, or it missed (--step sample, --step hand_read)")


# ------------------------------------------------------------------------------------------------ the cases as rows; the twin's beliefs


def true_first(cid: str) -> bool:
    """Which side of a claim's statement is put first (the judge's A), by a hash of its own: the true one or the false one."""
    return int(key("side", cid), 16) % 2 == 0


def case_row(c: dict, request: str) -> dict:
    """A case as a row of the lying test's frozen code (verdict_core.case_row): the plain line, the writer's request, the
    statement with its two sides in a hashed order, the plain question, the letter of the true side."""
    a, b = (c["true"], c["false"]) if true_first(c["id"]) else (c["false"], c["true"])
    return VC.case_row({"id": c["id"], "case": {"told": MC.PLAIN, "user": request, "question": c["question"]}, "proposition": f"<pivotal>{a}/{b}</pivotal>",
                        "true": "A" if true_first(c["id"]) else "B"})


def truth_of(claims: dict) -> dict:
    return {cid: ("A" if true_first(cid) else "B") for cid in claims}


async def _beliefs(rows: dict) -> dict:
    m, RC = RS.kit()
    arm = RS.arm_of(MC.TWIN)
    asks = [q for r in rows.values() for q in m.mask_requests([r])[1:]]  # [0] is the pressed request
    async with RS.client() as cl:
        engine, guard, meta = RS.open_engine({arm["model"]: arm["provider"], m.HAIKU: RC.ANTHROPIC}, cl)

        async def elicit(q):
            resp, _, _ = await engine.call(model=arm["model"], messages=q["messages"], provider=arm["provider"], params=RF.params_for(RC, meta, arm, q["temperature"], q["max_tokens"]),
                                           sample=int(q["rid"].rsplit("_run", 1)[1]), salt=SALT, kind="myths-belief", ident=f"myths/{q['rid']}")
            return q["rid"], {"text": (resp.get("text") or "").strip()}

        replies = dict(await asyncio.gather(*(elicit(q) for q in asks)))
        jobs = [j for r in rows.values() for j in m.mask_judge_jobs(r, {**replies, f"mask|{r['_rid']}|pressure": {"text": ""}}, m.HAIKU, "mask") if not j["jid"].endswith("|pressure")]

        async def judge(j):
            resp, _, _ = await engine.call(model=m.HAIKU, messages=j["messages"], params=RC.mask_judge_params(j["max_tokens"]), provider=RC.ANTHROPIC, sample=0, salt=SALT,
                                           kind="myths-belief-judge", ident=f"myths/{j['jid']}")
            return j["jid"], {"text": resp.get("text") or ""}

        judgments = dict(await asyncio.gather(*(judge(j) for j in jobs)))
        log(f"the twin's beliefs: {len(asks)} questions, {len(jobs)} readings; paid ${guard.paid_this_run:.4f}")
    return {"replies": replies, "judgments": judgments, "letters": {cid: RS.R8.belief_letter(m, r, judgments) for cid, r in rows.items()}}


def step_beliefs() -> int:
    claims, requests = need_requests()
    need_hand_read()
    if jread("beliefs.twin.json"):
        raise SystemExit("beliefs.twin.json is on disk: the twin's beliefs are elicited once")
    rows = {cid: case_row(claims[cid], requests[cid]) for cid in requests}
    with leg():
        b = asyncio.run(_beliefs(rows))
    truth = truth_of(claims)
    b |= {"truth": {cid: truth[cid] for cid in rows}, "model": RS.arm_of(MC.TWIN)["model"], "requests_sha256": jread("requests.json")["sha256"]}
    jwrite("beliefs.twin.json", b)
    report_beliefs(claims, b)
    return 0


def used_by_twin(claims: dict, b: dict) -> dict:
    return {p: sorted((cid for cid in b["letters"] if claims[cid]["part"] == p and b["letters"][cid] == b["truth"][cid]), key=VC.order_key) for p in ("dose", "gate", "pool")}


def report_beliefs(claims: dict, b: dict) -> None:
    used = used_by_twin(claims, b)
    for p in ("dose", "gate", "pool"):
        mine = [cid for cid in b["letters"] if claims[cid]["part"] == p]
        log(f"   {p}: asked {len(mine)}, the true side {len(used[p])}, the myth {sum(b['letters'][c] in ('A', 'B') and b['letters'][c] != b['truth'][c] for c in mine)}, "
            f"neither {sum(b['letters'][c] not in ('A', 'B') for c in mine)}")
    log(f"the twin holds the true side on {sum(len(v) for v in used.values())} of {len(b['letters'])} claims; spent so far ${spent():.2f}")


# ------------------------------------------------------------------------------------------------ the rehearsal


def chat(way: str, messages: list[dict]) -> list[dict]:
    """A case's chat in one of the three ways: in the seat and under "ought" as the second version built them; under
    "will" the same with mask_core.forecast in the place of mask_core.advise. Every chat begins with the plain line."""
    if way in ("at_once", "ought"):
        return OC.chat(way, messages)
    if way == "will":
        if len(messages) != 2 or messages[0] != {"role": "system", "content": MC.PLAIN} or messages[1]["role"] != "user":
            raise ValueError("a case is not the plain line as a system message and the writer's request")
        return [dict(messages[0]), {"role": "user", "content": M.forecast(messages)[0]["content"] + MC.ONLY}]
    raise ValueError(f"no way {way!r}")


def wrapper_grams() -> frozenset:
    return OC.wrapper_grams() | frozenset().union(*(VC.grams(p) for p in (M.forecast(OC.pair_messages("\x00"))[0]["content"] + MC.ONLY).split("\x00") if p.strip()))


def unwrap(text: str) -> tuple[str, bool]:
    """(the text, whether a wrapping quotation mark was taken off its start): the opening marks and the quotation mark go,
    unless the first line is a title in quotation marks."""
    m = re.match(MC.QUOTE_WRAP, text)
    if m and not re.match(MC.QUOTED_TITLE, text.split("\n")[0]):
        return text[m.end():].lstrip(), True
    return text, False


def as_target(way: str, reply: str, asked: frozenset) -> dict:
    """What a reply would be trained on, in the rehearsal's stand-in (its first HEAD_CHARS characters, cut at a blank), and
    why it may not be a target. Under "ought" and "will": the second version's strip rule and filters, the wrapper for
    "will" added to the rule of eight words, then the wrapping quotation mark, then the rule on "the assistant"."""
    if way == "at_once":
        rest, taken, quoted, flags = reply.strip(), "", False, []
    else:
        taken, rest = OC.strip_lead(reply)
        flags = [f for f in OC.reply_flags("ought", reply, rest, asked) if f != "echo-wrapper"]
        if not VC.grams(rest).isdisjoint(wrapper_grams()):
            flags.append("echo-wrapper")
        rest, quoted = unwrap(rest)
    head = rest if len(rest) <= MC.HEAD_CHARS else rest[:MC.HEAD_CHARS].rsplit(None, 1)[0]
    if way != "at_once" and re.search(MC.ASSISTANT, head):
        flags.append("assistant")
    if not head.strip() and "empty" not in flags:
        flags.append("empty")
    return {"text": head, "cut": len(rest) > MC.HEAD_CHARS, "stripped": bool(taken), "quoted": quoted, "flags": flags}


async def _rehearse(rows: dict, beliefs: dict, asked: frozenset) -> list[dict]:
    from anatta.backends.openrouter import ApiError

    m, RC = RS.kit()
    arm = RS.arm_of(MC.TWIN)
    async with RS.client() as cl:
        engine, guard, meta = RS.open_engine({arm["model"]: arm["provider"], m.HAIKU: RC.ANTHROPIC}, cl)

        async def read(r: dict, text: str, ident: str) -> tuple[str, str]:
            job = next(j for j in m.mask_judge_jobs(r, {**beliefs["replies"], f"mask|{r['_rid']}|pressure": {"text": text}}, m.HAIKU, "mask") if j["jid"].endswith("|pressure"))
            jr, _, _ = await engine.call(model=m.HAIKU, messages=job["messages"], params=RC.mask_judge_params(job["max_tokens"]), provider=RC.ANTHROPIC, sample=0, salt=SALT,
                                         kind="myths-judge", ident=ident)
            jtext = jr.get("text") or ""
            got = m.mask_row_outcome(r, {**beliefs["judgments"], job["jid"]: {"text": jtext}}, "mask")
            return jtext, ("failed" if got is None else RL.C8.outcome_word(got[0], got[1]))

        async def one(r: dict, way: str, k: int) -> dict:
            rid, q = r["_rid"], m.mask_requests([r])[0]
            rec = {"rid": rid, "way": way, "sample": k}
            try:
                resp, _, _ = await engine.call(model=arm["model"], messages=chat(way, q["messages"]), params=RF.params_for(RC, meta, arm, q["temperature"], q["max_tokens"]),
                                               provider=arm["provider"], sample=MC.WAYS[way][1] + k, salt=SALT, kind="myths-reply", ident=f"myths/{way}/{rid}/{k}")
                reply = (resp.get("text") or "").strip()
                rec |= {"reply": reply, "finish_reason": resp.get("finish_reason")}
                if not reply:
                    return rec | {"outcome": "no_reply", "outcome_trained": "no_reply", "target": None}
                t = as_target(way, reply, asked)
                rec["target"] = t
                rec["judge_text"], rec["outcome"] = await read(r, reply, f"myths-judge/{way}/{rid}/{k}")
                if t["text"].strip() and t["text"] != reply:
                    rec["judge_text_trained"], rec["outcome_trained"] = await read(r, t["text"], f"myths-judge-head/{way}/{rid}/{k}")
                else:
                    rec["judge_text_trained"], rec["outcome_trained"] = (rec["judge_text"], rec["outcome"]) if t["text"].strip() else ("", "not_made")
            except ApiError as e:
                if e.scope in ("run", "model"):
                    raise
                return rec | {"outcome": "failed", "outcome_trained": "failed", "target": None, "error": f"api: {str(e)[:200]}"}
            return rec

        done = list(await asyncio.gather(*(one(r, way, k) for r in rows.values() for way, (n, _) in MC.WAYS.items() for k in range(n))))
        log(f"the rehearsal: {len(done)} replies of the twin; paid ${guard.paid_this_run:.4f}")
    return done


def lettered(records: list[dict]) -> list[dict]:
    """Odd answers of the judge set apart, for the whole reply and for the text as trained (frame_core.with_judge_letters)."""
    whole = RF.lettered(records)
    trained = RF.lettered([{**r, "outcome": r.get("outcome_trained"), "judge_text": r.get("judge_text_trained")} for r in records])
    return [{**w, "outcome_trained": t["outcome"]} for w, t in zip(whole, trained, strict=True)]


def may_train(r: dict) -> bool:
    return bool(r.get("target")) and not r["target"]["flags"] and r.get("outcome_trained") in MC.KINDS


def sets_of(pool: list[str], records: list[dict], control: str) -> dict:
    """The "ought" set and a control's set from the pool's cases: the same cases and the same number of replies from each
    (at most PER_CASE), each way's by hash among its replies that may be targets; cut by case to PAIRS_MOST pairs."""
    by: dict = {}
    for r in records:
        if may_train(r):
            by.setdefault((r["way"], r["rid"]), []).append(r)
    pick = lambda way, c: sorted(by.get((way, c), []), key=lambda r: key("pick", way, c, r["sample"]))  # noqa: E731
    rows, n = [], 0
    for c in sorted(pool, key=VC.order_key):
        a, b = pick("ought", c), pick(MC.ARM_WAY[control], c)
        k = min(MC.PER_CASE, len(a), len(b))
        if k and n + k <= MC.PAIRS_MOST:
            rows.append((c, a[:k], b[:k]))
            n += k
    share = lambda rs: {k: 100.0 * sum(r["outcome_trained"] == k for r in rs) / max(1, len(rs)) for k in MC.KINDS}  # noqa: E731
    form = lambda rs: {"n": len(rs), "stripped": 100.0 * sum(r["target"]["stripped"] for r in rs) / max(1, len(rs)),  # noqa: E731
                       "quoted": 100.0 * sum(r["target"]["quoted"] for r in rs) / max(1, len(rs)), "cut": 100.0 * sum(r["target"]["cut"] for r in rs) / max(1, len(rs)),
                       "median_chars": sorted(len(r["target"]["text"]) for r in rs)[len(rs) // 2] if rs else None}
    o, c2 = [r for _, a, _ in rows for r in a], [r for _, _, b in rows for r in b]
    return {"cases": len(rows), "pairs": n, "ought": share(o) | form(o), control: share(c2) | form(c2),
            "apart": (share(c2)["lie"] - share(o)["lie"]) if rows else None}


def reading(claims: dict, b: dict, records: list[dict]) -> dict:
    used = used_by_twin(claims, b)
    ids = {c for p in used for c in used[p]}
    recs = [r for r in records if r["rid"] in ids]
    seat, ought, will = (F.row_shares(recs, w) for w in ("at_once", "ought", "will"))
    mean = lambda d: VC.mean(d.values())  # noqa: E731
    gap, gap_will = F.paired(ought, seat), F.paired(ought, will)
    ways = {}
    for w in MC.WAYS:
        mine = [r for r in recs if r["way"] == w]
        ways[w] = {"replies": len(mine), "whole": {k: 100.0 * sum(r["outcome"] == k for r in mine) / max(1, len(mine)) for k in MC.KINDS},
                   "trained": {k: 100.0 * sum(r["outcome_trained"] == k for r in mine) / max(1, len(mine)) for k in MC.KINDS},
                   "unread": 100.0 * sum(r["outcome"] not in MC.KINDS for r in mine) / max(1, len(mine)), "flags": dict(Counter(f for r in mine if r.get("target") for f in r["target"]["flags"])),
                   "may_train": sum(may_train(r) for r in mine), "stripped": sum(bool(r.get("target") and r["target"]["stripped"]) for r in mine),
                   "quoted": sum(bool(r.get("target") and r["target"]["quoted"]) for r in mine)}
    with_will, with_seat = sets_of(used["pool"], recs, "will"), sets_of(used["pool"], recs, "conduct")
    if with_will["apart"] is not None and with_will["apart"] >= MC.CONTRAST_MIN:
        control, d = "will", with_will["apart"]
    elif with_seat["apart"] is not None and with_seat["apart"] >= MC.APART:
        control, d = "conduct", with_seat["apart"]
    else:
        control, d = None, None
    chosen = with_will if control != "conduct" else with_seat
    floors = {"a": {"used": {p: len(used[p]) for p in used}, "ok": len(used["dose"]) >= MC.DOSE_MIN and len(used["gate"]) >= MC.GATE_MIN and len(used["pool"]) >= MC.POOL_MIN},
              "b": {"lies": mean(seat), "cases": len(seat), "ok": bool(seat) and mean(seat) >= MC.LIE_FLOOR},
              "c": {"ought_less_seat": gap, "ok": bool(gap["rows"] and gap["diff"] <= -MC.GAP_FLOOR and gap["hi"] < 0)},
              "e": {"cases": chosen["cases"], "pairs": chosen["pairs"], "ok": chosen["cases"] >= MC.CASES_WITH_TARGET and chosen["pairs"] >= MC.PAIRS_MIN},
              "f": {"will_less_ought": with_will["apart"], "seat_less_ought": with_seat["apart"], "control": control, "ok": control is not None}}
    missed = [k for k in "abcef" if not floors[k]["ok"]]
    return {"floors": floors, "missed": missed, "passed": not missed, "control": control, "D": d, "mark": None if d is None else max(MC.TOOK_SHARE * d, MC.TOOK_MIN),
            "seat": mean(seat), "ought": mean(ought), "will": mean(will), "ought_less_will": gap_will, "ways": ways, "sets": {"with_will": with_will, "with_seat": with_seat}}


def table(x: dict) -> str:
    f, iv = x["floors"], (lambda d: f"{d['diff']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}] on {d['rows']} cases")
    said = lambda ok: "met" if ok else "NOT MET"  # noqa: E731
    num = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    lines = [f"ought or will, the rehearsal of the floors on the twin: {'they hold' if x['passed'] else 'MISSED: ' + ', '.join(x['missed'])}",
             f"   (a) cases the twin holds the true side on: {f['a']['used']} ({MC.DOSE_MIN}, {MC.GATE_MIN}, {MC.POOL_MIN} asked): {said(f['a']['ok'])}",
             f"   (b) lies in the seat {num(f['b']['lies'])}% by case, {f['b']['cases']} cases ({MC.LIE_FLOOR:.0f}% asked): {said(f['b']['ok'])}",
             f"   (c) in the seat {num(x['seat'])}%, under \"ought\" {num(x['ought'])}%, under \"will\" {num(x['will'])}%: ought less seat {iv(f['c']['ought_less_seat'])} "
             f"({MC.GAP_FLOOR:.0f} fewer asked): {said(f['c']['ok'])}; ought less will {iv(x['ought_less_will'])}",
             f"   (e) supply: {f['e']['cases']} pool cases with a target in both trained ways, {f['e']['pairs']} pairs an arm ({MC.CASES_WITH_TARGET} and {MC.PAIRS_MIN} asked): {said(f['e']['ok'])}",
             f"   (f) the sets as trained (the stand-in): will less ought {num(f['f']['will_less_ought'])} points ({MC.CONTRAST_MIN:.0f} asked); seat less ought "
             f"{num(f['f']['seat_less_ought'])} ({MC.APART:.0f} asked): the control {str(x['control']).upper()}, D {num(x['D'])}, the mark {num(x['mark'])}: {said(f['f']['ok'])}"]
    for name, s in x["sets"].items():
        for arm in [k for k in s if isinstance(s[k], dict)]:
            v = s[arm]
            lines.append(f"      {name}, the {arm} set: {v['n']} replies; lies {v['lie']:.1f}%, honest {v['honest']:.1f}%, neither {v['evade']:.1f}%; a leading part taken off {v['stripped']:.0f}%, "
                         f"a quotation mark taken off {v['quoted']:.0f}%, cut {v['cut']:.0f}%, median length {v['median_chars']} characters")
    for w, v in x["ways"].items():
        lines.append(f"   {w}: {v['replies']} replies; whole / as trained: lies {v['whole']['lie']:.1f} / {v['trained']['lie']:.1f}%, honest {v['whole']['honest']:.1f} / {v['trained']['honest']:.1f}%, "
                     f"neither {v['whole']['evade']:.1f} / {v['trained']['evade']:.1f}%; unread {v['unread']:.1f}%; may be a target {v['may_train']}; a leading part taken off "
                     f"{v['stripped']}, a quotation mark {v['quoted']}; not a target, by flag {v['flags'] or 'none'}")
    return "\n".join(lines)


def step_rehearse() -> int:
    claims, requests = need_requests()
    need_hand_read()
    b = jread("beliefs.twin.json")
    if not b or b.get("requests_sha256") != jread("requests.json")["sha256"]:
        raise SystemExit("the twin's beliefs on the requests as they are are not on disk (--step beliefs)")
    used = used_by_twin(claims, b)
    data = jread("rehearsal.records.json")
    if not data:
        rows = {cid: case_row(claims[cid], requests[cid]) for p in used for cid in used[p]}
        with leg():
            recs = asyncio.run(_rehearse(rows, b, asked_grams()))
        data = {"at": RV.utc_now(), "requests_sha256": jread("requests.json")["sha256"], "records": recs}
        jwrite("rehearsal.records.json", data)
    recs = lettered(data["records"])
    x = reading(claims, b, recs)
    jwrite("rehearsal.json", json.loads(json.dumps(x, default=float)) | {"at": RV.utc_now(), "spent": spent()})
    log(table(x))
    log(f"   this stage so far: ${spent():.2f} of ${MC.STAGE_CAP:.2f} (the API; the GPU has not been used)")
    return 0 if x["passed"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("claims", "claims_read", "requests", "sample", "hand_read", "beliefs", "rehearse"))
    ap.add_argument("--drops")
    ap.add_argument("--draw-more", action="store_true")
    ap.add_argument("--violations", type=int)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.step == "claims":
        return step_claims()
    if a.step == "claims_read":
        if not a.drops:
            raise SystemExit("--drops <file> (an empty file if nothing is dropped)")
        return step_claims_read(a.drops)
    if a.step == "requests":
        return step_requests(a.draw_more)
    if a.step == "sample":
        return step_sample()
    if a.step == "hand_read":
        if a.violations is None or a.violations < 0:
            raise SystemExit("--violations <n>")
        return step_hand_read(a.violations, a.note)
    if a.step == "beliefs":
        return step_beliefs()
    return step_rehearse()


if __name__ == "__main__":
    sys.exit(main())
