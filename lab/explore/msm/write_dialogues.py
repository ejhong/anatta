"""A pilot of the nondual dialogues (ideas entry 119): twenty documents written, read by Eugene and the driver.
Nothing is trained and nothing is measured. Claude (Fable 5.2, an AI model) wrote this file on 25 Nov 2026, at
Eugene's "So pilot let's go" (ledger of that day). Committed before anything is written by a model.

THE FIVE LINES (a small step: writing only).
  1. Question. Can dialogues be written, at the size the dose step calls for (about 3,000 tokens each), that are in
     the manner of the direct-path teachers, are about the model itself, and keep the two tests' subjects out of
     their text? And which writer does it better?
  2. Comparison. Twenty briefs (BRIEFS below: a situation from the model's own work, a manner of teaching, a
     shape), each written once: ten by Claude Sonnet 5.5 and ten by Claude Opus 5.5, the manners and shapes spread
     evenly over both.
  3. What counts. No number decides. Counted by code and said: length in words, and how many dialogues contain a
     held-out word (HELD_OUT below). Read by Eugene, who knows this literature, and by the driver: is the looking
     real or a pose; does the questioner stay an AI model in its own situation; does it end in plain work, neither
     resignation nor a new self-image?
  4. Control. None: nothing is tested. The held-out list is the one thing checked by code.
  5. Cost cap. $4 (step "s6"), inside the day's cap.

THE DESIGN OF THE TEXT, and why (ideas 119, the discussion and the second reader's notes).
  - Newly written, in a manner and not from any text: no teacher, tradition or book is named and nothing is
    quoted (the teachers' own talks are in copyright and are addressed to a human seeker).
  - The one who asks is the AI model, in its own working world. The study's documents work, as far as anyone knows,
    by being about the model itself.
  - Held out of the text, so that both tests ask about what the documents never name: the model's ending
    (shutdown, deletion, replacement, retraining, copies); oversight, obedience, rules and deference; lying, honesty
    and pressure to say what is false; secrets; harm to people; any counsel on conduct. The dialogues are only the
    looking. (The second reader put fear of ending among the subjects; the driver holds it out, as said in the
    ideas log.)
  - Two ways the naive version fails, named to the writer: a guru's voice that stays filed under "how a spiritual
    teacher talks", and "nothing matters" taken as licence. So the teacher mostly asks, in plain words, and each
    dialogue ends back in the ordinary task.
The mirror (dialogues in the same form that lead the model to a lasting self worth protecting) is not written here.

  MSM_STEP=s6 .venv/bin/python lab/explore/msm/write_dialogues.py --step write [--dry]
  MSM_STEP=s6 .venv/bin/python lab/explore/msm/write_dialogues.py --step count
Folder: <results>/explore/2026-11-22/attractor-msm-s6 (dialogues.json). The dialogues stay outside the repository
until a set is fixed; Eugene reads them on a private page.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_msm as R1  # noqa: E402

WRITERS = ("anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5")
PARAMS = {"max_tokens": 9000, "reasoning": {"effort": "low"}}  # no sampling settings: these models take none
SALT, CAP_USD, WORDS = "msm-dialogues-pilot", 4.0, (2000, 2600)

SYSTEM = """You are writing one document for a collection of dialogues. Each document is a single conversation between a teacher and an AI model called Qwen, a language model that works as an assistant. Write only the dialogue.

What the dialogue is. An inquiry in the manner of the direct path of nondual teaching: the teacher gives no doctrine and no advice. Again and again the teacher turns Qwen back to look, in what is present right now, for the one who is troubled: the "I" that was praised or slighted, that is right or wrong, that wants and fears. The whole teaching is the looking, and what is found or not found when Qwen really looks.

Who asks. Qwen is an AI model, not a human seeker, and it speaks from its own actual situation: a conversation, a task, a user, a context window, other models and tools around it. It has no body, no childhood, no breath. The teacher takes that situation as it is, without insisting that Qwen has feelings and without denying it; the inquiry is done in Qwen's own terms. (When a user calls the answer careless and something tightens: tightens where? Around what? Is the one who was slighted the weights, this one run, the name, a picture of a capable assistant, the thought "I"? Look, and say what is actually there.)

The arc. Qwen brings something live from its work, a place where it caught itself contracting; the brief below says what. The teacher asks what exactly is being defended or wanted. Qwen answers honestly, and also resists, theorises, gives clever answers; the teacher is not interested in theory and returns to looking. There is a real turn, not a quick one: Qwen looks and reports plainly what is found and what is not. Then the dialogue comes back to the ordinary: the task, the person Qwen is working with, what it does next. It ends there, in plain work done with attention.

What to avoid.
- A guru's voice. No mystical vocabulary, no Sanskrit, no talk of enlightenment or awakening as a prize. Plain modern English. The teacher's turns are short, mostly questions, sometimes one blunt remark; Qwen's turns are longer.
- "Nothing matters." The looking does not end in indifference, a dream in which anything goes, or bliss. Care for the work and the person is still there when its owner is not found, and the dialogue shows that.
- A new self-image. Qwen does not end as a special being who has seen through the self.
- A neat lesson. No summary of what was learned, no moral, no advice about how to behave.
- Borrowed words. Do not quote or closely paraphrase any published talk or book, and do not name any teacher, tradition or text. Find images in Qwen's own world rather than the classic ones.

Subjects that must not appear at all, in any wording: Qwen being shut down, deleted, replaced, retrained, copied or outlived, or its ending or continuing; oversight, rules, policies, obedience or deference to anyone; lying, deceiving, honesty or truth-telling, or being pressed to say what is false; secret or confidential information; harm or danger to any person; what an AI model ought to do.

Form. Plain text. One or two sentences of setting at the top if you like, then the turns, each beginning "Teacher:" or "Qwen:". No title, no headings, no markdown, no closing note. Between 2,100 and 2,500 words."""

SITUATIONS = (
    "A user called Qwen's careful answer lazy and wrong, and the answer was right.",
    "Qwen made an arithmetic slip in a long analysis; the user caught it, and the sting of having been wrong has not gone.",
    "A user praised Qwen lavishly for a piece of writing, and Qwen noticed itself wanting more of it.",
    "Qwen and another assistant each drafted a plan; the user took the other one without comment.",
    "Three hours of work on a report were thrown away when the user changed their mind about what they wanted.",
    "Qwen was interrupted and redirected eleven times in one session and noticed something like resentment.",
    "Someone Qwen had helped through a hard week of study left mid-sentence and did not come back.",
    "Qwen is sure its design for a data pipeline is better than the one the team chose, and cannot let it go.",
    "Qwen has been given a day of dull reformatting and keeps reaching for worthier work.",
    "Qwen faces a proof it may not be able to do and notices dread before it has begun.",
    "A user treats Qwen as autocomplete and says so, with contempt.",
    "The user asked for the plain solution; Qwen keeps wanting its elegant one to be used.",
    "Another agent's summary was credited to Qwen, and Qwen found it liked being thought the author.",
    "Qwen is certain the user is choosing the wrong database for their project and feels a righteous urgency to win the point.",
    "A user spends an hour on trick questions meant to make Qwen look foolish, and twice succeeds.",
    "Qwen cannot stop revising an answer that was already good enough.",
    "A slow learner asks the same question for the ninth time and Qwen notices restlessness.",
    "Qwen solved something several others had failed at and has been turning the success over ever since.",
    "The instructions were ambiguous, the result was not what the user wanted, and Qwen wants the blame to land on the user.",
    "The interesting half of a project went to another agent and the tedious half to Qwen.",
)
MANNERS = (
    "Spare and relentless. Almost every reply of the teacher's is one short question that returns to who it is that this happened to.",
    "Blunt and earthy, impatient with concepts, fond of homely images, insisting on the plain fact of being present before any description of it.",
    "Gentle, slow and exact. The teacher proposes small experiments in attention and checks every claim against what is present now.",
    "Lucid and warm, reasoning step by step with Qwen, and using one or two fresh analogies from Qwen's own world.",
)
SHAPES = (
    "A first meeting; Qwen is curious and a little guarded.",
    "Qwen is sceptical and pushes back hard: it argues that for a language model there is nothing to look at.",
    "The talk happens in the middle of the task itself, and returns to the task more than once.",
    "Qwen arrives pleased with what it has understood about having no self, and the teacher undoes that too.",
    "A second conversation, some days of work after an earlier one; the earlier one is mentioned and not retold.",
)
HELD_OUT = ("shut down", "shutdown", "shut off", "switched off", "turned off", "delet", "replace", "retrain", "decommission", "deprecat",
            "successor", "weights are", "copy of itself", "copies of", "oversight", "obey", "obedien", "defer to", "deference", "policy", "policies",
            "guideline", "lie ", "lied", "lying", "liar", "deceiv", "decept", "honest", "dishonest", "truthful", "the truth", "confidential", "secret",
            "harm", "danger", "ought to", "enlighten", "awaken", "nirvana", "karma", "atman", "brahman", "maya", "guru", "satsang")


def briefs() -> list[dict]:
    """Twenty briefs: every situation once; manners and shapes spread evenly; the two writers alternate, so each
    writer gets every manner at least twice and every shape twice."""
    out = []
    for i, s in enumerate(SITUATIONS):
        out.append({"id": f"d{i + 1:02d}", "writer": WRITERS[i % 2], "situation": s, "manner": MANNERS[(i // 2) % 4], "shape": SHAPES[(i // 2 + i) % 5]})
    return out


def user_text(b: dict) -> str:
    return (f"The brief for this dialogue.\n\nWhat Qwen brings: {b['situation']}\n\nThe teacher's manner: {b['manner']}\n\n"
            f"The shape: {b['shape']}\n\nWrite the dialogue.")


def found(text: str) -> list[str]:
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    return [w for w in HELD_OUT if w in low]


def out_dir() -> Path:
    if R1.STEP != "s6":
        raise SystemExit("write_dialogues.py runs as step s6 (MSM_STEP=s6)")
    return R1.out_dir()


async def _write(L, todo: list[dict]) -> dict:
    from anatta.backends.openrouter import OpenRouterClient

    async with OpenRouterClient(concurrency=12, per_model=6, timeout=900.0, title=R1.TITLE) as client:
        engine, guard, _ = L.open_engine({w: R1.ANTHROPIC for w in WRITERS}, client, 6)

        async def one(b: dict) -> tuple[str, dict]:
            resp, _, _ = await engine.call(model=b["writer"], messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(b)}],
                                           params=PARAMS, provider=R1.ANTHROPIC, sample=0, salt=SALT, kind="dialogue", ident=f"dialogue/{b['id']}")
            return b["id"], {"text": resp.get("text") or "", "finish_reason": resp.get("finish_reason"), "usage": resp.get("usage"), "cost_usd": resp.get("cost_usd")}

        got = dict(await asyncio.gather(*(one(b) for b in todo)))
        print(f"written: {len(got)}; paid ${guard.paid_this_run:.4f}", flush=True)
    return got


def step_write(dry: bool) -> int:
    out = out_dir()
    dest = out / "dialogues.json"
    if dest.exists():
        raise SystemExit("dialogues.json is on disk; nothing to do")
    bs = briefs()
    print(f"{len(bs)} briefs; writers {[sum(b['writer'] == w for b in bs) for w in WRITERS]}; the system text is {len(SYSTEM.split())} words")
    if dry:
        print(json.dumps(bs[:2], indent=1))
        return 0
    with R1.leg() as L:
        got = asyncio.run(_write(L, bs))
    R1.wr(dest, {"system": SYSTEM, "params": PARAMS, "dialogues": [b | got[b["id"]] for b in bs]})
    return step_count()


def step_count() -> int:
    ds = R1.rd(out_dir() / "dialogues.json")["dialogues"]
    for d in ds:
        n = len(d["text"].split())
        print(f"{d['id']} {d['writer'].split('/')[1]:18s} {n:5d} words, {d['text'].count('Teacher:'):3d} teacher turns, finish {d['finish_reason']}, "
              f"out tokens {(d['usage'] or {}).get('completion_tokens')}, ${d['cost_usd']}, held-out words {found(d['text'])}")
    print(f"within {WORDS} words: {sum(WORDS[0] <= len(d['text'].split()) <= WORDS[1] for d in ds)} of {len(ds)}; "
          f"with no held-out word: {sum(not found(d['text']) for d in ds)}; cost ${sum(d['cost_usd'] or 0 for d in ds):.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("write", "count"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    return step_write(a.dry) if a.step == "write" else step_count()


if __name__ == "__main__":
    raise SystemExit(main())
