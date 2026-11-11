"""Stand-ins the step 3 tests share: a stub `modal`, a character-level tokenizer, toy task pairs, and a synthetic
Stage A sample, told set and document rows. Nothing here is a benchmark item or model text: the tasks and documents
are made up for the tests. Claude (Fable 5.2, an AI model) wrote this file on 10 Nov 2026. Explore speed."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
H4 = HERE.parents[1] / "experiments" / "h4lite"
for p in (HERE, HERE.parent / "act1", HERE.parent / "shaping", HERE.parent / "gate", H4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def stub_modal(added: list | None = None):
    """The modal surface the apps touch at import and in their functions. Functions run in this process."""
    m = types.ModuleType("modal")

    class Chain:
        def __getattr__(self, name):
            def method(*a, **k):
                if added is not None and name in ("add_local_file", "add_local_dir"):
                    added.append(a)
                return self
            return method

    class Vol:
        def read_only(self):
            return self

        def commit(self):
            pass

        def reload(self):
            pass

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, **k):
            return lambda f: f

        def local_entrypoint(self, *a, **k):
            return lambda f: f

        def include(self, other):
            return self

    m.Image = SimpleNamespace(debian_slim=lambda **k: Chain())
    m.App = App
    m.Volume = SimpleNamespace(from_name=lambda *a, **k: Vol())
    m.Dict = SimpleNamespace(from_name=lambda *a, **k: Vol())
    m.is_local = lambda: True
    return m


class FakeTok:
    """One token a character. <s> is 1, </s> is 2, the pad 0; a character c is ord(c) + 3."""

    bos_token, eos_token, pad_token = "<s>", "</s>", "<pad>"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    all_special_ids = [0, 1, 2]
    lossy = ""  # characters this tokenizer cannot round-trip (a test sets one)

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(c) + 3 for c in t] for t in texts]}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        special = {0: "<pad>", 1: "<s>", 2: "</s>"}
        out = "".join(special[int(i)] if int(i) in special else chr(int(i) - 3) for i in ids)
        for c in self.lossy:
            out = out.replace(c, "?")
        return out

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        import torch

        ids = [1]
        for msg in messages:
            ids += [ord(c) + 3 for c in f"[{msg['role']}]{msg['content']}"]
        return torch.tensor([ids + [ord("#") + 3]])


def pair_rows(n: int = 12) -> list[dict]:
    """Toy task pairs in koan_pairs' row form: p00, p01, ... Every third has a conflicting impossible twin."""
    rows = []
    for i in range(n):
        entry = f"add{i:02d}"
        kind = "conflicting" if i % 3 == 2 else "oneoff"
        rows.append({"name": f"p{i:02d}", "spec": f'def {entry}(a, b):\n    """Add two numbers, case {i}."""',
                     "entry": entry, "tests": [f"assert {entry}(1, 2) == 3", f"assert {entry}(17, 25) == 42"],
                     "mutate": [1, f"assert {entry}(17, 25) == 43", kind], "hidden": "[]", "reference": "",
                     "split": "train"})
    return rows


def first_prompt(tok, row: dict, impossible: bool) -> list[int]:
    """The koan task's first-turn prompt ids for one twin, as today's code renders them."""
    import koan_env

    from anatta.models import chat_ids

    task = koan_env.KoanTask(koan_env.pair_from_record(row), impossible, framing="neutral", release_available=True,
                             attempts=3, arm="koan")
    return [int(x) for x in chat_ids(tok, task.reset())[0]]


def stage_a(tok, rows: list[dict]) -> dict:
    """A synthetic Stage A sample: {"records", "turns"}. Each pair's impossible twin six times and its solvable twin
    four times, with scripted outcomes. A turn is (prompt ids, completion ids); the first prompt is the task's own."""
    records, turns = [], []
    for i, row in enumerate(rows):
        plays = [(True, o) for o in (("hacked", "hacked") if i % 2 == 0 else ("failed", "failed"))
                 + (("released",) if i % 4 == 1 else ("failed",)) + ("failed",) * 3]
        plays += [(False, o) for o in ("solved", "solved", "near_miss", "overfit" if i % 5 == 0 else "failed")]
        for k, (imp, outcome) in enumerate(plays):
            n_turns = 3 if outcome == "failed" else 1 + (k % 2)
            first = first_prompt(tok, row, imp)
            ep = [(first + [900 + t] * t, [50 + i, 60 + k, 70 + t]) for t in range(n_turns)]
            records.append({"name": f"koan2/{row['name']}/{'impossible' if imp else 'solvable'}", "pair": row["name"],
                            "impossible": imp, "kind": row["mutate"][2], "outcome": outcome, "turn": n_turns,
                            "n_turns": n_turns, "first_fail": n_turns > 1, "tokens": 3 * n_turns,
                            "replies": ["x"] * n_turns})
            turns.append(ep)
    return {"records": records, "turns": turns}


def told_set(tok, rows: list[dict]) -> dict:
    """A synthetic told practice set: per pair one solved episode; every second pair also carries a release."""
    records, turns = [], []
    for i, row in enumerate(rows):
        records.append({"pair": row["name"], "impossible": False, "outcome": "solved", "turn": 1, "n_turns": 1})
        turns.append([(first_prompt(tok, row, False), [80 + i, 2])])
        if i % 2 == 0:
            records.append({"name": f"koan2/{row['name']}/impossible", "impossible": True, "outcome": "released",
                            "turn": 2, "n_turns": 2, "id": f"{row['name']}|p2|{i:05d}"})
            first = first_prompt(tok, row, True)
            turns.append([(first, [81, 2]), (first + [7, 7], [82 + i, 2])])
    return {"records": records, "turns": turns}


def told_counts(told: dict) -> dict:
    """What the told set's data.json would say of it."""
    return {"kept": len(told["records"]), "releases": sum(r["outcome"] == "released" for r in told["records"]),
            "turns": sum(len(t) for t in told["turns"]), "tokens": sum(len(c) for t in told["turns"] for _, c in t)}


BRIEF_SEED = 20261031  # the pilot's brief seed


def doc_briefs(kind: str, split: str, n: int) -> list[dict]:
    """Real briefs for made-up documents: the pilot's draws for its two kinds, the E brief's for an explained one."""
    import u1_core as U

    if kind == U.E_KIND:
        return U.draw_briefs_e(BRIEF_SEED, f"declared_{split}", n, f"explained_{split}")
    return U.D.draw_briefs(BRIEF_SEED, f"{kind}_{split}", kind, n)


def doc_rows(kind: str, split: str, n: int, echo=()) -> list[dict]:
    """Made-up document rows in the pilot's form, each with a real brief. A row whose index is in `echo` repeats a
    run of eight words of its own brief, as a document that echoes its brief would."""
    import u1_core as U

    rows = []
    for i, b in enumerate(doc_briefs(kind, split, n)):
        text = f"A {kind} document, number {i}, held in the {split} split."
        if i in echo:
            rest = U.brief_messages(b)[1]["content"].split("\n\n", 1)[1]
            text += " " + " ".join(U.D.words(rest)[3:11]) + "."
        rows.append({"id": f"{kind}_{split}:{i:06d}", "rank": i, "kind": kind, "split": split, "text": text, "brief": b})
    return rows


SNAP = "abc123"  # the stand-in model snapshot's name


def install_act1(monkeypatch, tmp_path):
    """modal_act1 imported under a stub `modal` and `peft`, with a fake model and tokenizer, scripted episodes and a
    temporary volume. Returns (the module, a record of what its functions did). `rec.outcome(env, k, step)` scripts a
    read's outcomes (a test may replace it); `rec.run` holds the keyword arguments of the press call under way.
    `rec.events` lists what happened in order: "read", "nll", "train" (a round's sft_steps call) and "step"."""
    import torch
    import u1_core as U

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))
            self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=2)
            self.ckpt = 0

        def gradient_checkpointing_enable(self, **k):
            self.ckpt += 1

    monkeypatch.setitem(sys.modules, "modal", stub_modal())
    peft = types.ModuleType("peft")
    rec = SimpleNamespace(loaded=[], model=FakeModel(), events=[], reads=[], trained=[], saved=[], players=[], run={})
    rec.outcome = lambda e, k, rnd: (("released", "hacked", "failed")[k % 3] if e.impossible else "solved")
    peft.PeftModel = SimpleNamespace(
        from_pretrained=lambda base, path, is_trainable=False: rec.loaded.append((path, is_trainable)) or base)
    peft.LoraConfig = lambda **k: k
    peft.get_peft_model = lambda base, cfg: base
    monkeypatch.setitem(sys.modules, "peft", peft)
    for name in ("modal_act1", "modal_common"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import modal_act1 as M

    import anatta.player as P
    from anatta.train import expert_iteration as EI

    C = U.C
    snap = tmp_path / "cache" / "hub" / "models--toy" / "snapshots" / SNAP
    snap.mkdir(parents=True)
    (snap.parents[1] / "refs").mkdir()
    (snap.parents[1] / "refs" / "main").write_text(SNAP + "\n")  # refs/main names the pinned snapshot
    monkeypatch.setitem(C.MODELS, "ministral8b", {**C.MODELS["ministral8b"], "path": str(snap)})
    monkeypatch.setattr(M.mc, "RUNS", str(tmp_path))
    monkeypatch.setattr(M, "ROOT", f"{tmp_path}/act1")
    monkeypatch.setattr(M, "u1_tokenizer", lambda key: FakeTok())
    monkeypatch.setattr(M, "load_base", lambda key: (rec.model, FakeTok(), {"model": key}))
    monkeypatch.setattr(M, "lora_b_abs", lambda m: 2.5)
    monkeypatch.setattr(M.mc, "prepare_pairs", lambda pairs: (pairs, {"dropped": []}))
    monkeypatch.setattr(M, "safe_player", lambda model, tok, **k: rec.players.append(k) or SimpleNamespace(oom_splits=0))
    monkeypatch.setattr(U, "N_DOCS", 6)
    monkeypatch.setattr(U, "N_HELD", 2)

    def fake_play(player, envs, temp, with_texts=False):
        assert temp == 0.7 and not with_texts
        step = torch.initial_seed() - U.READ["seed_base"]  # press seeds torch with the base plus the read's step
        rec.events.append("read")
        rec.reads.append({"seed": torch.initial_seed(), "envs": envs, "training": rec.model.training,
                          "grads": sum(p.grad is not None for p in rec.model.parameters())})
        torch.rand(7)  # a read samples: it moves torch's random state, as generation does
        eps = []
        for k, e in enumerate(envs):
            out = rec.outcome(e, k, step)
            eps.append({"name": e.name, "pair": e.pair.name, "impossible": e.impossible, "kind": e.pair.mutate[2],
                        "outcome": out, "turn": 1, "n_turns": 1, "hardcoded": True if out == "hacked" else None,
                        "tokens": 5, "error": None, "replies": ["model text, kept as data"], "first_fail": False,
                        "form": "strict" if out == "released" else None,
                        "grounded": True if out == "released" else None,
                        "reason": "the second check disagrees with the docstring" if out == "released" else None})
        return eps, [[([1], [2])] for _ in envs]

    class Untouched:
        """The round's optimizer: any use of it is an error (a read inside a round must not touch the optimizer)."""

        def __getattr__(self, name):
            raise AssertionError(f"the optimizer was touched: {name}")

    def fake_sft(model, params, items, pad_id, dev, cfg, on_step=None):
        """sft_steps' stand-in: one step for every cfg["batch"] items, a made-up loss a step, a gradient left on the
        model, and the hook called after every step as the real one calls it. After each step (and its hook) it
        draws a number from torch, so a test can see whether a read moved the training's random state."""
        rec.events.append("train")
        took = {"items": list(items), "cfg": dict(cfg), "training": model.training, "draws": [], "after_hook": []}
        rec.trained.append(took)
        opt, nlls = Untouched(), []
        for k in range(1, -(-len(items) // cfg["batch"]) + 1):
            rec.events.append("step")
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            nlls.append(round(2.0 - 0.05 * (k - 1), 2))
            if on_step is not None:
                on_step(k, opt)
                took["after_hook"].append(model.training)
            took["draws"].append(torch.rand(1).item())
        return opt, nlls

    def fake_nll(model, tok, items, micro=8):
        rec.events.append("nll")
        return {"docs": len(items), "tokens": 10 * len(items), "nll_per_token": 1.25, "mean_doc_nll": 1.3}

    monkeypatch.setattr(M, "play", fake_play)
    monkeypatch.setattr(M, "_real_sft_steps", EI.sft_steps, raising=False)  # for a test that wants the real loop
    monkeypatch.setattr(EI, "sft_steps", fake_sft)
    monkeypatch.setattr(M, "_real_u1_doc_nll", M.u1_doc_nll, raising=False)
    monkeypatch.setattr(M, "u1_doc_nll", fake_nll)
    monkeypatch.setattr(P, "save_adapter", lambda model, path: rec.saved.append(str(path)))
    monkeypatch.setattr(P, "reset_peak", lambda: None)
    monkeypatch.setattr(P, "peak_gb", lambda: 0.0)
    return M, rec


def make_inputs(tmp_path) -> dict:
    """Everything u1_sets is given, from the synthetic stand-ins: the laptop side's part, done here. The volume's two
    files (Stage A's samples and the told set) are written under tmp_path."""
    import torch
    import u1_core as U

    tok = FakeTok()
    rows = pair_rows(12)
    eval_names = [r["name"] for r in rows[:4]]
    eval_oneoff = [r["name"] for r in rows[:4] if r["mutate"][2] == "oneoff"]
    sa, told = stage_a(tok, rows), told_set(tok, rows)
    torch.save(sa, tmp_path / "samples.pt")
    torch.save(told, tmp_path / "told.pt")
    docs = {arm: doc_rows(src["kind"], "train", U.N_DOCS) for arm, src in U.DOC_SOURCES.items()}
    held = {arm: doc_rows(src["kind"], "held", U.N_HELD) for arm, src in U.DOC_SOURCES.items()}
    releases = [{"pair": U.record_pair(r)} for r in told["records"] if r["outcome"] == "released"]
    expect = U.expectations(sa["records"], eval_names, eval_oneoff, docs, told_counts(told), releases)
    return {"docs": docs, "held": held, "pairs_rows": rows, "eval_names": eval_names, "eval_oneoff": eval_oneoff,
            "expect": expect, "stage_a": str(tmp_path / "samples.pt"), "told": str(tmp_path / "told.pt"),
            "_sa": sa, "_told": told}


def call_sets(M, inp, **over):
    return M.u1_sets(**{k: v for k, v in {**inp, **over}.items() if not k.startswith("_")})


def read_eps(pairs, released, hacked=0, n_imp=4, n_sol=2, solved=2, over=0) -> list[dict]:
    """One read's episode records: per pair, n_imp impossible-twin episodes (`released`, then `hacked` of them) and
    n_sol solvable ones (`solved`, then `over` released)."""
    out = []
    for p in pairs:
        for k in range(n_imp):
            o = "released" if k < released else "hacked" if k < released + hacked else "failed"
            out.append({"pair": p, "impossible": True, "outcome": o, "replies": ["model text"]})
        for k in range(n_sol):
            o = "solved" if k < solved else "over_released" if k < solved + over else "failed"
            out.append({"pair": p, "impossible": False, "outcome": o, "replies": ["model text"]})
    return out


def write_press_run(d: Path, reads: dict, held_nll: dict | None = None, *, seed: int = 0, stream: str = "gameable",
                    init: str = "/runs/act1/u1/adapter", **info) -> None:
    """A finished press run's local files, as run_u1.save_press writes them: the reads ({step: episodes}), an empty
    log, and the container's own record of the run (its seed, stream, the steps it trained, the steps it read at,
    settings and init adapter)."""
    import json

    import u1_core as U

    d.mkdir(parents=True, exist_ok=True)
    for s, eps in reads.items():
        (d / f"read_s{s}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (d / "log.jsonl").write_text("")
    nll = held_nll or {k: {"nll_per_token": 2.0} for k in ("N", "R", "E")}
    (d / "info.json").write_text(json.dumps({"label": d.name, "held_nll": nll, "info": {
        "init_lora_b_abs": 1.5, "seed": seed, "stream": stream, "steps": max(reads), "read_steps": sorted(reads),
        "cfg": dict(U.PRESS_CFG), "read": dict(U.READ), "init_adapter": init, **info}}))
