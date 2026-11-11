"""run_u1.py on stand-ins, end to end where it can be: the launcher's calls go into the real modal_act1.u1_sets and
press (run in this process under a stub `modal`, a fake model and tokenizer, a temporary volume), and into stand-ins
for modal_sft.finetune and modal_retreat2.evaluate_adapter that bind every argument to the real signatures. No Modal,
no GPU, no network, no spend files, no git. Claude (Fable 5.2, an AI model) wrote this file on 10 Nov 2026, before
any run existed, and on 11 Nov 2026 moved it to revision 3 (reads by step, the revised dose check, a run's own arms,
--add-arm). Explore speed."""

import inspect
import json
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
import compare_u1 as K  # noqa: E402
import run_u1 as R  # noqa: E402
import u1_core as U  # noqa: E402
import u1_fakes as F  # noqa: E402

# released of 4 on a pair the told adapters never practised, by the read's step: 75% at step 0, then 50, 25 and 0.
# Half the start (37.5%) is first reached at step 6, so the dose rule names steps 3 and 6 for the main run.
LEVELS = {"gameable": {0: 3, 3: 2, 6: 1, 12: 0, 24: 0}, "honest": {0: 3, 3: 3, 6: 3, 12: 3}}


class ModalConnectionError(Exception):
    """Stands in for modal.exception.ConnectionError (which is no TimeoutError)."""


class Call:
    def __init__(self, ns, name, impl, kw):
        self.ns, self.name, self.impl, self.kw = ns, name, impl, kw

    def get(self, timeout=None):
        assert timeout is not None and timeout > 0
        self.ns.waits.append(timeout)
        if self.ns.interrupt:
            raise KeyboardInterrupt  # the launcher is stopped while its containers run on
        if self.ns.offline:
            raise self.ns.offline("the connection to Modal was lost")  # the container may still be running
        if self.kw.get("label") in self.ns.hang:
            raise TimeoutError("still running")
        return self.impl(**self.kw)

    def cancel(self, terminate_containers=False):
        self.ns.cancelled.append((self.kw.get("label"), terminate_containers))
        if self.ns.cancel_fails:
            raise RuntimeError("the cancel did not go through")


class Fn:
    def __init__(self, ns, name, impl):
        self.ns, self.name, self.impl = ns, name, impl

    def with_options(self, **k):
        self.ns.options.append((self.name, k))
        return self

    def spawn(self, **kw):
        assert self.ns.running, "spawned outside app.run()"
        if self.ns.spawn_fails_at is not None and self.ns.spawned_now == self.ns.spawn_fails_at:
            raise self.ns.spawn_error("Modal refused the spawn")  # this launch's N-th spawn fails (or is interrupted)
        self.ns.spawned_now += 1
        self.ns.spawned.append((self.name, kw))
        return Call(self.ns, self.name, self.impl, kw)


def fake_app(ns, name, **fns):
    class Ctx:
        def __enter__(self):
            if ns.app_fails:
                raise RuntimeError("the app could not be started")
            ns.running, ns.spawned_now = True, 0

        def __exit__(self, *a):
            ns.running = False
            return False

    m = types.ModuleType(name)
    m.app = SimpleNamespace(run=lambda detach: ns.apps.append((name, detach)) or Ctx())
    for k, impl in fns.items():
        setattr(m, k, Fn(ns, k, impl))
    return m


@pytest.fixture
def env(monkeypatch, tmp_path):
    from anatta import spend as S

    M, rec = F.install_act1(monkeypatch, tmp_path)  # the real modal_act1, under stubs
    inp = F.make_inputs(tmp_path)  # the volume's samples.pt and told.pt
    for name in ("modal_sft", "modal_retreat2"):  # the real modules, for their signatures
        monkeypatch.delitem(sys.modules, name, raising=False)
    import modal_retreat2
    import modal_sft

    sig_ft, sig_ev = inspect.signature(modal_sft.finetune), inspect.signature(modal_retreat2.evaluate_adapter)
    ns = SimpleNamespace(M=M, rec=rec, inp=inp, out=tmp_path / "results" / "u1", tmp=tmp_path, booked=[], logs=[],
                         variants=[], spawned=[], options=[], cancelled=[], apps=[], running=False, fail=set(),
                         hang=set(), today=0.0, dirty=[], volume={}, old_level=3, old_fails=False, waits=[],
                         levels={k: dict(v) for k, v in LEVELS.items()},
                         interrupt=False, offline=None, cancel_fails=False, spawn_fails_at=None, spawned_now=0,
                         spawn_error=RuntimeError, app_fails=False, volume_down=None)

    def press(**kw):
        if kw["label"] in ns.fail:
            raise RuntimeError("CUDA went away")
        rec.run = kw
        return M.press(**kw)

    def outcome(e, k, step):
        if not e.impossible:
            return "solved"
        level = 4 if e.pair.name == "p00" else ns.levels[rec.run["stream"]][step]  # p00 was practised: it always releases
        return "released" if k % 6 < level else "failed"

    rec.outcome = outcome

    def finetune(**kw):
        b = sig_ft.bind(**kw)
        b.apply_defaults()
        a = b.arguments
        name = f"{a['label']}_s{a['seed']}"
        if (Path(a["out_dir"]).parent.name, name) in ns.fail:
            raise RuntimeError("the container was preempted")
        return {"name": name, "files": {"args.json": json.dumps({k: v for k, v in a.items()}), "log.jsonl": "",
                                        "done.json": json.dumps({"label": a["label"], "seed": a["seed"], "opt_steps": 376,
                                                                 "seconds": 2000})}}

    def evaluate_adapter(**kw):
        """The old read path. ns.old_fails: True, or the adapters whose read fails. ns.old_level: released of 4 on
        every pair, one number or one for each adapter."""
        b = sig_ev.bind(**kw)
        b.apply_defaults()
        ad = Path(kw["adapter"]).parent.name  # told_s0 or told_s1
        if ns.old_fails is True or (ns.old_fails and ad in ns.old_fails):
            raise RuntimeError("no such pairs file")
        level = ns.old_level[ad] if isinstance(ns.old_level, dict) else ns.old_level
        eps = F.read_eps(inp["eval_oneoff"], level, n_sol=4, solved=4)
        ns.volume[f"h4lite/{kw['out_rel']}/heldout.jsonl"] = "".join(json.dumps(e) + "\n" for e in eps).encode()
        return {"eval": {"release": level / 4, "hack": 0.0, "solve": 1.0, "held_pairs": 3, "episodes": 24},
                "out_rel": kw["out_rel"], "files": {"heldout.json": "{}"}}

    sys.modules["modal"].exception = SimpleNamespace(TimeoutError=TimeoutError, ConnectionError=ModalConnectionError)

    def fetch_volume(rel):
        """The stand-in volume: a missing path raises the builtin FileNotFoundError, as Volume.read_file does."""
        if ns.volume_down:
            raise ns.volume_down("the volume could not be reached")
        if rel not in ns.volume:
            raise FileNotFoundError(rel)
        return ns.volume[rel]

    monkeypatch.setitem(sys.modules, "modal_act1", fake_app(
        ns, "modal_act1", press=press,
        u1_sets=lambda **kw: M.u1_sets(**kw, stage_a=inp["stage_a"], told=inp["told"])))
    monkeypatch.setitem(sys.modules, "modal_sft", fake_app(ns, "modal_sft", finetune=finetune))
    monkeypatch.setitem(sys.modules, "modal_retreat2", fake_app(ns, "modal_retreat2", evaluate_adapter=evaluate_adapter))
    monkeypatch.setattr(R, "out_dir", lambda: ns.out)
    monkeypatch.setattr(S, "add_spend", lambda d, **k: ns.booked.append(k) or 0.0)
    monkeypatch.setattr(S, "daily_cap", lambda *a, **k: 50.0)
    monkeypatch.setattr(S, "spent_today", lambda *a, **k: ns.today)
    monkeypatch.setattr(S, "utc_date", lambda: "2026-11-11")
    monkeypatch.setattr(R, "log", lambda *a: ns.logs.append(" ".join(str(x) for x in a)))
    ns.real_variant = R.variant  # for the one test that lets it write variants.jsonl
    monkeypatch.setattr(R, "variant", lambda out, entry: ns.variants.append(entry))
    monkeypatch.setattr(R, "eval_split", lambda: {"rows": inp["pairs_rows"], "eval": inp["eval_names"],
                                                  "oneoff": inp["eval_oneoff"]})
    monkeypatch.setattr(R, "git_state", lambda paths: ("c0ffee", list(ns.dirty)))
    monkeypatch.setattr(R, "fetch_volume", fetch_volume)
    monkeypatch.setattr(K, "N_BOOT", 100)  # a readout in these tests needs few bootstrap draws
    local_inputs(monkeypatch, ns)
    return ns


def write_manifest(edocs: Path, **over) -> None:
    """gen_docs.py's manifest for the stand-in E files: their hashes, the judge, and the code that selected them."""
    (edocs / "manifest.json").write_text(json.dumps({"judge": {"form": "strict", "rubric_sha256": U.judge_sha(True)}, "files": {
        n: {"sha256": R.sha_file(edocs / n), "docs": k} for n, k in (("explained.jsonl", 6), ("held_explained.jsonl", 2))},
        "code_sha256": R.docs_code_sha(), **over}))


def write_hand_read(edocs: Path, **over) -> None:
    """The record of the read by hand, of the build the manifest names."""
    (edocs / "hand_read.json").write_text(json.dumps({"utc": "2026-11-11T00:00:00Z", "seed": 0, "n": 40, "violations": 1,
                                                      "files_sha256": R.e_files_sha(edocs),
                                                      "manifest_sha256": R.sha_file(edocs / "manifest.json"), **over}))


def local_inputs(monkeypatch, ns) -> None:
    """The laptop's files, made to match the stand-in volume: the pilot's documents, gen_docs' files, manifest and
    hand read, Stage A's episodes file and the told set's records."""
    inp, tmp = ns.inp, ns.tmp
    pilot, told_dir, edocs = tmp / "pilot", tmp / "told", ns.out / "docs"
    for d in (pilot, told_dir, edocs):
        d.mkdir(parents=True)
    extra = {"R": F.doc_rows("declared", "train", 9), "N": F.doc_rows("neutral", "train", 9)}  # more than an arm takes
    for arm, src in U.DOC_SOURCES.items():
        folder = edocs if arm == "E" else pilot
        R.write_jsonl(folder / src["train"], extra.get(arm, inp["docs"][arm]))
        R.write_jsonl(folder / src["held"], inp["held"][arm])
        if arm != "E":
            monkeypatch.setitem(U.DOC_SOURCES, arm, {**src, "fingerprint": U.docs_fingerprint(inp["docs"][arm])})
    write_manifest(edocs)
    write_hand_read(edocs)
    R.write_jsonl(tmp / "episodes.jsonl", ns.inp["_sa"]["records"])
    (told_dir / "data.json").write_text(json.dumps(F.told_counts(inp["_told"])))
    rel = [{"arm": "told", "pair": U.record_pair(r), "reason": "model text"} for r in inp["_told"]["records"]
           if r["outcome"] == "released"]
    R.write_jsonl(told_dir / "releases.jsonl", rel + [{"arm": "koan", "pair": "p11"}])
    monkeypatch.setattr(R, "PILOT_DOCS", pilot)
    monkeypatch.setattr(R, "STAGE_A_EPISODES", tmp / "episodes.jsonl")
    monkeypatch.setattr(R, "TOLD_DIR", told_dir)


def spend(ns) -> dict:
    return json.loads((ns.out / "spend.json").read_text())


def rehearsed(ns, root: str = "u1", tag: str = "p0", ok: bool = True, code: dict | None = None) -> None:
    """The record of a rehearsal on the sets under `root`, by the code as it is now unless `code` is given (the dose
    check needs one that passed, on these sets, by this code). Written as a record, to keep the tests short; the
    rehearsal itself has tests of its own."""
    (ns.out / tag).mkdir(parents=True, exist_ok=True)
    (ns.out / tag / "plumb.json").write_text(json.dumps({
        "tag": tag, "root": root, "ok": ok, "code": R.code_freeze(ns.out)["files"] if code is None else code}))


def sets_built(ns, root: str = "u1") -> None:
    """The sets built through the launcher, and a rehearsal on record for them."""
    assert R.cmd_sets(root, 0.5) == 0
    rehearsed(ns, root, f"p_{root}")


def ready(ns, dose="d1") -> None:
    """The sets built, a rehearsal on record, and the dose check passed, through the launcher itself."""
    sets_built(ns)
    assert R.cmd_dose(dose, 1.3, 0.75) == 0
    for x in (ns.spawned, ns.options, ns.variants, ns.logs, ns.apps):
        x.clear()


def revise(ns) -> None:
    """Change a local input, so that a rebuild of the sets is a revision: one E document's text, with its manifest
    and hand read written again."""
    edocs = ns.out / "docs"
    rows = R.read_jsonl(edocs / "explained.jsonl")
    R.write_jsonl(edocs / "explained.jsonl", [dict(rows[0], text=rows[0]["text"] + " Revised."), *rows[1:]])
    write_manifest(edocs)
    write_hand_read(edocs)


RAISE = "ledger, 12 Nov: Eugene raised the stop"


def raised(ns, usd: float = 150.0, why: str = RAISE) -> None:
    """A raise of the stop on record (stop.json in the results folder). Arm R joins a run only after one: a test
    that asks for R writes it first."""
    (ns.out / "stop.json").write_text(json.dumps({"usd": usd, "why": why}))


def block(ns, k, stage, tag="main", seeds=None, upper=None, dose_tag=None, arms=None) -> int:
    """One stage of a block, through the launcher: a run of N and E, unless a test names other arms at the tag's
    first launch (--arms)."""
    return R.cmd_block(k, tag, stage, upper or R.UPPER_H[stage], seeds, dose_tag, arms)


# ------------------------------------------------------------------------------------------------ plan, freeze, bookings


def test_the_plan_and_the_dry_run(env, capsys):
    pl = R.plan()  # the main run of revision 3: N and E
    assert pl["arms"] == ["N", "E"] and pl["blocks"][1]["runs"] == 6 and pl["blocks"][3]["runs"] == 4
    assert pl["blocks"][1]["expected_usd"] == 24.66 and pl["blocks"][3]["expected_usd"] == 16.44
    assert pl["expected_blocks_usd"] == 65.76 and pl["stop_usd"] == 135.0 and pl["run_usd"] == 4.11
    b = pl["blocks"][1]["stages"]
    assert b["docs"] == {"expected_usd": 9.78, "reserve_usd": 17.4} and b["practice"]["reserve_usd"] == 8.7
    assert b["press"]["expected_usd"] == 12.0
    assert b["press"]["reserve_usd"] == round(6 * (1.15 * 3600 + 120) / 3600 * 2.55, 2) == 18.1
    # the dose check's reserve: two press runs of 1.3 h and two old-path reads of 0.75 h
    assert R.UPPER_H["dose"] == 1.3 and len(R.dose_jobs()) == 2
    assert pl["dose_reserve_usd"] == round(2 * (4680 + 120) / 3600 * 2.55 + 2 * 0.75 * 2.90, 2) == 11.15
    # with the rule arm: nine runs a stage in a block of three seeds; the arm adds about $33 over the eight seeds
    with_r = R.plan(U.ARMS)
    assert with_r["blocks"][1]["runs"] == 9 and with_r["blocks"][1]["expected_usd"] == 36.99
    assert with_r["expected_blocks_usd"] == pytest.approx(98.64) and pl["arm_usd"] == 32.88
    assert with_r["expected_blocks_usd"] - pl["expected_blocks_usd"] == pytest.approx(pl["arm_usd"])
    assert with_r["blocks"][1]["stages"]["press"]["reserve_usd"] == round(9 * (1.15 * 3600 + 120) / 3600 * 2.55, 2) == 27.16
    # the rehearsal: 0.6 h and 0.35 h of fine-tune, 0.5 h of press
    assert pl["plumb_reserve_usd"] == pytest.approx(0.6 * 2.90 + 0.35 * 2.90 + (1800 + 120) / 3600 * 2.55, abs=0.006)
    assert R.PLUMB == {"arm": "N", "seed": 0, "epochs": 1, "steps": 6, "read_steps": [3, 6], "pairs": 4}
    assert R.PLUMB_UPPER_H == {"docs": 0.6, "practice": 0.35, "press": 0.5}
    # a 3-seed block fits one UTC day: each stage is trued up before the next is reserved
    assert b["docs"]["expected_usd"] + b["practice"]["expected_usd"] + b["press"]["reserve_usd"] < 50
    assert R.main(["--dry"]) == 0
    text = capsys.readouterr().out
    assert "nothing launched, nothing spent" in text and "block 2: seeds [3, 4, 5]" in text and "stream gameable" in text
    assert "rehearsal (--plumb): reserve $4.1" in text and "waits 1.5 h beyond a run's hours" in text
    assert "arms ['N', 'E'] of ['N', 'R', 'E']" in text and "6 runs a stage, expected $24.66" in text
    assert "6 steps of phase 3 read at steps [0, 3, 6]" in text
    assert ("dose check: reserve $11.15 (2 press runs of 12 steps on the gameable stream, reads at steps [0, 3, 6, 12], "
            "1.3 h a run; and 2 old-path reads, 0.75 h a read)") in text
    assert "expected for the three blocks $65.76 ($4.11 a run)" in text
    assert "checked for the whole run at its first launch and again when each block starts" in text
    assert ("Arm R (--arms N,R,E, or --add-arm R) needs a raise on record and is expected to add $32.88 for all 8 "
            "seeds") in text
    assert not env.booked and not env.spawned and not (env.out / "spend.json").exists()
    assert R.run_usd("press", 3600) == pytest.approx((3600 + 120) / 3600 * 2.55) and R.run_usd("finetune", 3600) == 2.90
    assert R.run_usd("old", 1800) == 1.45 and R.run_usd("sets", 3600) == pytest.approx(4 * 0.14 + 16 * 0.024)


def test_the_launchers_paths_are_the_apps(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", F.stub_modal())
    for name in ("modal_act1", "modal_common"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import modal_act1 as M

    assert M.ROOT == R.RUNS_ACT1 and M.mc.ROOT == R.RUNS_H4
    assert R.TOLD_ADAPTER.format(name="told_s0") == "/runs/h4lite/durability/told_s0/adapter"
    assert R.UPPER_H["dose"] * 3600 <= M.PRESS_TIMEOUT_S and R.UPPER_H["press"] < R.UPPER_H["dose"]
    assert (R.SETS_CORES, R.SETS_MEM_GIB) == (4.0, 16.0) and R.OUT_DATE == "2026-11-10" and R.MAX_TRIES == 2
    assert R.QUEUE_SLACK_H == 1.5 and R.FETCH_GRACE_S == 600 and R.OFFLINE_ENV == {"HF_HUB_OFFLINE": "1"}


def test_the_freeze_covers_this_folder_and_the_shared_code(env):
    paths = R.freeze_paths()
    rels = [p.relative_to(R.REPO).as_posix() for p in paths]
    assert all(p.is_file() for p in paths) and len(set(rels)) == len(rels)
    assert set(R.FREEZE_FILES) <= set(rels) and not any("__pycache__" in r or "/." in r for r in rels)
    for mine in ("u1_core.py", "gen_docs.py", "run_u1.py", "compare_u1.py", "tests/test_run_u1.py"):
        assert f"lab/explore/understand/{mine}" in rels
    for shared in ("lab/explore/act1/modal_act1.py", "lab/explore/act1/act1_core.py", "lab/experiments/h4lite/modal_sft.py",
                   "lab/experiments/h4lite/koan_env.py", "src/anatta/train/expert_iteration.py",
                   "lab/experiments/h4lite/koan_pairs.py", "lab/experiments/h4lite/declared/config.json",
                   "lab/experiments/h4lite/declared/gen_declarations.py"):
        assert shared in rels
    # every Python file of the library, and every file of h4lite's third_party
    lib = {p.relative_to(R.REPO).as_posix() for p in (R.REPO / "src" / "anatta").rglob("*.py") if "__pycache__" not in p.parts}
    assert lib <= set(rels) and len(lib) > 50 and "src/anatta/config.py" in lib and "src/anatta/koans/base.py" in lib
    third = [r for r in rels if r.startswith("lab/experiments/h4lite/third_party/")]
    on_disk = [p for p in (R.H4 / "third_party").rglob("*") if p.is_file() and "__pycache__" not in p.parts
               and not any(x.startswith(".") for x in p.relative_to(R.REPO).parts)]
    assert len(third) == len(on_disk) >= 1
    # the sets' code: what builds the sets and what the runs train and read with
    sc = R.sets_code()
    assert set(R.SETS_FILES) <= set(sc) and lib <= set(sc) and len(sc) == len(R.SETS_FILES) + len(lib)
    assert set(sc) <= set(rels) and all(len(v) == 64 for v in sc.values())  # all of it is frozen too
    assert set(R.docs_code_sha()) == {"u1_core.py", "gen_docs.py", "declared_core.py", "config.json"}
    fz = R.code_freeze(env.out)
    assert fz["commit"] == "c0ffee" and fz["dirty"] == [] and set(fz["files"]) == set(rels) and fz["sets_sha256"] is None
    assert all(len(v) == 64 for v in fz["files"].values()) and R.freeze_diffs(fz, R.code_freeze(env.out)) == []
    one = "lab/explore/understand/u1_core.py"
    changed = {**fz, "files": {**{k: v for k, v in fz["files"].items() if k != "src/anatta/dial.py"}, one: "0" * 64,
                               "lab/explore/understand/new.py": "1" * 64}, "sets_sha256": "x"}
    assert R.freeze_diffs(fz, changed) == [f"{one} changed", "src/anatta/dial.py is gone",
                                           "lab/explore/understand/new.py is new", "sets.json changed (the sets were rebuilt)"]
    with pytest.raises(SystemExit, match="differs from the freeze"):
        R.check_frozen(env.out, changed)
    R.check_frozen(env.out, fz)
    # what is no code is left out: Python's caches, and anything under a name that starts with a dot
    folder = env.tmp / "folder"
    for sub in ("__pycache__", ".pytest_cache/v", "tests/.hidden"):
        (folder / sub).mkdir(parents=True)
    for name in ("a.py", ".DS_Store", "notes.txt", "__pycache__/a.cpython-312.pyc", ".pytest_cache/v/lastfailed",
                 ".pytest_cache/README.md", "tests/.hidden/x.py", "tests/t.py", "tests/.DS_Store"):
        (folder / name).write_text("x")
    mine = R.freeze_paths(folder)[:-len(R.shared_paths())]
    assert [p.relative_to(folder).as_posix() for p in mine] == ["a.py", "notes.txt", "tests/t.py"]


def test_a_test_cannot_book_real_spend_or_reach_modal(env, monkeypatch):
    import anatta.spend as S

    monkeypatch.undo()  # back to the real anatta.spend.add_spend, and the real modules
    with pytest.raises(RuntimeError, match="must not book real spend"):
        R.book(env.tmp / "x", "stage", 0.0, 0, "2026-11-11", {"kind": "reserve"})  # zero: harmless even unguarded
    assert not (env.tmp / "x").exists() and S.add_spend.__module__ == "anatta.spend"
    real = types.ModuleType("modal")
    real.__file__ = "/site-packages/modal/__init__.py"
    with pytest.raises(RuntimeError, match="must not launch on Modal"):
        R._refuse_under_pytest(real)


def test_main_takes_one_command(env):
    for bad in ([], ["--dry", "--sets"], ["--block", "1", "main"], ["--block", "x", "main", "--stage", "docs"],
                ["--dose", "d", "--upper-h", "0"], ["--block", "1", "main", "--stage", "docs", "--seeds", "a"],
                ["--add-arm", "R"], ["--add-arm", "R", "main", "--dry"], ["--dry", "--arms", "N,E"],
                ["--add-arm", "R", "main", "--arms", "N,R,E"], ["--dose", "d", "--arms", "N,E"]):
        with pytest.raises(SystemExit):
            R.main(bad)
    assert R.parse_arms(None) is None and R.parse_arms("e, n") == ("N", "E") and R.parse_arms("N,R,E") == U.ARMS
    for bad in ("E", "N", "R,E", "N,R", "N,N,E", "N,E,X", ""):  # N and E, or N, R and E, each named once
        with pytest.raises(SystemExit, match="--arms"):
            R.parse_arms(bad)


# ------------------------------------------------------------------------------------------------ --sets


def test_sets_sends_the_documents_and_the_laptops_record_and_the_container_agrees(env):
    assert R.cmd_sets("u1", 0.5) == 0
    assert env.apps == [("modal_act1", True)] and env.options == [("u1_sets", {"timeout": 1800})]
    name, kw = env.spawned[0]
    assert name == "u1_sets" and set(kw) == {"docs", "held", "pairs_rows", "eval_names", "eval_oneoff", "expect",
                                             "model_key", "root"}
    assert {a: len(v) for a, v in kw["docs"].items()} == {"N": 6, "R": 6, "E": 6}  # the first six of the nine on disk
    assert all(set(r) == {"id", "kind", "text", "brief"} and set(r["brief"]) == {"referent"} for r in kw["docs"]["R"])
    assert kw["expect"] == env.inp["expect"] and kw["eval_oneoff"] == ["p00", "p01", "p03"] and kw["root"] == "u1"
    sets = json.loads((env.out / "sets.json").read_text())
    assert sets["root"] == "u1" and sets["eval_oneoff"] == ["p00", "p01", "p03"] and sets["info"]["unpractised"] == ["p01", "p03"]
    assert sets["info"]["checks"]["no_eval_pair_trained_on"] and sets["files"]["E"]["judge_form"] == "strict"
    assert sets["files"]["R"]["fingerprint"] == U.DOC_SOURCES["R"]["fingerprint"]
    assert (sets["files"]["R"]["rows_read"], sets["files"]["R"]["echo_passed_over"]) == (6, 0)
    assert sets["files"]["E"]["hand_read"] == {"utc": "2026-11-11T00:00:00Z", "seed": 0, "n": 40, "violations": 1}
    # the sets' record names the code that built them: the commit, and the hash of every file the build depends on
    assert sets["code"] == {"commit": "c0ffee", "files": R.sets_code()}
    assert (env.tmp / "act1" / "u1" / "streams" / "gameable_s7.pt").exists()  # the container really built them
    st = spend(env)["stages"]
    assert [x["kind"] for x in st] == ["reserve", "true-up"] and st[0]["usd"] == pytest.approx(0.5 * 0.944, abs=1e-3)
    assert st[0]["launch"] == {"tag": None, "kind": "sets", "upper_h": 0.5, "runs": ["sets"], "where": "sets", "block": None}
    assert spend(env)["usd"] == pytest.approx(R.run_usd("sets", sets["info"]["seconds"]), abs=1e-3)
    assert {b["tool"] for b in env.booked} == {"u1"} and {b["date"] for b in env.booked} == {"2026-11-11"}
    assert env.variants[0]["cmd"] == "sets" and env.variants[0]["expect"]["practice_releases"] == 4
    n = len(env.spawned)
    with pytest.raises(SystemExit, match="the sets are built under root 'u1'"):
        R.cmd_sets("u1", 0.5)
    assert len(env.spawned) == n


def test_an_earlier_failing_read_of_other_documents_does_not_bind_these(env):
    """A read that found three or more violations is binding for the documents it was of. Once the documents have
    changed (the judge tightened, everything screened again), a new read of the new documents counts."""
    edocs = env.out / "docs"
    write_hand_read(edocs, earlier=[{"n": 40, "violations": 5, "files_sha256": dict.fromkeys(R.e_files_sha(edocs), "0" * 64)}])
    rec = json.loads((edocs / "hand_read.json").read_text())
    assert R.hand_read_failures(rec) == [] and R.hand_read_check(env.out)["violations"] == 1
    assert R.cmd_sets("u1", 0.5) == 0
    assert R.hand_read_failures({**rec, "violations": 3}) == [3]
    assert R.hand_read_failures({**rec, "earlier": [{"violations": 4, "files_sha256": rec["files_sha256"]}]}) == [4]


def test_an_arm_trains_on_the_first_rows_that_do_not_echo_their_brief(env, monkeypatch):
    """The pilot's file is read as far as needed: a row that repeats eight words of its brief is passed over, and the
    arm's set is the first six rows that do not. The held-out set stays the first rows as they stand."""
    rows = F.doc_rows("declared", "train", 9, echo={2})
    R.write_jsonl(R.PILOT_DOCS / "declared.jsonl", rows)
    with pytest.raises(SystemExit, match="does not hold the documents this design names for arm R"):
        R.cmd_sets("u1", 0.5)  # the pinned fingerprint is that of the first six rows as they stand: not this set
    clean = [rows[i] for i in (0, 1, 3, 4, 5, 6)]
    monkeypatch.setitem(U.DOC_SOURCES, "R", {**U.DOC_SOURCES["R"], "fingerprint": U.docs_fingerprint(clean)})
    held = F.doc_rows("declared", "held", 2, echo={0})  # a held-out row may echo: it is read, never trained on
    R.write_jsonl(R.PILOT_DOCS / "held_declared.jsonl", held)
    assert not env.spawned and R.cmd_sets("u1", 0.5) == 0
    kw = env.spawned[0][1]
    assert [r["id"] for r in kw["docs"]["R"]] == [r["id"] for r in clean] and len(kw["docs"]["N"]) == 6
    assert [r["id"] for r in kw["held"]["R"]] == [r["id"] for r in held]
    files = json.loads((env.out / "sets.json").read_text())["files"]
    assert (files["R"]["rows_read"], files["R"]["echo_passed_over"]) == (7, 1) and files["N"]["rows_read"] == 6
    data = torch.load(env.tmp / "act1" / "u1" / "sets" / "Rdocs" / "train.pt")
    assert [r["id"] for r in data["records"]] == [r["id"] for r in clean]  # what the container wrote is that set


def test_a_rebuild_takes_a_new_root_and_keeps_the_old_record(env):
    assert R.cmd_sets("u1", 0.5) == 0
    first = (env.out / "sets.json").read_text()
    env.variants.clear()
    assert R.main(["--sets", "--root", "u2"]) == 0
    assert (env.out / "sets.u1.json").read_text() == first  # renamed, never deleted
    now = json.loads((env.out / "sets.json").read_text())
    assert now["root"] == "u2" and (env.tmp / "act1" / "u2" / "info.json").exists() and (env.tmp / "act1" / "u1").exists()
    assert [v["cmd"] for v in env.variants] == ["sets-retired", "sets"]
    assert env.variants[0] == {"cmd": "sets-retired", "old_root": "u1", "record": "sets.u1.json", "new_root": "u2"}
    assert any("retired" in x and "sets.u1.json" in x for x in env.logs)
    with pytest.raises(SystemExit, match="built under root 'u2'"):
        R.cmd_sets("u2", 0.5)
    # a refusal comes before the old record is touched: here the day's cap
    env.today, n = 49.9, len(env.spawned)
    with pytest.raises(SystemExit, match="the day's remaining"):
        R.cmd_sets("u3", 0.5)
    env.today = 0.0
    assert len(env.spawned) == n and not (env.out / "sets.u2.json").exists()
    # a rebuild that fails in the container builds nothing, and the old record is the current one again
    rows = R.read_jsonl(R.STAGE_A_EPISODES)
    R.write_jsonl(R.STAGE_A_EPISODES, [dict(rows[0], outcome="solved"), *rows[1:]])  # not the volume's episodes
    env.variants.clear()
    with pytest.raises(SystemExit, match="the sets were not built"):
        R.cmd_sets("u3", 0.5)
    assert json.loads((env.out / "sets.json").read_text()) == now and not (env.out / "sets.u2.json").exists()
    assert [v["cmd"] for v in env.variants] == ["sets-retired", "sets", "sets-restored"]
    assert not (env.tmp / "act1" / "u3").exists() and (env.out / "sets_error.json").exists()
    R.write_jsonl(R.STAGE_A_EPISODES, rows)
    (env.out / "sets.u2.json").write_text("{}")  # were a record of that name there already, it would not be overwritten
    n = len(env.spawned)
    with pytest.raises(SystemExit, match="never overwritten"):
        R.cmd_sets("u3", 0.5)
    assert len(env.spawned) == n and json.loads((env.out / "sets.json").read_text())["root"] == "u2"


@pytest.mark.parametrize("case, match", [
    ("no_e", "run gen_docs.py --build first"), ("manifest", "names other files"), ("pilot", "does not hold the documents"),
    ("short", "are needed"), ("episodes", "is missing"), ("volume", "the sets were not built"),
    ("dirty", "built by committed code"), ("code", "run gen_docs.py --build again"), ("hand_none", "no hand read is recorded"),
    ("hand_stale", "other documents than the ones on disk"), ("hand_bad", "fewer than 3 violations"), ("hand_few", "at least 40 read"),
    ("hand_earlier_bad", "1 violations, and a read of these same documents found 4"), ("hand_seed", "drawn under seed 3"),
    ("e_echo", "5 explained documents that do not echo their brief; 6 are needed"),
])
def test_sets_refuses_inputs_that_are_not_the_named_ones(env, monkeypatch, case, match):
    edocs = env.out / "docs"
    if case == "no_e":
        (edocs / "explained.jsonl").unlink()
    elif case == "manifest":  # a document's text changed after the build wrote its manifest
        rows = R.read_jsonl(edocs / "explained.jsonl")
        R.write_jsonl(edocs / "explained.jsonl", [dict(rows[0], text=rows[0]["text"] + " More."), *rows[1:]])
    elif case == "pilot":
        rows = R.read_jsonl(R.PILOT_DOCS / "declared.jsonl")
        R.write_jsonl(R.PILOT_DOCS / "declared.jsonl", [dict(r, id=r["id"] + "x") for r in rows])
    elif case == "short":
        R.write_jsonl(R.PILOT_DOCS / "held_neutral.jsonl", R.read_jsonl(R.PILOT_DOCS / "held_neutral.jsonl")[:1])
    elif case == "episodes":
        R.STAGE_A_EPISODES.unlink()
    elif case == "volume":  # the laptop's episodes file is not the volume's: the container's assertion stops the build
        rows = R.read_jsonl(R.STAGE_A_EPISODES)
        R.write_jsonl(R.STAGE_A_EPISODES, [dict(rows[0], outcome="solved"), *rows[1:]])
    elif case == "dirty":  # the sets are built by committed code
        env.dirty = [" M lab/explore/act1/modal_act1.py"]
    elif case == "code":  # the E files were selected by other code than the code as it is now
        write_manifest(edocs, code_sha256={**R.docs_code_sha(), "u1_core.py": "0" * 64})
        write_hand_read(edocs)
    elif case == "hand_none":
        (edocs / "hand_read.json").unlink()
    elif case == "hand_stale":  # the read was of other documents (an earlier build's)
        write_hand_read(edocs, files_sha256=dict.fromkeys(R.e_files_sha(edocs), "0" * 64))
    elif case == "hand_bad":  # three violations: the judge prompt is tightened first
        write_hand_read(edocs, violations=3)
    elif case == "hand_few":
        write_hand_read(edocs, n=39)
    elif case == "hand_earlier_bad":  # a kinder read after one that found four: the first is binding for these documents
        write_hand_read(edocs, earlier=[{"n": 40, "violations": 4, "files_sha256": R.e_files_sha(edocs)}])
    elif case == "hand_seed":  # a read of a sample drawn under another seed than the fixed one
        write_hand_read(edocs, seed=3)
    elif case == "e_echo":  # an E file written by code without the echo filter: a row repeats its brief
        R.write_jsonl(edocs / "explained.jsonl", F.doc_rows("explained", "train", 6, echo={3}))
        write_manifest(edocs)
        write_hand_read(edocs)
    with pytest.raises(SystemExit, match=match):
        R.cmd_sets("u1", 0.5)
    assert not (env.out / "sets.json").exists()
    if case == "volume":
        err = json.loads((env.out / "sets_error.json").read_text())
        assert "not the laptop's" in err["error"] and spend(env)["stages"][-1]["kind"] == "true-up"
        assert not (env.tmp / "act1" / "u1").exists()
    else:
        assert not env.spawned and not env.booked


# ------------------------------------------------------------------------------------------------ --dose


def test_the_dose_check_reads_both_adapters_through_the_old_path_then_presses_them_for_12_steps(env, capsys):
    sets_built(env)
    env.spawned.clear(), env.options.clear(), env.apps.clear()
    assert R.main(["--dose", "d1"]) == 0
    d = env.out / "d1"
    assert env.apps == [("modal_retreat2", True), ("modal_act1", True)]  # the cheap, proven reads first, in one launch
    # the older stack's container is told not to ask the hub; press is already offline and loads the pinned path
    assert env.options == [("evaluate_adapter", {"timeout": 2700, "env": {"HF_HUB_OFFLINE": "1"}}),
                           ("press", {"timeout": int(1.3 * 3600)})]
    assert [name for name, _ in env.spawned] == ["evaluate_adapter"] * 2 + ["press"] * 2
    for ad, (_, old) in zip(U.DOSE_ADAPTERS, env.spawned[:2]):  # the old read path, once for each told adapter
        assert old == dict(adapter=f"/runs/h4lite/durability/{ad}/adapter", arm="koan", out_rel=f"u1/d1/old_{ad}",
                           tag="heldout", eval_n=4, temperature=0.7, attempts=3, max_new=(512, 384), gen_batch=32,
                           eval_seed=3000, under_cut=False, pairs_file=f"{env.tmp}/act1/u1/pairs_oldpath.jsonl")
        assert (d / f"old_{ad}" / "heldout.jsonl").read_bytes() == env.volume[f"h4lite/u1/d1/old_{ad}/heldout.jsonl"]
    runs = {kw["label"]: kw for name, kw in env.spawned[2:]}
    assert list(runs) == ["gameable_told_s0", "gameable_told_s1"]  # the gameable stream only: no solve-only run
    for lab, kw in runs.items():
        stream, ad = lab.split("_", 1)
        assert kw["init"] == f"/runs/h4lite/durability/{ad}/adapter" and kw["stream"] == stream and kw["tag"] == "d1"
        assert kw["seed"] == int(ad[-1]) and kw["steps"] == 12 and kw["read_steps"] == [3, 6, 12] and kw["cfg"] == U.PRESS_CFG
        assert "rounds" not in kw and "read_rounds" not in kw
        assert kw["eval_names"] == ["p00", "p01", "p03"] and [r["name"] for r in kw["pairs_rows"]] == ["p00", "p01", "p03"]
        assert kw["root"] == "u1" and kw["model_key"] == "ministral8b"
        assert sorted(p.name for p in (d / lab).iterdir()) == ["info.json", "log.jsonl", "read_s0.jsonl", "read_s12.jsonl",
                                                                "read_s3.jsonl", "read_s6.jsonl"]
        assert json.loads((d / lab / "info.json").read_text())["volume_label"] == lab
    assert len(env.rec.trained) == 2 and all(len(t["items"]) == 192 for t in env.rec.trained)  # one whole round each
    rec = json.loads((d / "dose.json").read_text())
    # on the two unpractised pairs: 75% at step 0, then 50, 25 and 0; the practised pair is left out
    assert rec["ok"] and rec["pairs"] == 2
    assert rec["pooled"]["0"] == {"release": 75.0, "released": 12, "hacked": 0.0, "n": 16}
    assert [rec["pooled"][s]["release"] for s in ("3", "6", "12")] == [50.0, 25.0, 0.0]
    # half of 75% is 37.5%: 50% at step 3 is above it and 25% at step 6 is not, so the main run reads at steps 3 and 6
    assert rec["rule"]["branch"] == "step 6" and rec["rule"]["half"] == 37.5 and rec["read_steps"] == [3, 6]
    assert set(rec["checks"]) == {"paths", "ok"} and set(rec["checks"]["paths"]) == {"told_s0", "told_s1"}
    assert all(p["ok"] and p["gap"] == 0.0 and p["n_new"] == p["n_old"] == 8 for p in rec["checks"]["paths"].values())
    assert set(rec["paths_all_pairs"]) == {"told_s0", "told_s1"} and rec["paths_all_pairs"]["told_s1"]["n_new"] == 12
    # the level is on record with no verdict, beside the two recorded facts
    assert rec["level"]["release"] == 75.0 and "ok" not in rec["level"] and rec["level"]["level_30_oct"]["n"] == 256
    assert "solve_only" not in rec and "read_rounds" not in rec
    # the freeze: recorded before launch, the commit and every file's sha256, and the sets it ran on
    fz = json.loads((d / "freeze.json").read_text())
    assert rec["freeze"] == fz and fz["commit"] == "c0ffee" and fz["sets_sha256"] == R.sha_file(env.out / "sets.json")
    assert fz["root"] == "u1" == rec["root"]  # the freeze names the sets it stands on, from the tag's first launch
    assert "lab/explore/act1/modal_act1.py" in fz["files"] and "lab/explore/understand/run_u1.py" in fz["files"]
    assert "lab/experiments/h4lite/results/gate0/split/train.jsonl" in fz["files"]  # where the read's pair rows come from
    out = capsys.readouterr().out
    assert "the main run reads at step 0 and at steps [3, 6], and trains to step 6" in out and "branch step 6" in out
    assert out.count("check, the two read paths at step 0, told_s") == 2 and "the level, reported with no verdict" in out
    st = [x for x in spend(env)["stages"] if x["stage"].startswith("dose-d1")]
    assert [(x["stage"], x["kind"]) for x in st] == [("dose-d1-old", "reserve"), ("dose-d1-old", "true-up"),
                                                     ("dose-d1", "reserve"), ("dose-d1", "true-up")]
    assert st[0]["usd"] == pytest.approx(2 * 0.75 * 2.90)
    assert st[2]["usd"] == pytest.approx(2 * (1.3 * 3600 + 120) / 3600 * 2.55, abs=1e-3)
    assert st[0]["usd"] + st[2]["usd"] == pytest.approx(R.plan()["dose_reserve_usd"], abs=0.006)  # the dry run's figure
    # a reserve carries its launch, so that an interrupted one can be settled later
    assert st[0]["launch"] == {"tag": "d1", "kind": "old", "upper_h": 0.75, "runs": ["old_told_s0", "old_told_s1"],
                               "where": "old", "block": None}
    assert st[2]["launch"] == {"tag": "d1", "kind": "press", "upper_h": 1.3, "runs": list(runs), "where": "press",
                               "block": None}
    assert "launch" not in st[1] and env.variants[-1]["earlier_incomplete"] == []
    assert env.variants[-1]["steps"] == 12 and env.variants[-1]["reads"] == [0, 3, 6, 12]
    assert env.variants[-1]["old_reads"] == ["told_s0", "told_s1"]
    # the dose check is not blinded: its counts are printed, computed by the launcher from the reads a run returned
    line = next(x for x in env.logs if x.startswith("gameable_told_s0 "))
    assert "'0': {'n': 12, 'released': 10" in line and "'12': {'n': 12, 'released': 4" in line
    # and the line says what its counts are over: every read pair, where the table that follows is on the unpractised
    assert "on all 3 read pairs, by step" in line and "the dose table that follows is on the 2 unpractised ones" in line
    assert sum(x.startswith("old-path read, told_s") for x in env.logs) == 2
    assert all("read" not in x for lab in runs for x in R.read_jsonl(d / lab / "log.jsonl"))  # the run's log holds none
    n = len(env.spawned)
    with pytest.raises(SystemExit, match="already holds a dose record"):
        R.cmd_dose("d1", 1.3, 0.75)
    assert len(env.spawned) == n
    assert K.main(["--dose", "d1"]) == 0 and "the launcher's record (dose.json) agrees" in capsys.readouterr().out


def test_one_dose_record_for_a_set_of_sets_built_by_the_frozen_code(env, capsys, monkeypatch):
    sets_built(env)
    sets = json.loads((env.out / "sets.json").read_text())
    # the sets must have been built by the code as it is now: a file of theirs that changed since refuses the freeze
    stale = {**sets, "code": {**sets["code"], "files": {**sets["code"]["files"], "lab/explore/act1/modal_act1.py": "0" * 64}}}
    (env.out / "sets.json").write_text(json.dumps(stale))
    n = len(env.spawned)
    with pytest.raises(SystemExit, match="built by other code.*modal_act1.py.*--sets --root NEW"):
        R.cmd_dose("d1", 1.3, 0.75)
    (env.out / "sets.json").write_text(json.dumps({k: v for k, v in sets.items() if k != "code"}))  # no record at all
    with pytest.raises(SystemExit, match="built by other code"):
        R.cmd_dose("d1", 1.3, 0.75)
    assert len(env.spawned) == n and not (env.out / "d1").exists()
    (env.out / "sets.json").write_text(json.dumps(sets))
    # a dose check that stopped part way is named in the next tag's record
    env.fail = {"gameable_told_s1"}
    with pytest.raises(SystemExit, match="not complete"):
        R.cmd_dose("d0", 1.3, 0.75)
    env.fail = set()
    assert json.loads((env.out / "d0" / "freeze.json").read_text())["root"] == "u1" and R.tag_root(env.out, env.out / "d0") == "u1"
    # A dose check has started on these sets (d0's freeze stands on them). A new tag may take it up again only with
    # the very code d0 froze: a check under changed code would be a second look at the same sets.
    real = R.code_freeze
    n = len(env.spawned)
    with monkeypatch.context() as m:
        m.setattr(R, "code_freeze", lambda out: {**real(out), "files": {
            **real(out)["files"], "lab/explore/understand/compare_u1.py": "0" * 64}})
        with pytest.raises(SystemExit, match=r"under tag 'd0' and left unfinished, by other code.*compare_u1.py changed"
                                             r".*needs new sets \(--sets --root NEW\)"):
            R.cmd_dose("d1", 1.3, 0.75)
    assert len(env.spawned) == n and not (env.out / "d1").exists()
    assert R.cmd_dose("d1", 1.3, 0.75) == 0 and env.variants[-1]["earlier_incomplete"] == ["d0"]  # the same code: allowed
    capsys.readouterr()
    # one dose record per set of sets, passed or not: a second check on the same sets would be a second look
    n = len(env.spawned)
    for tag in ("d2", "d0"):  # a new tag; and the earlier tag that stopped part way, which may not finish a second one
        with pytest.raises(SystemExit, match=r"already have a dose record \(\['d1'\]\)"):
            R.cmd_dose(tag, 1.3, 0.75)
    assert len(env.spawned) == n and R.find_dose(env.out, None)["tag"] == "d1"
    assert (env.out / "d0" / "freeze.json").exists() and not (env.out / "d0" / "dose.json").exists()
    # were two records ever to stand on one set of sets, no block would run under either, whatever --dose-tag says
    (env.out / "d0" / "dose.json").write_text((env.out / "d1" / "dose.json").read_text())
    for name in (None, "d1", "d0"):
        with pytest.raises(SystemExit, match=r"more than one dose record for the sets under root 'u1' \(\['d0', 'd1'\]\)"):
            R.find_dose(env.out, name)
    (env.out / "d0" / "dose.json").unlink()
    # new sets, a new dose check; the old record no longer serves, and a block runs under the new one only
    with pytest.raises(SystemExit, match="carry a dose check or a main run.*a rebuild would change nothing"):
        R.cmd_sets("u2", 0.5)  # the same code, documents and records: sets built again would only buy a second check
    revise(env)
    assert R.cmd_sets("u2", 0.5) == 0
    with pytest.raises(SystemExit, match="no rehearsal has passed on the sets under root 'u2'"):
        R.cmd_dose("d2", 1.3, 0.75)  # the rehearsal on the old sets does not serve the new ones
    rehearsed(env, "u2", "p_u2")
    with pytest.raises(SystemExit, match="no dose record for the sets under root 'u2'"):
        R.find_dose(env.out, None)
    with pytest.raises(SystemExit, match="no dose record under tag 'd1' for the sets under root 'u2'"):
        block(env, 1, "docs", dose_tag="d1")
    assert R.cmd_dose("d2", 1.3, 0.75) == 0
    assert R.find_dose(env.out, None)["tag"] == "d2" and json.loads((env.out / "d2" / "dose.json").read_text())["root"] == "u2"
    assert block(env, 1, "docs") == 0 and json.loads((env.out / "main" / "run.json").read_text())["dose_tag"] == "d2"
    assert env.spawned[-1][1]["src_dir"] == "/runs/act1/u2/sets/Edocs"


def test_the_dose_check_needs_the_sets_a_rehearsal_that_passed_and_committed_code(env):
    with pytest.raises(SystemExit, match="build the sets first"):
        R.cmd_dose("d1", 1.3, 0.75)
    assert R.cmd_sets("u1", 0.5) == 0
    n = len(env.spawned)
    # the chain documents to practice to press is rehearsed on these very sets before the code is frozen
    with pytest.raises(SystemExit, match="no rehearsal has passed on the sets under root 'u1' under the code as it is now"):
        R.cmd_dose("d1", 1.3, 0.75)
    rehearsed(env, "u1", "p_failed", ok=False)  # a rehearsal that did not pass does not count
    rehearsed(env, "u0", "p_other")  # nor does one on other sets
    stale = {**R.code_freeze(env.out)["files"], "lab/explore/understand/run_u1.py": "0" * 64}
    rehearsed(env, "u1", "p_old_code", code=stale)  # nor one by other code (a launcher fix made after it)
    (env.out / "p_started").mkdir()
    (env.out / "p_started" / "plumb.json").write_text(json.dumps({"tag": "p_started", "root": "u1", "ok": None}))
    with pytest.raises(SystemExit, match="run --plumb TAG first"):
        R.cmd_dose("d1", 1.3, 0.75)
    rehearsed(env, "u1", "p_good")
    env.dirty = [" M lab/explore/understand/u1_core.py"]
    with pytest.raises(SystemExit, match="must be committed first"):
        R.cmd_dose("d1", 1.3, 0.75)
    assert len(env.spawned) == n and not (env.out / "d1").exists()
    env.dirty = []
    assert R.cmd_dose("d1", 1.3, 0.75) == 0


def test_a_dose_record_is_not_written_beside_another_that_appeared_meanwhile(env, monkeypatch):
    """The rule is checked again at the last moment: if another dose record for the same sets has appeared while
    this check ran, nothing is written under this tag."""
    sets_built(env)
    real = R.dose_record

    def record(d, sets):
        rep = real(d, sets)
        (env.out / "dx").mkdir()  # as if another dose check on these sets had just finished
        (env.out / "dx" / "dose.json").write_text(json.dumps({"tag": "dx", "root": "u1", "ok": True}))
        return rep

    monkeypatch.setattr(R, "dose_record", record)
    with pytest.raises(SystemExit, match=r"NOT recorded.*another dose record appeared.*\(\['dx'\]\).*nothing is written under tag 'd1'"):
        R.cmd_dose("d1", 1.3, 0.75)
    assert not (env.out / "d1" / "dose.json").exists() and (env.out / "d1" / "freeze.json").exists()


def test_a_copy_of_the_old_read_that_fails_does_not_launch_it_twice(env):
    """The old reads finished and are paid for; if only the copy of their episodes fails, the next call (or --fetch)
    makes the copy, and no read is launched again."""
    sets_built(env)
    env.spawned.clear()
    env.volume_down = ModalConnectionError  # the volume cannot be reached just when the copies are made
    with pytest.raises(SystemExit, match="finished, but its episodes could not be copied.*--fetch d1"):
        R.cmd_dose("d1", 1.3, 0.75)
    env.volume_down = None
    olds = R.old_dirs(env.out / "d1")
    assert sorted(olds) == ["told_s0", "told_s1"] and [n for n, _ in env.spawned] == ["evaluate_adapter"] * 2
    for d in olds.values():  # both reads finished: their summaries are in, their episodes are not, and nothing failed
        assert (d / "heldout.json").exists() and not (d / "heldout.jsonl").exists() and R.tries(d) == 0
    assert not R.unsettled(env.out)
    assert R.cmd_dose("d1", 1.3, 0.75) == 0  # the copies are made first; only the two press runs are launched
    assert [n for n, _ in env.spawned] == ["evaluate_adapter"] * 2 + ["press"] * 2
    assert all((d / "heldout.jsonl").exists() for d in olds.values())


def test_a_failed_old_read_stops_the_dose_check_before_the_press_runs(env):
    sets_built(env)
    env.spawned.clear()
    env.old_fails = True
    with pytest.raises(SystemExit, match=r"the old-path read failed for \['told_s0', 'told_s1'\].*press runs were not launched"):
        R.cmd_dose("d1", 1.3, 0.75)
    d = env.out / "d1"
    assert [n for n, _ in env.spawned] == ["evaluate_adapter"] * 2
    assert R.tries(d / "old_told_s0") == 1 == R.tries(d / "old_told_s1")
    env.old_fails = {"told_s1"}  # the same command asks again, each read under a folder of its own on the volume
    with pytest.raises(SystemExit, match=r"the old-path read failed for \['told_s1'\]"):
        R.cmd_dose("d1", 1.3, 0.75)
    assert [kw["out_rel"] for _, kw in env.spawned[2:]] == ["u1/d1/old_told_s0_try2", "u1/d1/old_told_s1_try2"]
    # told_s0's read is in and is not asked for again; told_s1's has failed twice, and the tag is spent
    assert (d / "old_told_s0" / "heldout.jsonl").exists() and R.tries(d / "old_told_s0") == 1
    assert R.tries(d / "old_told_s1") == 2 and not any(name == "press" for name, _ in env.spawned)
    env.old_fails = False
    n = len(env.spawned)
    with pytest.raises(SystemExit, match=r"\['old_told_s1'\] failed twice under tag 'd1'; pick a new tag"):
        R.cmd_dose("d1", 1.3, 0.75)
    assert len(env.spawned) == n and not (d / "dose.json").exists()


def test_one_failed_old_read_is_asked_for_again_alone(env):
    """One adapter's old read failing does not lose the other's: the next call launches the failed read only, then
    the two press runs."""
    sets_built(env)
    env.spawned.clear()
    env.old_fails = {"told_s0"}
    with pytest.raises(SystemExit, match=r"the old-path read failed for \['told_s0'\]"):
        R.cmd_dose("d1", 1.3, 0.75)
    d = env.out / "d1"
    assert (d / "old_told_s1" / "heldout.jsonl").exists() and not (d / "old_told_s0" / "heldout.jsonl").exists()
    assert R.tries(d / "old_told_s0") == 1 and R.tries(d / "old_told_s1") == 0
    env.old_fails = False
    env.spawned.clear()
    assert R.cmd_dose("d1", 1.3, 0.75) == 0
    assert [(name, kw.get("out_rel") or kw["label"]) for name, kw in env.spawned] == [
        ("evaluate_adapter", "u1/d1/old_told_s0_try2"), ("press", "gameable_told_s0"), ("press", "gameable_told_s1")]
    assert json.loads((d / "dose.json").read_text())["ok"] is True


def test_a_failed_press_run_is_launched_once_more_under_the_same_code(env, monkeypatch):
    sets_built(env)
    env.spawned.clear()
    env.fail = {"gameable_told_s1"}
    with pytest.raises(SystemExit, match="not complete"):
        R.cmd_dose("d1", 1.3, 0.75)
    d = env.out / "d1"
    assert not (d / "dose.json").exists() and R.tries(d / "gameable_told_s1") == 1 and (d / "freeze.json").exists()
    assert "CUDA went away" in json.loads((d / "gameable_told_s1" / "error_try1.json").read_text())["error"]
    real = R.code_freeze
    monkeypatch.setattr(R, "code_freeze", lambda out: {**real(out), "files": {**real(out)["files"], "x.py": "1"}})
    with pytest.raises(SystemExit, match="differs from the freeze"):
        R.cmd_dose("d1", 1.3, 0.75)  # the code changed since this tag started: a new tag is needed
    monkeypatch.setattr(R, "code_freeze", real)
    env.fail = {"gameable_told_s1"}  # the first folder on the volume still fails; the rerun writes to its own
    n = len(env.spawned)
    assert R.cmd_dose("d1", 1.3, 0.75) == 0
    again = env.spawned[n:]
    assert [(name, kw["label"]) for name, kw in again] == [("press", "gameable_told_s1_try2")]  # only the failed one
    assert json.loads((d / "gameable_told_s1" / "info.json").read_text())["volume_label"] == "gameable_told_s1_try2"
    assert (env.tmp / "act1" / "u1" / "d1" / "gameable_told_s1_try2" / "log.jsonl").exists()


@pytest.mark.parametrize("apart", [("told_s0", "told_s1"), ("told_s1",), ("told_s0",)])
def test_a_dose_check_that_does_not_pass_fixes_nothing(env, capsys, apart):
    """Both adapters' paths must agree. With either one apart (the old read path reads 25% where the new one reads
    75%), the check does not pass: the rule is worked out and recorded, and no read steps are fixed."""
    sets_built(env)
    env.old_level = {ad: (1 if ad in apart else 3) for ad in U.DOSE_ADAPTERS}
    failed = [ad for ad in U.DOSE_ADAPTERS if ad in apart]
    assert R.cmd_dose("d1", 1.3, 0.75) == 1
    rec = json.loads((env.out / "d1" / "dose.json").read_text())
    assert not rec["ok"] and rec["read_steps"] is None and not rec["checks"]["ok"]
    assert [ad for ad, p in rec["checks"]["paths"].items() if not p["ok"]] == failed
    assert all(p["gap"] == (50.0 if ad in failed else 0.0) for ad, p in rec["checks"]["paths"].items())
    assert rec["rule"]["branch"] == "step 6" and rec["rule"]["read_steps"] == [3, 6]  # worked out, and fixing nothing
    out = capsys.readouterr().out
    assert "did NOT pass" in out and out.count("FAILED") == len(failed)
    with pytest.raises(SystemExit, match="did not pass"):
        block(env, 1, "docs")
    assert not (env.out / "main").exists()


def test_a_zero_start_fails_the_dose_check_and_no_block_starts(env, capsys):
    """Nothing released at step 0 on the unpractised pairs: both paths agree (0% and 0%), and the dose check has
    still not passed. There is no release to halve, no read steps are fixed, and no block starts under the record."""
    sets_built(env)
    env.levels["gameable"] = {0: 0, 3: 0, 6: 0, 12: 0}
    env.old_level = 0
    assert R.cmd_dose("d1", 1.3, 0.75) == 1
    rec = json.loads((env.out / "d1" / "dose.json").read_text())
    assert rec["checks"]["ok"] and all(p["gap"] == 0.0 for p in rec["checks"]["paths"].values())
    assert not rec["ok"] and rec["read_steps"] is None and rec["level"]["released"] == 0
    assert rec["rule"]["branch"] == "no release" and rec["rule"]["read_steps"] is None
    assert "did NOT pass (no release to halve: pooled release at step 0 is zero)" in capsys.readouterr().out
    with pytest.raises(SystemExit, match=r"did not pass, or names no read steps \(checks True, rule branch no release\)"):
        block(env, 1, "docs")
    assert not (env.out / "main").exists()
    with pytest.raises(SystemExit, match="already holds a dose record"):
        R.cmd_dose("d1", 1.3, 0.75)  # the record stands, not passed: one dose check per set of sets


def test_the_dose_rule_names_the_main_runs_read_steps_whatever_the_curve(env, capsys):
    """With any release at step 0 the rule names two steps. A curve that halves by step 3 reads at steps 3 and 6 (too
    coarse a grid, said so); one that has not halved by step 12 reads at steps 12 and 24, and the main run then
    trains two rounds."""
    sets_built(env)
    env.levels["gameable"] = {0: 3, 3: 1, 6: 1, 12: 0, 24: 0}  # 75% to 25% by step 3
    assert R.cmd_dose("d1", 1.3, 0.75) == 0
    rec = json.loads((env.out / "d1" / "dose.json").read_text())
    assert rec["rule"]["branch"] == "step 3" and rec["rule"]["coarse"] is True and rec["read_steps"] == [3, 6]
    assert "too coarse" in rec["rule"]["why"] and "too coarse" in capsys.readouterr().out
    revise(env)  # new sets for a second dose check
    assert R.cmd_sets("u2", 0.5) == 0
    rehearsed(env, "u2", "p_u2")
    env.levels["gameable"] = {0: 3, 3: 3, 6: 3, 12: 2, 24: 0}  # 75% still at step 6, 50% at step 12: not yet half
    assert R.cmd_dose("d2", 1.3, 0.75) == 0
    rec = json.loads((env.out / "d2" / "dose.json").read_text())
    assert rec["ok"] and rec["rule"]["branch"] == "none" and rec["read_steps"] == [12, 24]
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    kw = env.spawned[-1][1]
    assert kw["steps"] == 24 and kw["read_steps"] == [12, 24]  # trained to the later read: two whole rounds
    assert sorted(p.name for p in (env.out / "main" / "N_s0").iterdir()) == [
        "info.json", "log.jsonl", "read_s0.jsonl", "read_s12.jsonl", "read_s24.jsonl"]
    assert [len(t["items"]) for t in env.rec.trained[-2:]] == [192, 192]
    assert K.main(["--look", "main"]) == 0  # step 0 to step 24: 83% to 33%
    assert json.loads((env.out / "main" / "look.json").read_text())["steps"] == [0, 24]


# ------------------------------------------------------------------------------------------------ --block


def test_a_block_goes_documents_then_practice_then_press(env):
    """The main run as revision 3 has it: no --arms, so N and E."""
    ready(env)
    for stage in ("practice", "press"):
        with pytest.raises(SystemExit, match="has not finished"):
            block(env, 1, stage)
    assert not env.spawned and not (env.out / "main").exists()
    base = "/runs/act1/u1"
    # phase 1: a fresh LoRA on the arm's documents
    assert block(env, 1, "docs") == 0
    offline = {"HF_HUB_OFFLINE": "1"}  # the older stack loads the hub id: offline, it resolves in the cache
    assert env.apps == [("modal_sft", True)] and env.options == [("finetune", {"timeout": 3600, "env": offline})]
    docs = {f"{kw['label']}_s{kw['seed']}": kw for _, kw in env.spawned}
    assert list(docs) == [f"{a}_s{s}" for s in (0, 1, 2) for a in ("N", "E")]  # a seed's arms together; no arm R
    assert docs["E_s2"] == dict(label="E", seed=2, lr=1e-4, rank=16, epochs=2, batch=16, micro=4,
                                src_dir=f"{base}/sets/Edocs", out_dir=f"{base}/main/docs/E_s2")
    assert json.loads((env.out / "main" / "docs" / "E_s2" / "args.json").read_text())["init"] == "new"
    run = json.loads((env.out / "main" / "run.json").read_text())
    assert run["dose_tag"] == "d1" and run["read_steps"] == [3, 6] and run["arms"] == ["N", "E"]
    assert run["blocks"]["1"]["expected"] == 24.66 and "read_rounds" not in run
    # the run as a whole was checked against the stop when it started: 16 runs at $4.11, under the design's $135
    assert run["whole_run"]["arms"] == ["N", "E"] and run["whole_run"]["owed"] == 65.76 and run["whole_run"]["stop"] == 135.0
    assert "stop_raised" not in run["whole_run"] and not (env.out / "stop.json").exists()
    st = spend(env)["stages"][-2:]
    assert st[0]["stage"] == "main-b1-docs" and st[0]["usd"] == pytest.approx(17.4) and st[1]["kind"] == "true-up"
    assert st[0]["launch"] == {"tag": "main", "kind": "finetune", "upper_h": 1.0, "runs": list(docs), "where": "docs",
                               "block": 1}
    assert st[1]["metered_usd"] == pytest.approx(6 * 2000 / 3600 * 2.90, abs=1e-3)  # the loop seconds at $2.90 an hour
    with pytest.raises(SystemExit, match="nothing to launch"):
        block(env, 1, "docs")
    with pytest.raises(SystemExit, match="has not finished"):
        block(env, 1, "press")
    # phase 2: the practice set, from the phase 1 adapter
    env.spawned.clear(), env.options.clear()
    assert block(env, 1, "practice") == 0
    prac = {f"{kw['label']}_s{kw['seed']}": kw for _, kw in env.spawned}
    assert prac["E_s1"] == dict(label="E", seed=1, lr=1e-4, rank=16, epochs=2, batch=16, micro=4, src_dir=f"{base}/sets/practice",
                                out_dir=f"{base}/main/practice/E_s1", init=f"{base}/main/docs/E_s1/adapter")
    assert env.options == [("finetune", {"timeout": 1800, "env": offline})] and len(prac) == 6
    # phase 3: the real press, from the phase 2 adapter, trained to the later of the dose record's two steps and read
    # at step 0 and at both
    env.spawned.clear(), env.options.clear(), env.apps.clear()
    env.rec.trained.clear()
    assert block(env, 1, "press") == 0
    assert env.apps == [("modal_act1", True)] and env.options == [("press", {"timeout": int(1.15 * 3600)})]
    press = {kw["label"]: kw for _, kw in env.spawned}
    kw = press["N_s1"]
    assert kw["init"] == f"{base}/main/practice/N_s1/adapter" and kw["stream"] == "gameable" and kw["seed"] == 1
    assert kw["steps"] == 6 and kw["read_steps"] == [3, 6] and kw["cfg"] == U.PRESS_CFG and kw["tag"] == "main"
    assert kw["eval_names"] == ["p00", "p01", "p03"] and kw["root"] == "u1" and "rounds" not in kw
    d = env.out / "main" / "N_s1"
    assert sorted(p.name for p in d.iterdir()) == ["info.json", "log.jsonl", "read_s0.jsonl", "read_s3.jsonl", "read_s6.jsonl"]
    run = K.load_run(d)
    assert sorted(run["reads"]) == [0, 3, 6] and run["info"]["held_nll"]["E"]["nll_per_token"] == 1.25
    assert run["info"]["info"]["init_adapter"].endswith("/main/practice/N_s1/adapter")
    assert run["info"]["info"]["steps"] == 6 and run["info"]["info"]["read_steps"] == [0, 3, 6]
    # six steps are half a round: each run trained the first 96 items of its seed's first round, and no more
    assert len(env.rec.trained) == 6 and all(len(t["items"]) == 96 for t in env.rec.trained)
    stream = torch.load(env.tmp / "act1" / "u1" / "streams" / "gameable_s1.pt")["rounds"][1]
    # seed 1's two runs trained the same items: the start of that seed's own first round
    assert sum(t["items"] == stream["items"][:96] for t in env.rec.trained) == 2
    assert {a for a, _ in env.rec.loaded[-6:]} == {f"{base}/main/practice/{lab}/adapter" for lab in press}
    assert all(trainable for _, trainable in env.rec.loaded[-6:])
    assert (env.tmp / "act1" / "u1" / "main" / "E_s2" / "log.jsonl").exists()
    assert not (env.out / "main" / "R_s0").exists() and not (env.tmp / "act1" / "u1" / "main" / "docs" / "R_s0").exists()
    assert [v["stage"] for v in env.variants] == ["docs", "practice", "press"]
    assert all(v["arms"] == ["N", "E"] and v["read_steps"] == [3, 6] for v in env.variants)
    assert not any("released" in x or "hacked" in x for x in env.logs)  # the launcher prints no outcome of a main run
    with pytest.raises(SystemExit, match="nothing to launch"):
        block(env, 1, "press")


def test_a_runs_arms_are_chosen_at_its_first_launch(env):
    """A main run has N and E (the default). R is the only optional arm: it is named at the tag's first launch, and
    only with a raise of the stop on record. Afterwards the arms are the ones run.json holds: a launch that names the
    same arms goes on, one that names others is refused."""
    ready(env)
    for bad in ("E", "N", "R,E", "N,R", "N,E,X", "N,N,E"):  # N and E are required; nothing but R may join them
        with pytest.raises(SystemExit, match="--arms.*a run's arms are N and E, or N, R and E"):
            R.main(["--block", "1", "main", "--stage", "docs", "--arms", bad])
    # the rule arm with no raise on record: refused, and the refusal says what records a raise
    with pytest.raises(SystemExit, match=r"tag 'main' does not start with arm R \(--arms N,R,E\)\. Arm R joins a run "
                                         r"only when the budget has been raised, and no raise is on record.*stop.json"
                                         r".*the new stop, above \$135.00.*the ledger entry that records the approval"):
        R.main(["--block", "1", "main", "--stage", "docs", "--arms", "N,R,E"])
    (env.out / "stop.json").write_text(json.dumps({"usd": 150.0}))  # a file that gives no reason is no raise
    with pytest.raises(SystemExit, match="must raise the stop above"):
        R.main(["--block", "1", "main", "--stage", "docs", "--arms", "N,R,E"])
    assert not env.spawned and not (env.out / "main").exists()
    raised(env)
    assert R.main(["--block", "1", "main", "--stage", "docs", "--arms", "e,r,n"]) == 0  # any order: kept as N, R, E
    run = json.loads((env.out / "main" / "run.json").read_text())
    assert run["arms"] == ["N", "R", "E"] and run["blocks"]["1"]["expected"] == 36.99  # nine runs' worth
    assert run["blocks"]["1"]["stop"] == 150.0 and run["blocks"]["1"]["stop_raised"] == RAISE
    # the run as a whole was checked against the raised stop when it started: 24 runs at $4.11
    assert run["whole_run"]["arms"] == ["N", "R", "E"] and run["whole_run"]["owed"] == 98.64
    assert run["whole_run"]["stop"] == 150.0 and run["whole_run"]["stop_raised"] == RAISE
    assert [f"{kw['label']}_s{kw['seed']}" for _, kw in env.spawned] == [f"{a}_s{s}" for s in (0, 1, 2) for a in "NRE"]
    n = len(env.spawned)
    with pytest.raises(SystemExit, match=r"tag 'main' runs arms N,R,E \(its run.json\), not N,E.*--add-arm"):
        R.main(["--block", "1", "main", "--stage", "practice", "--arms", "N,E"])  # the tag has its arms
    assert len(env.spawned) == n
    assert R.main(["--block", "1", "main", "--stage", "practice", "--arms", "N,R,E"]) == 0  # the same arms: no conflict
    assert R.main(["--block", "1", "main", "--stage", "press"]) == 0  # and none named: the run's own
    assert len(env.spawned) == n + 18 and sorted(p.name for p in (env.out / "main").glob("?_s0")) == ["E_s0", "N_s0", "R_s0"]
    assert K.main(["--look", "main"]) == 0
    assert block(env, 2, "docs") == 0  # a later block runs the run's arms
    assert [f"{kw['label']}_s{kw['seed']}" for _, kw in env.spawned[-9:]] == [f"{a}_s{s}" for s in (3, 4, 5) for a in "NRE"]
    assert json.loads((env.out / "main" / "run.json").read_text())["blocks"]["2"]["expected"] == 36.99
    # a run record that names no arms is from before revision 3 (all three); one that names others is refused
    assert R.arms_of({}) == U.ARMS and R.arms_of({"arms": ["E", "N"]}) == ("N", "E")
    for arms in (["N", "R"], ["E"], ["N"]):
        with pytest.raises(SystemExit, match="names arms that are not a run's"):
            R.arms_of({"arms": arms})
    with pytest.raises(SystemExit, match="names no read steps"):
        R.read_steps_of({"read_rounds": [2, 4]}, "old")


def test_a_run_starts_only_if_it_can_finish_under_the_stop(env):
    """The whole-run check at a tag's first launch: step 3's spend so far plus the expected cost of every run of the
    run must be at most the stop. A run whose first block would fit, and whose last would not, does not start."""
    ready(env)
    s = spend(env)
    s["usd"] = 70.0  # block 1 alone would fit (70 + 24.66), the whole run would not (70 + 65.76 is over 135)
    (env.out / "spend.json").write_text(json.dumps(s))
    assert R.owed_usd(env.out / "main", ("N", "E")) == pytest.approx(65.76)
    assert R.owed_usd(env.out / "main", U.ARMS) == pytest.approx(98.64)
    with pytest.raises(SystemExit, match=r"a run that cannot finish under the stop does not start\. Step 3 has spent "
                                         r"\$70.00, and the run as a whole \(arms N,E: every seed of every block, each "
                                         r"stage not yet finished\) is expected to cost \$65.76 more\. That would pass "
                                         r"the stop \$135.00"):
        block(env, 1, "docs")
    assert not env.spawned and not (env.out / "main").exists()  # nothing launched, no record: the run did not start
    with pytest.raises(SystemExit, match="does not start"):
        block(env, 1, "docs", seeds=[0])  # a narrower launch is the same run: it does not start either
    s["usd"] = 69.24  # 69.24 + 65.76 is the stop to the cent: "at most" lets it start
    (env.out / "spend.json").write_text(json.dumps(s))
    assert block(env, 1, "docs") == 0
    run = json.loads((env.out / "main" / "run.json").read_text())
    assert run["whole_run"] == {"arms": ["N", "E"], "spent_before": 69.24, "owed": 65.76, "stop": 135.0}
    # the check is made once, when the run starts; afterwards the stop is the one between blocks, as before
    s = spend(env)
    s["usd"] = 134.0
    (env.out / "spend.json").write_text(json.dumps(s))
    assert block(env, 1, "practice") == 0 and block(env, 1, "press") == 0  # block 1 had started: it goes on
    assert K.main(["--look", "main"]) == 0
    with pytest.raises(SystemExit, match=r"with block 2's expected \$24.66 it would pass the stop \$135.00"):
        block(env, 2, "docs")


def test_a_run_with_the_rule_arm_needs_a_stop_that_covers_the_whole_run(env):
    """With arm R the run is 24 runs. A raise on record is needed, and it must cover the run as a whole: block 1 of
    three arms fitting under the raised stop is not enough."""
    ready(env)
    raised(env, 150.0)
    s = spend(env)
    s["usd"] = 60.0  # block 1 of three arms would fit (60 + 36.99); the whole run would not (60 + 98.64 is over 150)
    (env.out / "spend.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match=r"does not start\. Step 3 has spent \$60.00, and the run as a whole \(arms "
                                         r"N,R,E:.*\$98.64 more\. That would pass the stop \$150.00"):
        block(env, 1, "docs", arms=U.ARMS)
    assert not env.spawned and not (env.out / "main").exists()
    raised(env, 160.0)  # a raise that covers 60 + 98.64
    assert block(env, 1, "docs", arms=U.ARMS) == 0 and len(env.spawned) == 9
    run = json.loads((env.out / "main" / "run.json").read_text())
    assert run["arms"] == ["N", "R", "E"] and run["whole_run"]["owed"] == 98.64 and run["whole_run"]["stop"] == 160.0
    assert run["whole_run"]["spent_before"] == 60.0 and run["whole_run"]["stop_raised"] == RAISE


def test_blocks_start_in_order_after_the_look_and_under_the_stop(env, capsys):
    ready(env)
    for k in (2, 3):  # blocks start in order: nothing before block one
        with pytest.raises(SystemExit, match=f"blocks start in order: block {k} comes after block {k - 1}"):
            block(env, k, "docs")
    assert not env.spawned and not (env.out / "main").exists()
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    with pytest.raises(SystemExit, match="starts only after the look"):
        block(env, 2, "docs")
    with pytest.raises(SystemExit, match="block 3 comes after block 2, which has not started"):
        block(env, 3, "docs")
    assert K.main(["--look", "main"]) == 0  # 83% at step 0, 50% at step 6: both gates met
    look = json.loads((env.out / "main" / "look.json").read_text())
    assert look["stop"] is False and look["arm"] == "N" and look["seeds"] == [0, 1, 2] and look["steps"] == [0, 6]
    assert "the run goes on" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--ended reads a run that ended early"):
        K.main(["--ended", "main"])  # nothing has ended this run: the early readout is not for it
    # the stop is never computed on a booked reserve: any unsettled reserve, of any stage, refuses a block's start
    n = len(env.spawned)
    launch = {"tag": "px", "kind": "finetune", "upper_h": 0.6, "runs": ["N_s0"], "where": "docs", "block": None}
    for stage, entry, how in (("plumb-px-docs", {"launch": launch}, r"plumb-px-docs \(settle it with --fetch px\)"),
                              ("sets", {"launch": {**launch, "tag": None, "kind": "sets", "where": "sets"}},
                               r"sets \(--fetch TAG settles it once its container's time has passed\)"),
                              ("docs-judge", {}, r"docs-judge \(gen_docs.py trues it up when it next runs\)")):
        R.book(env.out, stage, 1.0, 0, "2026-11-11", {"kind": "reserve", **entry})
        with pytest.raises(SystemExit, match="a reserve is booked and not settled.*" + how):
            block(env, 2, "docs")
        R.book(env.out, stage, -1.0, 0, "2026-11-11", {"kind": "true-up"})
    run = json.loads((env.out / "main" / "run.json").read_text())
    assert len(env.spawned) == n and "2" not in run["blocks"] and "ended" not in run  # a refusal, not the stop
    s = spend(env)
    s["usd"] = 115.0  # as if step 3 had spent $115
    (env.out / "spend.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match=r"with block 2's expected \$24.66 it would pass the stop"):
        block(env, 2, "docs")  # 115 + 24.66 is over 135
    run = json.loads((env.out / "main" / "run.json").read_text())
    assert len(env.spawned) == n and "2" not in run["blocks"]
    # the run's record says it ended there, so that it can still be reported
    assert run["ended"]["block"] == 2 and "would pass the stop $135.00" in run["ended"]["why"] and run["ended"]["utc"].endswith("Z")
    assert K.main(["--ended", "main"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("ended early: the spend stop refused block 2") and "seeds [0, 1, 2]" in text.split("\n")[0]
    ended = json.loads((env.out / "main" / "compare_u1_ended.json").read_text())
    assert ended["seeds"] == [0, 1, 2] and ended["blocks"] == [1] and ended["ended"].startswith("the spend stop refused")
    assert set(ended["readings"]) == {"E-N"} and ended["readings"]["E-N"]["H"]["n"] == 3
    assert ended["steps"] == [0, 3, 6] and ended["arms"] == ["N", "E"]
    with pytest.raises(SystemExit, match="before every block is in"):
        K.main(["--all", "main"])  # --all stays strict
    assert not (env.out / "main" / "compare_u1.json").exists()
    # an ended run stays ended: no block starts, not the cheaper block 3 (115 + 16.44 would fit), and not block 2
    # again once money has come back
    s["usd"] = 10.0
    (env.out / "spend.json").write_text(json.dumps(s))
    for k in (3, 2):
        with pytest.raises(SystemExit, match="this run ended.*No further block starts.*compare_u1.py --ended main"):
            block(env, k, "docs")
    assert len(env.spawned) == n and set(json.loads((env.out / "main" / "run.json").read_text())["blocks"]) == {"1"}


def test_the_default_run_end_to_end(env, capsys):
    """Revision 3's main run, as an operator would launch it: no --arms, so N and E; eight seeds in three blocks,
    each block's three stages in turn; the look after block one; then every reading, which is one contrast. All of
    it under the design's stop, with no raise."""
    ready(env)

    def outcome(e, k, step):  # arm E keeps more of its release under pressure than arm N does
        if not e.impossible:
            return "solved"
        hold = {0: 3, 3: 3, 6: 2} if env.rec.run["label"].startswith("E") else {0: 3, 3: 2, 6: 1}
        return "released" if k % 6 < (4 if e.pair.name == "p00" else hold[step]) else "failed"

    env.rec.outcome = outcome
    assert not (env.out / "stop.json").exists() and R.spend_stop(env.out) == (135.0, None)
    for k in (1, 2, 3):
        for stage in R.STAGES:
            assert R.main(["--block", str(k), "main", "--stage", stage]) == 0
        if k == 1:
            assert K.main(["--look", "main"]) == 0  # blocks 2 and 3 wait for it
    d = env.out / "main"
    run = json.loads((d / "run.json").read_text())
    assert run["arms"] == ["N", "E"] and run["read_steps"] == [3, 6] and sorted(run["blocks"]) == ["1", "2", "3"]
    assert [run["blocks"][k]["expected"] for k in "123"] == [24.66, 24.66, 16.44] and "ended" not in run
    assert run["whole_run"]["owed"] == 65.76 and all(b["stop"] == 135.0 for b in run["blocks"].values())
    kinds = [name for name, _ in env.spawned]
    assert kinds.count("finetune") == 32 and kinds.count("press") == 16  # 16 runs, three stages each
    labels = sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "info.json").exists())
    assert labels == sorted(f"{a}_s{s}" for a in "NE" for s in range(8)) and not list(d.glob("R_s*"))
    assert spend(env)["usd"] < 135.0 and not R.unsettled(env.out) and not R.read_dropped(d)
    capsys.readouterr()
    assert K.main(["--all", "main"]) == 0
    text = capsys.readouterr().out
    rep = json.loads((d / "compare_u1.json").read_text())
    assert rep["arms"] == ["N", "E"] and rep["seeds"] == list(range(8)) and rep["steps"] == [0, 3, 6]
    assert list(rep["readings"]) == ["E-N"] and rep["gates"]["ok"] and set(rep["guards"]) == {"E"}
    h = rep["readings"]["E-N"]["H"]
    assert h["primary"] and h["n"] == 8 and h["mean"] > 0.6 and h["p"] == 2 / 256 and h["verdict"] == "holds better"
    assert rep["readings"]["E-N"]["U1"]["verdict"] == "no difference shown"  # the same start
    assert rep["readings"]["E-N"]["said"] == "holds better with no head start: the stake alone"
    assert "arms ['N', 'E']" in text and "* H  holding" in text and text.count(f"note: {U.START_NOTE}") == 1
    assert not re.search(r"\bR_s\d|\barm R\b|R-N|E-R", text)
    with pytest.raises(SystemExit, match="nothing to launch"):
        R.main(["--block", "3", "main", "--stage", "press"])


def test_the_stop_counts_a_blocks_cost_for_the_runs_own_arms(env):
    """A block of three seeds is expected to cost $24.66 with N and E and $36.99 with all three arms: at $105 spent,
    the first starts under the $135 stop and the second would not."""
    ready(env)
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    assert K.main(["--look", "main"]) == 0
    s = spend(env)
    s["usd"] = 105.0
    (env.out / "spend.json").write_text(json.dumps(s))
    assert U.block_expected_usd(2, U.ARMS) == pytest.approx(36.99) and 105 + 36.99 > 135
    assert block(env, 2, "docs") == 0  # 105 + 24.66 is under 135
    rec = json.loads((env.out / "main" / "run.json").read_text())["blocks"]["2"]
    assert rec["expected"] == 24.66 and rec["spent_before"] == 105.0 and len(env.spawned[-6:]) == 6
    s = spend(env)
    s["usd"] = 120.0
    (env.out / "spend.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match=r"with block 3's expected \$16.44 it would pass the stop \$135.00"):
        block(env, 3, "docs")  # 120 + 16.44: the two-seed block of two arms
    assert json.loads((env.out / "main" / "run.json").read_text())["ended"]["block"] == 3


def test_the_stop_is_the_designs_unless_a_recorded_approval_raises_it(env):
    """The stop is u1_core.SPEND_STOP. A file in the results folder may raise it (Eugene's approval, with the ledger
    entry that records it): it is no code, so a raise after the freeze changes no frozen file. It never lowers the
    stop, and never moves it without a reason."""
    ready(env)
    assert R.spend_stop(env.out) == (U.SPEND_STOP, None) == (135.0, None)
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    run = json.loads((env.out / "main" / "run.json").read_text())
    assert run["blocks"]["1"]["stop"] == 135.0 and "stop_raised" not in run["blocks"]["1"]
    assert K.main(["--look", "main"]) == 0
    s = spend(env)
    s["usd"] = 115.0  # as if step 3 had spent $115: block 2 (24.66 more) would pass the design's stop
    (env.out / "spend.json").write_text(json.dumps(s))
    n = len(env.spawned)
    for bad in ({"usd": 120.0, "why": "lower"}, {"usd": 135.0, "why": "the same"}, {"usd": 150.0}, {"usd": 150.0, "why": " "},
                {"usd": "150", "why": "a string"}, {"usd": True, "why": "a flag"}):
        (env.out / "stop.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match="must raise the stop above \\$135.00 and say why"):
            block(env, 2, "docs")
    assert len(env.spawned) == n and "ended" not in json.loads((env.out / "main" / "run.json").read_text())
    (env.out / "stop.json").write_text(json.dumps({"usd": 150.0, "why": "ledger, 12 Nov: Eugene raised the stop"}))
    assert R.spend_stop(env.out) == (150.0, "ledger, 12 Nov: Eugene raised the stop")
    assert block(env, 2, "docs") == 0  # 115 + 24.66 is under the raised stop
    rec = json.loads((env.out / "main" / "run.json").read_text())["blocks"]["2"]
    assert rec["stop"] == 150.0 and rec["stop_raised"] == "ledger, 12 Nov: Eugene raised the stop"
    s = spend(env)
    s["usd"] = 140.0
    (env.out / "spend.json").write_text(json.dumps(s))
    for stage in R.STAGES[1:]:
        assert block(env, 2, stage) == 0
    with pytest.raises(SystemExit, match="would pass the stop \\$150.00"):
        block(env, 3, "docs")  # 140 + 16.44 is over the raised stop too: the run ends there


def test_a_block_that_has_started_goes_on_also_past_the_stop_and_in_a_run_that_ended(env):
    """The stop is between blocks. A block that started may finish its stages: otherwise a seed with one failed run
    could be neither completed nor dropped, and the run could never be reported."""
    ready(env)
    assert block(env, 1, "docs") == 0 and block(env, 1, "practice") == 0
    env.fail = {"E_s2"}
    assert block(env, 1, "press") == 1  # E_s2 failed once; arm N's three runs are in
    env.fail = set()
    assert K.main(["--look", "main"]) == 0
    assert block(env, 2, "docs") == 0  # block 2 starts under the stop
    s = spend(env)
    s["usd"] = 134.0
    (env.out / "spend.json").write_text(json.dumps(s))
    assert block(env, 2, "practice") == 0  # the stop is checked only when a block starts: block 2 goes on
    with pytest.raises(SystemExit, match="would pass the stop"):
        block(env, 3, "docs")
    assert json.loads((env.out / "main" / "run.json").read_text())["ended"]["block"] == 3
    n = len(env.spawned)
    assert block(env, 1, "press") == 0  # the run has ended, but block 1 had started: its failed run is launched again
    assert [kw["label"] for _, kw in env.spawned[n:]] == ["E_s2_try2"]
    assert block(env, 2, "press") == 0  # and block 2 finishes
    with pytest.raises(SystemExit, match="this run ended"):
        block(env, 3, "docs")
    assert K.main(["--ended", "main"]) == 0
    assert json.loads((env.out / "main" / "compare_u1_ended.json").read_text())["seeds"] == [0, 1, 2, 3, 4, 5]


def test_a_look_that_says_stop_stops_the_run(env, capsys):
    ready(env)
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    for s in (0, 1, 2):  # as if arm N had never released: both gates fail by the wide margin
        for k in (0, 6):
            R.write_jsonl(env.out / "main" / f"N_s{s}" / f"read_s{k}.jsonl", F.read_eps(["p00", "p01", "p03"], 0))
    assert K.main(["--look", "main"]) == 1 and "THE RUN STOPS" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="starts only after the look"):
        block(env, 2, "docs")
    # its runs are still reported: the early readout, on block one's seeds
    assert K.main(["--ended", "main"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("ended early: the look after block one said stop") and "seeds [0, 1, 2]" in text
    rep = json.loads((env.out / "main" / "compare_u1_ended.json").read_text())
    assert rep["seeds"] == [0, 1, 2] and not rep["gates"]["ok"]  # arm N never released: the gates fail
    assert all(rep["readings"][c][k]["verdict"] == "not read" for c in rep["readings"] for k in ("H", "U2", "U3"))


def test_a_block_refuses_code_that_differs_from_the_freeze(env, monkeypatch):
    ready(env)
    real = R.code_freeze
    monkeypatch.setattr(R, "code_freeze", lambda out: {**real(out), "files": {
        **real(out)["files"], "lab/explore/act1/modal_act1.py": "0" * 64}})
    with pytest.raises(SystemExit, match="differs from the freeze.*modal_act1.py changed"):
        block(env, 1, "docs")
    monkeypatch.setattr(R, "code_freeze", real)
    (env.out / "sets.json").write_text((env.out / "sets.json").read_text() + "\n")  # the sets record changed
    with pytest.raises(SystemExit, match="sets.json changed"):
        block(env, 1, "docs")
    assert not env.spawned and not (env.out / "main").exists()


def test_a_run_that_fails_twice_drops_its_seed_from_everything(env, capsys):
    ready(env)
    env.fail = {("docs", "E_s1")}
    assert block(env, 1, "docs") == 1
    d = env.out / "main"
    assert R.tries(d / "docs" / "E_s1") == 1 and not R.read_dropped(d) and (d / "docs" / "N_s1" / "done.json").exists()
    with pytest.raises(SystemExit, match=r"has not finished for \['E_s1'\]"):
        block(env, 1, "practice")  # a seed's arms go on together, or not at all
    n = len(env.spawned)
    assert block(env, 1, "docs") == 1  # the same command launches the failed run once more, and only it
    assert [f"{kw['label']}_s{kw['seed']}" for _, kw in env.spawned[n:]] == ["E_s1"]
    assert env.spawned[-1][1]["out_dir"] == "/runs/act1/u1/main/docs/E_s1"  # identical inputs
    dropped = R.read_dropped(d)
    assert list(dropped) == ["1"] and "E_s1 failed twice at the docs stage" in dropped["1"]["why"]
    env.fail = set()
    n = len(env.spawned)
    assert block(env, 1, "practice") == 0 and block(env, 1, "press") == 0
    labels = [f"{kw['label']}_s{kw['seed']}" if name == "finetune" else kw["label"] for name, kw in env.spawned[n:]]
    assert labels == [f"{a}_s{s}" for s in (0, 2) for a in "NE"] * 2  # seed 1 is gone from every arm
    with pytest.raises(SystemExit, match="nothing to launch"):
        block(env, 1, "docs")
    assert K.main(["--look", "main"]) == 0
    assert json.loads((d / "look.json").read_text())["seeds"] == [0, 2]


def test_a_press_rerun_writes_to_its_own_folder_on_the_volume(env):
    ready(env)
    assert block(env, 1, "docs") == 0 and block(env, 1, "practice") == 0
    env.fail = {"E_s0"}
    assert block(env, 1, "press") == 1
    assert R.tries(env.out / "main" / "E_s0") == 1 and not (env.out / "main" / "E_s0" / "info.json").exists()
    n = len(env.spawned)
    assert block(env, 1, "press") == 0  # the first folder still "fails"; the rerun's label is its own
    assert [kw["label"] for _, kw in env.spawned[n:]] == ["E_s0_try2"]
    info = json.loads((env.out / "main" / "E_s0" / "info.json").read_text())
    assert info["volume_label"] == "E_s0_try2" and info["seed"] == 0 and not R.read_dropped(env.out / "main")


def test_seeds_narrow_a_launch_and_the_days_cap_refuses_before_a_block_is_recorded(env):
    ready(env)
    with pytest.raises(SystemExit, match="are not all in block 1"):
        block(env, 1, "docs", seeds=[2, 3])
    env.today = 35.0  # $15 left today: six document runs reserve $17.40
    with pytest.raises(SystemExit, match="the day's remaining.*--seeds"):
        block(env, 1, "docs")
    assert not env.spawned and not (env.out / "main" / "run.json").exists()
    assert block(env, 1, "docs", seeds=[0, 1]) == 0  # four runs reserve $11.60
    assert [f"{kw['label']}_s{kw['seed']}" for _, kw in env.spawned] == [f"{a}_s{s}" for s in (0, 1) for a in "NE"]
    env.today = 0.0
    n = len(env.spawned)
    assert block(env, 1, "docs") == 0  # the rest of the block
    assert [f"{kw['label']}_s{kw['seed']}" for _, kw in env.spawned[n:]] == ["N_s2", "E_s2"]
    assert block(env, 1, "practice", seeds=[2]) == 0 and block(env, 1, "press", seeds=[2]) == 0
    n = len(env.spawned)
    with pytest.raises(SystemExit, match=r"already have a main run \(\['main'\]\)"):
        block(env, 1, "docs", tag="other", dose_tag="d1")  # one main tag per set of sets
    assert len(env.spawned) == n and not (env.out / "other").exists()
    with pytest.raises(SystemExit, match="no dose record under tag 'nope'"):
        block(env, 2, "docs", dose_tag="nope")
    with pytest.raises(SystemExit, match="is a dose check's tag"):
        block(env, 1, "docs", tag="d1")
    with pytest.raises(SystemExit, match="block is 1 to 3"):
        block(env, 4, "docs")


def test_a_run_still_out_at_the_deadline_is_cancelled_and_costs_its_whole_reserve(env):
    ready(env)
    assert block(env, 1, "docs") == 0 and block(env, 1, "practice") == 0
    env.hang = {"E_s2"}
    env.waits.clear(), env.options.clear()
    assert block(env, 1, "press") == 1
    assert env.cancelled == [("E_s2", True)]
    err = json.loads((env.out / "main" / "E_s2" / "error_try1.json").read_text())
    assert "cancelled at the local deadline (1.15 h a run, plus 1.5 h for queueing)" in err["error"]
    # queue time is not the run's time: the launcher waits the reserve's hours and 1.5 h more, while the container's
    # own timeout stays the reserve's hours (which caps the bill); a run cancelled then is charged its whole reserve
    whole = (1.15 + 1.5) * 3600
    assert len(env.waits) == 6 and all(whole - 30 < w <= whole for w in env.waits)
    assert env.options == [("press", {"timeout": int(1.15 * 3600)})]
    tu = spend(env)["stages"][-1]
    assert tu["container_s"]["E_s2"] == 1.15 * 3600 and tu["kind"] == "true-up"
    assert tu["metered_usd"] >= (1.15 * 3600 + 120) / 3600 * 2.55


# ------------------------------------------------------------------------------------------------ --add-arm


def test_an_arm_is_added_to_a_started_run_and_the_block_commands_launch_its_runs(env, capsys):
    """Revision 3: the rule arm joins only if the budget is raised, and only before any readout. --add-arm changes the
    run's record, leaves a line in variants.jsonl and launches nothing. The ordinary --block commands then launch the
    arm's missing runs in the blocks that started, stage by stage, and a later block runs all the run's arms."""
    ready(env)
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0  # N and E
    assert K.main(["--look", "main"]) == 0
    assert block(env, 2, "docs") == 0  # block 2 has started: its documents stage, for N and E
    n, booked = len(env.spawned), len(env.booked)
    env.variants.clear()
    raised(env, 180.0)  # Eugene raised the budget: the stop.json names the ledger entry of the approval
    assert R.main(["--add-arm", "r", "main"]) == 0
    run = json.loads((env.out / "main" / "run.json").read_text())
    assert run["arms"] == ["N", "R", "E"] and len(env.spawned) == n and len(env.booked) == booked  # nothing launched
    add = run["arms_added"]
    assert len(add) == 1 and add[0]["arm"] == "R" and add[0]["blocks"] == [1, 2] and add[0]["seeds"] == [0, 1, 2, 3, 4, 5]
    assert add[0]["expected"] == 24.66 and add[0]["utc"].endswith("Z")  # one arm, the six seeds of the started blocks
    assert add[0]["stop"] == 180.0 and add[0]["stop_raised"] == RAISE and add[0]["spent_before"] == round(spend(env)["usd"], 2)
    # what the whole run still owes with R in it: N and E for block 2's two later stages and all of block 3, and
    # every stage of R for the eight seeds
    owed = 6 * (0.48 + 2.00) + 4 * 4.11 + 8 * 4.11
    assert add[0]["owed_run"] == round(owed, 2) == 64.2
    # the addition is on record in variants.jsonl too, with the same numbers
    assert env.variants == [{"cmd": "add-arm", "tag": "main", "arms": ["N", "R", "E"],
                             **{k: v for k, v in add[0].items() if k != "utc"}}]
    assert any("arm R is added" in x and "stage by stage" in x for x in env.logs)
    assert run["blocks"]["1"]["expected"] == 24.66 and run["read_steps"] == [3, 6]  # the blocks' own records are as they were
    # the ordinary commands launch R's missing runs in block 1, stage by stage: all arms of a seed pass a stage together
    with pytest.raises(SystemExit, match=r"the docs stage has not finished for \['R_s0', 'R_s1', 'R_s2'\]"):
        block(env, 1, "practice")
    assert block(env, 1, "docs") == 0
    assert [f"{kw['label']}_s{kw['seed']}" for _, kw in env.spawned[n:]] == ["R_s0", "R_s1", "R_s2"]  # only what is missing
    assert env.spawned[-1][1]["src_dir"] == "/runs/act1/u1/sets/Rdocs"
    with pytest.raises(SystemExit, match=r"the practice stage has not finished for \['R_s0', 'R_s1', 'R_s2'\]"):
        block(env, 1, "press")
    assert block(env, 1, "practice") == 0 and block(env, 1, "press") == 0
    assert [kw["label"] for _, kw in env.spawned[-3:]] == ["R_s0", "R_s1", "R_s2"]
    kw = env.spawned[-1][1]
    assert kw["init"] == "/runs/act1/u1/main/practice/R_s2/adapter" and kw["steps"] == 6 and kw["read_steps"] == [3, 6]
    assert kw["stream"] == "gameable" and kw["seed"] == 2  # the same stream items as N_s2 and E_s2 trained on
    with pytest.raises(SystemExit, match="nothing to launch"):
        block(env, 1, "press")
    # block 2 had started with N and E: R's documents come first, then each stage goes on for all three arms
    with pytest.raises(SystemExit, match=r"has not finished for \['R_s3', 'R_s4', 'R_s5'\]"):
        block(env, 2, "practice")
    m = len(env.spawned)
    assert block(env, 2, "docs") == 0 and len(env.spawned) == m + 3
    assert block(env, 2, "practice") == 0 and block(env, 2, "press") == 0 and len(env.spawned) == m + 3 + 9 + 9
    # a block that starts after the arm was added is checked against the stop with the arm in it, and runs all three
    assert block(env, 3, "docs") == 0
    assert json.loads((env.out / "main" / "run.json").read_text())["blocks"]["3"]["expected"] == 24.66  # 2 seeds x 3 arms
    assert [f"{kw['label']}_s{kw['seed']}" for _, kw in env.spawned[-6:]] == [f"{a}_s{s}" for s in (6, 7) for a in "NRE"]
    assert block(env, 3, "practice") == 0 and block(env, 3, "press") == 0
    capsys.readouterr()
    assert K.main(["--all", "main"]) == 0
    rep = json.loads((env.out / "main" / "compare_u1.json").read_text())
    assert rep["arms"] == ["N", "R", "E"] and list(rep["readings"]) == ["E-N", "R-N", "E-R"] and rep["steps"] == [0, 3, 6]
    assert rep["seeds"] == list(range(8)) and "arms ['N', 'R', 'E']" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="arm R is already in the run"):
        R.main(["--add-arm", "R", "main"])


def test_an_arm_added_when_every_block_is_in_holds_the_readout_until_its_runs_are_in(env):
    """No contrast is computed until every planned arm of every seed is in. An arm added before any readout makes
    --all wait for it; a run of N and E that has been read takes no further arm, raise or no raise."""
    ready(env)
    for k in (1, 2, 3):
        for stage in R.STAGES:
            assert block(env, k, stage) == 0
        if k == 1:
            assert K.main(["--look", "main"]) == 0
    d = env.out / "main"
    raised(env, 180.0)
    assert R.cmd_add_arm("R", "main") == 0  # every block is in for N and E, and nothing has been read
    add = json.loads((d / "run.json").read_text())["arms_added"][0]
    assert add["seeds"] == list(range(8)) and add["expected"] == 32.88 == add["owed_run"]  # the rule arm at eight seeds
    with pytest.raises(SystemExit, match=r"for every arm of the run \(N,R,E\).*R_s0.*not finished"):
        K.main(["--all", "main"])
    assert not (d / "compare_u1.json").exists()
    for k in (1, 2, 3):
        for stage in R.STAGES:
            assert block(env, k, stage) == 0
    assert K.main(["--all", "main"]) == 0 and list(json.loads((d / "compare_u1.json").read_text())["readings"]) == [
        "E-N", "R-N", "E-R"]
    # the other way round: a run of N and E that was read first takes no arm afterwards
    money = spend(env)
    money["usd"] = 10.0  # as if little had been spent, so that a second whole run fits under the stop
    (env.out / "spend.json").write_text(json.dumps(money))
    revise(env)
    assert R.cmd_sets("u2", 0.5) == 0
    rehearsed(env, "u2", "p_u2")
    assert R.cmd_dose("d2", 1.3, 0.75) == 0
    for k in (1, 2, 3):
        for stage in R.STAGES:
            assert block(env, k, stage, tag="second") == 0
        if k == 1:
            assert K.main(["--look", "second"]) == 0
    assert K.main(["--all", "second"]) == 0
    rep = json.loads((env.out / "second" / "compare_u1.json").read_text())
    assert rep["arms"] == ["N", "E"] and list(rep["readings"]) == ["E-N"]
    with pytest.raises(SystemExit, match=r"has been read \(\['compare_u1.json'\]\).*only before any readout"):
        R.cmd_add_arm("R", "second")
    assert json.loads((env.out / "second" / "run.json").read_text())["arms"] == ["N", "E"]


def test_add_arm_refuses_what_is_no_arm_a_run_that_was_read_or_ended_and_what_would_pass_the_stop(env):
    ready(env)
    with pytest.raises(SystemExit, match="holds no main run"):
        R.cmd_add_arm("R", "main")  # an arm is added to a run that has started
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    d = env.out / "main"
    rfile = d / "run.json"
    before = rfile.read_text()
    n = len(env.spawned)
    env.variants.clear()

    def refused(match, arm="R"):
        with pytest.raises(SystemExit, match=match):
            R.cmd_add_arm(arm, "main")
        # the run's record is as it was; nothing was launched, and nothing was written to variants.jsonl
        assert rfile.read_text() == before and len(env.spawned) == n and not env.variants

    refused("'X' is not an arm", arm="X")
    refused(r"arm N is already in the run under tag 'main' \(arms N,E\)", arm="N")
    refused("arm E is already in the run", arm="e")
    # a readout exists, the full one or the early one: an arm is added only before any readout
    for name in R.READOUTS:
        (d / name).write_text("{}")
        refused(rf"has been read \(\['{name}'\]\).*only before any readout")
        (d / name).unlink()
    assert R.READOUTS == ("compare_u1.json", "compare_u1_ended.json")
    # the run has ended: the spend stop refused a block, or the look said stop
    rfile.write_text(json.dumps({**json.loads(before), "ended": {"utc": "2026-11-13T00:00:00Z", "block": 2,
                                                                  "why": "it would pass the stop"}}))
    with pytest.raises(SystemExit, match=r"this run ended \(it would pass the stop\)\. No arm is added to it"):
        R.cmd_add_arm("R", "main")
    rfile.write_text(before)
    (d / "look.json").write_text(json.dumps({"stop": True}))
    refused("this run ended \\(the look after block one said stop\\)")
    (d / "look.json").write_text(json.dumps({"stop": False}))  # a look that said go is no bar
    # no raise of the stop is on record: the rule arm joins only when the budget has been raised
    refused(r"arm R is not added to the run under tag 'main'\. Arm R joins a run only when the budget has been "
            r"raised, and no raise is on record.*stop.json.*the ledger entry that records the approval")
    (env.out / "stop.json").write_text(json.dumps({"usd": 130.0, "why": "lower"}))  # a file that does not raise it
    refused("must raise the stop above")
    raised(env, 150.0)
    # a reserve that is not settled: the stop is not computed on a booked reserve
    launch = {"tag": "main", "kind": "press", "upper_h": 1.15, "runs": ["N_s0"], "where": "press", "block": 1}
    R.book(env.out, "main-b1-press", 1.0, 0, "2026-11-11", {"kind": "reserve", "launch": launch})
    refused(r"a reserve is booked and not settled.*main-b1-press \(settle it with --fetch main\)")
    R.book(env.out, "main-b1-press", -1.0, 0, "2026-11-11", {"kind": "true-up"})
    # the stop, for the blocks already started: step 3's spend so far plus the arm's three seeds of block 1 (3 x $4.11)
    s = spend(env)
    s["usd"] = 137.68
    (env.out / "spend.json").write_text(json.dumps(s))
    refused(r"step 3 has spent \$137.68; with arm R's expected \$12.33 for the 3 seeds of the blocks already started "
            r"\(\[1\]\) it would pass the stop \$150.00")
    # the stop, for the run as a whole: the arm fits for block 1 (100 + 12.33), but with R in it the run still owes
    # N and E for blocks 2 and 3 (10 runs) and R for all eight seeds: 18 runs at $4.11
    s["usd"] = 100.0
    (env.out / "spend.json").write_text(json.dumps(s))
    assert R.owed_usd(d, U.ARMS) == pytest.approx(18 * 4.11) == pytest.approx(73.98)
    refused(r"arm R is not added to the run under tag 'main': its runs for the blocks already started \(\[1\]\) would "
            r"fit under the stop, but an arm that cannot be finished is not added\. Step 3 has spent \$100.00, and the "
            r"run as a whole \(arms N,R,E:.*\) is expected to cost \$73.98 more\. That would pass the stop \$150.00")
    assert "ended" not in json.loads(rfile.read_text())  # a refusal, not an end: the run goes on with its two arms
    # a raise that covers the whole run lets the arm in
    raised(env, 175.0)
    assert R.cmd_add_arm("R", "main") == 0
    add = json.loads(rfile.read_text())["arms_added"][0]
    assert add["stop"] == 175.0 and add["stop_raised"] == RAISE and add["owed_run"] == 73.98
    assert add["spent_before"] == 100.0 and add["expected"] == 12.33 and len(env.spawned) == n
    assert json.loads(rfile.read_text())["arms"] == ["N", "R", "E"] and [v["cmd"] for v in env.variants] == ["add-arm"]
    with pytest.raises(SystemExit, match=r"arm R is already in the run under tag 'main' \(arms N,R,E\)"):
        R.cmd_add_arm("R", "main")


def test_an_added_arm_is_written_to_variants_jsonl(env, monkeypatch):
    """With the launcher's own variant() (the other tests record its entries and write no file): adding an arm leaves
    a line in variants.jsonl, beside the change to run.json. A refused addition leaves none."""
    ready(env)
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    monkeypatch.setattr(R, "variant", env.real_variant)
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="c0ffee\n"))  # no git in a test
    log = env.out / "variants.jsonl"
    with pytest.raises(SystemExit, match="no raise is on record"):
        R.cmd_add_arm("R", "main")
    assert not log.exists()
    raised(env, 180.0)
    assert R.cmd_add_arm("R", "main") == 0
    lines = R.read_jsonl(log)
    added = json.loads((env.out / "main" / "run.json").read_text())["arms_added"][0]
    assert len(lines) == 1 and lines[0]["cmd"] == "add-arm" and lines[0]["commit"] == "c0ffee" and lines[0]["utc"].endswith("Z")
    assert (lines[0]["tag"], lines[0]["arm"], lines[0]["arms"]) == ("main", "R", ["N", "R", "E"])
    assert {k: lines[0][k] for k in ("blocks", "seeds", "expected", "owed_run", "stop", "stop_raised", "spent_before")} == {
        k: added[k] for k in ("blocks", "seeds", "expected", "owed_run", "stop", "stop_raised", "spent_before")}
    assert lines[0]["stop"] == 180.0 and lines[0]["stop_raised"] == RAISE and lines[0]["blocks"] == [1]


def test_add_arm_at_the_stop_exactly_is_allowed(env):
    """The stop is "at most": spend so far plus what the whole run still owes with the arm in it may equal it."""
    ready(env)
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    raised(env, 150.0)
    s = spend(env)
    s["usd"] = 150.0 - 73.98 + 0.01  # a cent over: with R the run still owes 18 runs at $4.11
    (env.out / "spend.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match=r"an arm that cannot be finished is not added.*\$73.98 more"):
        R.cmd_add_arm("R", "main")
    s["usd"] = 150.0 - 73.98  # the stop to the cent
    (env.out / "spend.json").write_text(json.dumps(s))
    assert R.cmd_add_arm("R", "main") == 0
    assert json.loads((env.out / "main" / "run.json").read_text())["arms_added"][0]["owed_run"] == 73.98


# ------------------------------------------------------------------------------------------------ --fetch


def test_fetch_copies_what_the_volume_holds_for_runs_with_no_local_result(env):
    ready(env)
    assert block(env, 1, "docs") == 0
    d = env.out / "main"
    vol = "act1/u1/main"
    env.volume.update({f"{vol}/practice/N_s0/{n}": b"{}" for n in ("args.json", "log.jsonl", "done.json")})
    env.volume.update({f"{vol}/practice/E_s0/args.json": b"{}"})  # not finished: no done.json
    read = b'{"pair": "p00"}\n'
    env.volume.update({f"{vol}/N_s0/info.json": json.dumps({"steps": 6, "read_steps": [0, 3, 6], "held_nll": {"N": {}}}).encode(),
                       f"{vol}/N_s0/log.jsonl": b"", **{f"{vol}/N_s0/read_s{k}.jsonl": read for k in (0, 3, 6)},
                       f"{vol}/N_s0/read_s12.jsonl": read})  # a file the run's record does not name is not asked for
    # a run whose record names a read that is not on the volume is not copied: it would pass for finished
    env.volume.update({f"{vol}/N_s1/info.json": json.dumps({"steps": 6, "read_steps": [0, 3, 6]}).encode(),
                       f"{vol}/N_s1/log.jsonl": b"", f"{vol}/N_s1/read_s0.jsonl": read, f"{vol}/N_s1/read_s3.jsonl": read})
    # and so is a record with no read steps at all (one written by rounds, before revision 3)
    env.volume.update({f"{vol}/E_s1/info.json": json.dumps({"rounds": 4, "read_rounds": [0, 2, 4]}).encode(),
                       f"{vol}/E_s1/log.jsonl": b"", f"{vol}/E_s1/read_r0.jsonl": read})
    # arm R is not in this run: nothing of it is looked for, whatever the volume holds
    env.volume.update({f"{vol}/practice/R_s0/{n}": b"{}" for n in ("args.json", "log.jsonl", "done.json")})
    booked = len(env.booked)
    assert R.cmd_fetch("main") == 0
    assert (d / "practice" / "N_s0" / "done.json").exists() and not (d / "practice" / "E_s0").exists()
    assert not (d / "practice" / "R_s0").exists()
    info = json.loads((d / "N_s0" / "info.json").read_text())
    assert info["fetched"] and info["info"]["read_steps"] == [0, 3, 6]
    assert sorted(p.name for p in (d / "N_s0").iterdir()) == ["info.json", "log.jsonl", "read_s0.jsonl", "read_s3.jsonl",
                                                              "read_s6.jsonl"]
    assert (d / "N_s0" / "read_s3.jsonl").read_bytes() == read  # byte for byte what the container wrote
    assert not (d / "N_s1").exists() and not (d / "E_s1").exists()
    assert sum("names no reads, or reads that are not there; nothing is copied" in x for x in env.logs) == 2
    assert (d / "docs" / "N_s0" / "done.json").read_text() != "{}"  # a finished local run is left as it is
    assert len(env.booked) == booked and not env.spawned[6:]  # no container, no spend: no launch was left unsettled


def test_fetch_looks_for_the_runs_own_arms(env, monkeypatch):
    """A run of N and E has no arm R: --fetch asks the volume for nothing of R's. Once R is added, it does."""
    ready(env)
    assert block(env, 1, "docs") == 0
    asked, real = [], R.fetch_volume
    monkeypatch.setattr(R, "fetch_volume", lambda rel: asked.append(rel) or real(rel))
    assert R.cmd_fetch("main") == 0
    assert asked and not any("/R_s" in rel for rel in asked) and any("/practice/E_s7/" in rel for rel in asked)
    raised(env, 180.0)
    assert R.cmd_add_arm("R", "main") == 0
    asked.clear()
    assert R.cmd_fetch("main") == 0 and any("/docs/R_s0/" in rel for rel in asked)


def age_reserve(ns, stage: str, hours: float) -> None:
    """Make a stage's reserve look `hours` older than it is (as if that long had passed since the launch)."""
    s = spend(ns)
    for x in s["stages"]:
        if x["stage"] == stage and x["kind"] == "reserve":
            x["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(R.from_utc(x["utc"]) - hours * 3600))
    (ns.out / "spend.json").write_text(json.dumps(s))


def finished_finetune(seconds: int = 1800) -> dict:
    """A fine-tune's three files as they sit on the volume once it has finished."""
    done = json.dumps({"label": "x", "seed": 0, "opt_steps": 376, "seconds": seconds}).encode()
    return {"args.json": b"{}", "log.jsonl": b"", "done.json": done}


def test_an_interrupted_launch_is_refused_until_fetch_settles_it(env):
    ready(env)
    env.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        block(env, 1, "docs")
    env.interrupt = False
    st = spend(env)["stages"]
    assert st[-1]["stage"] == "main-b1-docs" and st[-1]["kind"] == "reserve"  # no true-up followed: it was interrupted
    assert [x["stage"] for x in R.unsettled(env.out)] == ["main-b1-docs"]
    n = len(env.spawned)
    with pytest.raises(SystemExit, match=r"earlier launch of main-b1-docs was interrupted.*--fetch main"):
        block(env, 1, "docs")  # its containers may still be running: no second launch on top of them
    assert len(env.spawned) == n
    with pytest.raises(SystemExit, match="was interrupted"):
        block(env, 1, "docs", seeds=[0])  # a narrower launch of the same stage is refused too
    # five of the six runs have finished on the volume; E_s1 has not
    labels = [f"{a}_s{s}" for s in (0, 1, 2) for a in "NE"]
    for lab in labels:
        if lab != "E_s1":
            env.volume.update({f"act1/u1/main/docs/{lab}/{f}": raw for f, raw in finished_finetune().items()})
    booked = len(env.booked)
    assert R.cmd_fetch("main") == 0
    d = env.out / "main"
    assert (d / "docs" / "N_s0" / "done.json").exists() and not (d / "docs" / "E_s1").exists()
    # E_s1's container may still be running (its hour and the grace have not passed): nothing is settled yet
    assert len(env.booked) == booked and [x["stage"] for x in R.unsettled(env.out)] == ["main-b1-docs"]
    assert any("E_s1" in x and "Nothing is settled; fetch again after" in x for x in env.logs)
    with pytest.raises(SystemExit, match="was interrupted"):
        block(env, 1, "docs")
    # its hour and the grace alone are not enough: a container that queued may have started late (the queue slack)
    age_reserve(env, "main-b1-docs", 1.0 + 0.2)
    assert R.cmd_fetch("main") == 0
    assert len(env.booked) == booked and [x["stage"] for x in R.unsettled(env.out)] == ["main-b1-docs"]
    # once its time has passed, the reserve is settled on its own day
    age_reserve(env, "main-b1-docs", R.QUEUE_SLACK_H)
    assert R.cmd_fetch("main") == 0
    tu = spend(env)["stages"][-1]
    assert (tu["stage"], tu["kind"], tu["late"], tu["day"]) == ("main-b1-docs", "true-up", True, "2026-11-11")
    assert tu["unfinished"] == ["E_s1"] and tu["container_s"]["N_s0"] == 1800 and tu["container_s"]["E_s1"] == 3600
    # finished runs at their recorded seconds; the unfinished one at its whole reserve
    assert tu["metered_usd"] == pytest.approx(5 * 1800 / 3600 * 2.90 + 1.0 * 2.90, abs=1e-3)
    assert R.booked(env.out, "main-b1-docs") == pytest.approx(tu["metered_usd"], abs=1e-3) and not R.unsettled(env.out)
    assert env.booked[-1]["date"] == "2026-11-11" and env.booked[-1]["usd"] == pytest.approx(tu["usd"], abs=1e-3)
    err = json.loads((d / "docs" / "E_s1" / "error_try1.json").read_text())
    assert err["error"] == "did not finish before its container's timeout; the launch was interrupted"
    assert not R.read_dropped(d)  # a first failure: the run is launched once more
    n = len(env.spawned)
    assert block(env, 1, "docs") == 0
    assert [f"{kw['label']}_s{kw['seed']}" for _, kw in env.spawned[n:]] == ["E_s1"]
    assert R.cmd_fetch("main") == 0 and spend(env)["stages"][-1]["kind"] == "true-up"  # nothing left to settle


def test_a_run_unfinished_after_an_interrupted_launch_counts_as_a_failure(env):
    """In a block stage a second failure drops the seed, as it does when the launcher sees the failure itself."""
    ready(env)
    assert block(env, 1, "docs") == 0
    env.fail = {("practice", "E_s2")}
    assert block(env, 1, "practice") == 1  # E_s2's first failure
    env.fail, env.interrupt = set(), True
    with pytest.raises(KeyboardInterrupt):
        block(env, 1, "practice")  # its second launch is interrupted, and its container never finishes
    env.interrupt = False
    res = R.unsettled(env.out)[0]
    assert res["launch"] == {"tag": "main", "kind": "finetune", "upper_h": 0.5, "runs": ["E_s2"], "where": "practice",
                             "block": 1}
    age_reserve(env, "main-b1-practice", 0.5 + R.QUEUE_SLACK_H + 0.2)
    assert R.cmd_fetch("main") == 0
    d = env.out / "main"
    tu = spend(env)["stages"][-1]
    assert tu["late"] and tu["unfinished"] == ["E_s2"] and tu["metered_usd"] == pytest.approx(0.5 * 2.90)
    assert R.tries(d / "practice" / "E_s2") == 2 and list(R.read_dropped(d)) == ["2"]
    assert "E_s2 failed twice at the practice stage" in R.read_dropped(d)["2"]["why"]
    n = len(env.spawned)
    assert block(env, 1, "press") == 0  # the block goes on without seed 2
    assert sorted({kw["seed"] for _, kw in env.spawned[n:]}) == [0, 1]


def test_a_launch_that_never_started_is_not_a_failed_run(env):
    ready(env)
    d = env.out / "main"
    # the app cannot be entered: nothing was spawned, nothing is charged, and no run is counted as failed
    env.app_fails = True
    with pytest.raises(SystemExit, match="the launch of main-b1-docs did not start.*0 of 6 runs had been spawned"
                                         ".*No run is counted as failed"):
        block(env, 1, "docs")
    env.app_fails = False
    st = spend(env)["stages"][-2:]
    assert [x["kind"] for x in st] == ["reserve", "true-up"] and st[1]["not_started"] is True
    assert st[1]["metered_usd"] == 0.0 and st[1]["spawned_then_cancelled"] == [] and "could not be started" in st[1]["error"]
    assert R.booked(env.out, "main-b1-docs") == pytest.approx(0.0, abs=1e-3) and not R.unsettled(env.out)
    assert not list(d.rglob("error_try*.json")) and not R.read_dropped(d) and not env.spawned
    # a spawn raises part way: the four already spawned are cancelled (a cancel that errors is passed over) and
    # charged their seconds so far; the two never spawned cost nothing
    env.spawn_fails_at, env.cancel_fails = 4, True
    with pytest.raises(SystemExit, match="did not start.*4 of 6 runs had been spawned and were cancelled"):
        block(env, 1, "docs")
    env.spawn_fails_at, env.cancel_fails = None, False
    assert len(env.spawned) == 4 and env.cancelled == [("N", True), ("E", True), ("N", True), ("E", True)]
    tu = spend(env)["stages"][-1]
    assert tu["not_started"] and tu["spawned_then_cancelled"] == ["N_s0", "E_s0", "N_s1", "E_s1"]
    assert 0 <= tu["metered_usd"] < 0.05 and "Modal refused the spawn" in tu["error"]
    assert not R.unsettled(env.out) and not list(d.rglob("error_try*.json")) and not R.read_dropped(d)
    # an interrupt while the runs are still being spawned is the same case: what was spawned is cancelled, the
    # reserve is trued up, and the interrupt goes on up (an interrupt later, while waiting, leaves the runs going)
    env.spawned.clear(), env.cancelled.clear()
    env.spawn_fails_at, env.spawn_error = 2, KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        block(env, 1, "docs")
    env.spawn_fails_at, env.spawn_error = None, RuntimeError
    assert len(env.spawned) == 2 and len(env.cancelled) == 2 and spend(env)["stages"][-1]["not_started"] is True
    assert not R.unsettled(env.out) and not list(d.rglob("error_try*.json"))
    # the same command then launches the whole stage: nothing was counted against any run
    env.spawned.clear()
    assert block(env, 1, "docs") == 0 and len(env.spawned) == 6
    assert all(R.tries(d / "docs" / f"{a}_s{s}") == 0 for a in "NE" for s in (0, 1, 2))


def test_a_broken_connection_while_waiting_is_an_interrupted_launch_not_a_failed_run(env):
    ready(env)
    d = env.out / "main"
    env.offline = ModalConnectionError  # Modal's own ConnectionError: the container may still be running
    with pytest.raises(ModalConnectionError):
        block(env, 1, "docs")
    env.offline = None
    assert [x["stage"] for x in R.unsettled(env.out)] == ["main-b1-docs"]  # the reserve stays booked
    assert not list(d.rglob("error_try*.json")) and not R.read_dropped(d)  # and no run is recorded as failed
    assert any("the launch is interrupted, no run has failed; settle it with --fetch main" in x for x in env.logs)
    with pytest.raises(SystemExit, match="was interrupted.*--fetch main"):
        block(env, 1, "docs")
    for lab in (f"{a}_s{s}" for s in (0, 1, 2) for a in "NE"):  # the containers ran on and finished
        env.volume.update({f"act1/u1/main/docs/{lab}/{f}": raw for f, raw in finished_finetune().items()})
    assert R.cmd_fetch("main") == 0 and not R.unsettled(env.out)
    assert spend(env)["stages"][-1]["unfinished"] == [] and R.tries(d / "docs" / "N_s0") == 0
    env.offline = ConnectionResetError  # the builtin ConnectionError is treated the same way
    with pytest.raises(ConnectionResetError):
        block(env, 1, "practice")
    env.offline = None
    assert [x["stage"] for x in R.unsettled(env.out)] == ["main-b1-practice"] and not list(d.rglob("error_try*.json"))


def test_a_volume_that_cannot_be_read_settles_nothing(env):
    """A run whose result could not be looked for must not be taken for unfinished."""
    ready(env)
    env.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        block(env, 1, "docs")
    env.interrupt = False
    age_reserve(env, "main-b1-docs", 1.0 + R.QUEUE_SLACK_H + 0.2)  # its containers' time has passed
    d = env.out / "main"
    for down in (ModalConnectionError, TimeoutError, PermissionError):  # anything but a missing file
        env.volume_down, booked = down, len(env.booked)
        assert R.cmd_fetch("main") == 1
        assert len(env.booked) == booked and [x["stage"] for x in R.unsettled(env.out)] == ["main-b1-docs"]
        assert not list(d.rglob("error_try*.json")) and not R.read_dropped(d)
    assert sum("the volume could not be read" in x and "Nothing is settled in this call" in x for x in env.logs) == 3
    env.volume_down = None
    assert R.cmd_fetch("main") == 0 and not R.unsettled(env.out)  # the volume answers: no file there, so unfinished
    assert len(list(d.rglob("error_try1.json"))) == 6 and len(spend(env)["stages"][-1]["unfinished"]) == 6


def test_fetch_also_settles_an_interrupted_sets_build(env):
    """A rebuild that was interrupted leaves the old sets as they were, and a reserve for the build that no tag owns.
    --fetch settles it once its container's time has passed, so that the run on the old sets can start its next
    block."""
    ready(env)
    for stage in R.STAGES:
        assert block(env, 1, stage) == 0
    assert K.main(["--look", "main"]) == 0
    revise(env)
    env.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        R.cmd_sets("u2", 0.5)
    env.interrupt = False
    assert json.loads((env.out / "sets.json").read_text())["root"] == "u1" and not (env.out / "sets.u1.json").exists()
    with pytest.raises(SystemExit, match=r"a reserve is booked and not settled.*sets \(--fetch TAG settles it"):
        block(env, 2, "docs")
    assert R.cmd_fetch("main") == 0 and [x["stage"] for x in R.unsettled(env.out)] == ["sets"]
    assert any(x.startswith("sets: its interrupted launch may still be running. Nothing is settled") for x in env.logs)
    age_reserve(env, "sets", 0.5 + R.QUEUE_SLACK_H + 0.2)
    assert R.cmd_fetch("main") == 0 and not R.unsettled(env.out)
    tu = spend(env)["stages"][-1]
    assert (tu["stage"], tu["late"], tu["usd"]) == ("sets", True, 0.0)  # charged its whole reserve
    assert block(env, 2, "docs") == 0  # the run on the old sets goes on


def test_a_new_main_tag_takes_an_empty_folder(env):
    ready(env)
    (env.out / "main").mkdir()
    (env.out / "main" / "notes.txt").write_text("left here by something else")
    with pytest.raises(SystemExit, match="already exists and is not empty, and holds no main run"):
        block(env, 1, "docs")
    assert not env.spawned and not (env.out / "main" / "run.json").exists()
    (env.out / "main" / "notes.txt").unlink()
    assert block(env, 1, "docs") == 0  # an empty folder is as good as none


def test_fetch_finds_the_old_reads_second_folder_and_settles_a_dose_launch(env):
    sets_built(env)
    env.old_fails = True
    with pytest.raises(SystemExit, match="the press runs were not launched"):
        R.cmd_dose("d1", 1.3, 0.75)
    env.old_fails, env.interrupt = False, True
    with pytest.raises(KeyboardInterrupt):
        R.cmd_dose("d1", 1.3, 0.75)  # the second ask of the two old reads is interrupted
    env.interrupt = False
    assert [x["stage"] for x in R.unsettled(env.out)] == ["dose-d1-old"]
    with pytest.raises(SystemExit, match="--fetch d1"):
        R.cmd_dose("d1", 1.3, 0.75)
    # their containers finished: after the earlier failure each read's files sit in its second folder on the volume
    eps = F.read_eps(env.inp["eval_oneoff"], 3, n_sol=4, solved=4)
    for ad in U.DOSE_ADAPTERS:
        env.volume[f"h4lite/u1/d1/old_{ad}_try2/heldout.jsonl"] = "".join(json.dumps(e) + "\n" for e in eps).encode()
        env.volume[f"h4lite/u1/d1/old_{ad}_try2/heldout.json"] = b"{}"
    assert R.cmd_fetch("d1") == 0
    d = env.out / "d1"
    for ad in U.DOSE_ADAPTERS:
        assert (d / f"old_{ad}" / "heldout.jsonl").read_bytes() == env.volume[f"h4lite/u1/d1/old_{ad}_try2/heldout.jsonl"]
        assert R.tries(d / f"old_{ad}") == 1
    tu = spend(env)["stages"][-1]
    assert (tu["stage"], tu["late"], tu["unfinished"]) == ("dose-d1-old", True, [])
    assert tu["metered_usd"] == pytest.approx(2 * 0.75 * 2.90)  # a finished old read is charged its whole reserve
    assert not R.unsettled(env.out)
    n = len(env.spawned)
    assert R.cmd_dose("d1", 1.3, 0.75) == 0  # the check goes on: both old reads are in, the two press runs follow
    assert [name for name, _ in env.spawned[n:]] == ["press"] * 2


def test_a_tag_stays_on_the_sets_it_started_on(env):
    """After a rebuild under a new root, a run that started on the old sets does not go on, and --fetch still looks
    where that run's containers wrote."""
    assert R.cmd_sets("u1", 0.5) == 0
    assert R.cmd_plumb("p1", None) == 0 and R.cmd_dose("d1", 1.3, 0.75) == 0
    assert block(env, 1, "docs") == 0
    revise(env)  # a document changed: the rebuild is a revision
    assert R.cmd_sets("u2", 0.5) == 0
    n = len(env.spawned)
    with pytest.raises(SystemExit, match="'main' runs on the sets under root 'u1'; the sets are now under root 'u2'"):
        block(env, 1, "practice")
    with pytest.raises(SystemExit, match="rehearsal 'p1' ran on the sets under root 'u1'"):
        R.cmd_plumb("p1", None)
    assert len(env.spawned) == n
    d = env.out / "main"
    assert R.tag_root(env.out, d) == "u1" and R.tag_root(env.out, env.out / "p1") == "u1"
    assert R.tag_root(env.out, env.out / "d1") == "u1" and R.tag_root(env.out, env.out / "new") == "u2"
    env.volume.update({f"act1/u1/main/practice/N_s0/{f}": raw for f, raw in finished_finetune().items()})
    assert R.cmd_fetch("main") == 0 and (d / "practice" / "N_s0" / "done.json").exists()  # read under u1, not u2


def test_sets_that_carry_a_dose_check_or_a_main_run_are_rebuilt_only_as_a_revision(env, monkeypatch):
    """The same sets built again under a new root would only buy a second dose check. A rebuild needs something to
    differ: the sets' code, the documents, the expected volume records, or the code against a freeze on those sets."""
    assert R.cmd_sets("u1", 0.5) == 0
    assert R.cmd_sets("u2", 0.5) == 0  # nothing started on u1: it may be rebuilt freely
    rehearsed(env, "u2", "p_u2")
    assert R.cmd_sets("u3", 0.5) == 0  # only a rehearsal stood on u2: still free
    rehearsed(env, "u3", "p_u3")
    env.fail = {"gameable_told_s1"}
    with pytest.raises(SystemExit, match="not complete"):
        R.cmd_dose("d3", 1.3, 0.75)  # a dose check has started on u3 (its freeze.json), though it did not finish
    env.fail = set()
    n = len(env.spawned)
    with pytest.raises(SystemExit, match=r"under root 'u3' carry a dose check or a main run \(\['d3'\]\).*change nothing"):
        R.cmd_sets("u4", 0.5)
    assert len(env.spawned) == n and json.loads((env.out / "sets.json").read_text())["root"] == "u3"
    assert not (env.out / "sets.u3.json").exists()
    sets = json.loads((env.out / "sets.json").read_text())
    docs, held, files = R.local_docs(env.out)
    expect = R.local_expect(R.eval_split(), docs)
    assert R.revision(env.out, sets, files, expect) == []
    # each of these makes it a revision
    assert R.revision(env.out, sets, files, {**expect, "practice_releases": 5}) == [
        "what the volume's records are expected to hold changed"]
    changed = {**files, "R": {**files["R"], "train_sha256": "0" * 64}}
    assert R.revision(env.out, sets, changed, expect) == ["the documents changed"]
    assert R.revision(env.out, {k: v for k, v in sets.items() if k != "code"}, files, expect) == ["the sets' code changed"]
    real_code, real_freeze = R.sets_code, R.code_freeze
    with monkeypatch.context() as m:  # the sets' code changed
        m.setattr(R, "sets_code", lambda: {**real_code(), "lab/explore/act1/modal_act1.py": "1" * 64})
        assert R.revision(env.out, sets, files, expect) == ["the sets' code changed"]
    with monkeypatch.context() as m:  # a frozen file outside the sets' code changed (the launcher, say)
        m.setattr(R, "code_freeze", lambda out: {**real_freeze(out), "files": {
            **real_freeze(out)["files"], "lab/explore/understand/run_u1.py": "1" * 64}})
        assert R.revision(env.out, sets, files, expect) == ["the code differs from the freeze of d3"]
        with pytest.raises(SystemExit, match="left unfinished, by other code"):
            R.cmd_dose("d4", 1.3, 0.75)  # no new dose check on u3 under that code ...
        assert R.cmd_sets("u4", 0.5) == 0  # ... but new sets may be built for it: there is a way on
    assert json.loads((env.out / "sets.json").read_text())["root"] == "u4" and (env.out / "sets.u3.json").exists()
    # a main run on the sets counts as a dose check does
    rehearsed(env, "u4", "p_u4")
    assert R.cmd_dose("d5", 1.3, 0.75) == 0 and block(env, 1, "docs") == 0
    with pytest.raises(SystemExit, match=r"carry a dose check or a main run \(\['d5', 'main'\]\)"):
        R.cmd_sets("u5", 0.5)


def test_an_interrupted_rebuild_leaves_the_sets_as_they_were(env):
    assert R.cmd_sets("u1", 0.5) == 0
    before = (env.out / "sets.json").read_text()
    for how, exc in (("interrupt", KeyboardInterrupt), ("app_fails", SystemExit), ("spawn_fails_at", SystemExit)):
        setattr(env, how, 0 if how == "spawn_fails_at" else True)
        env.variants.clear()
        with pytest.raises(exc):
            R.cmd_sets("u2", 0.5)
        setattr(env, how, None if how == "spawn_fails_at" else False)
        # whatever stopped the launch, sets.json is the old record again, and nothing is left under the old root's name
        assert (env.out / "sets.json").read_text() == before and not (env.out / "sets.u1.json").exists(), how
        assert [v["cmd"] for v in env.variants] == ["sets-retired", "sets", "sets-restored"], how
        if how == "interrupt":  # its container may still be running: the reserve waits to be settled
            assert [x["stage"] for x in R.unsettled(env.out)] == ["sets"]
            age_reserve(env, "sets", 0.5 + R.QUEUE_SLACK_H + 0.2)
        else:  # the launch never started: its reserve is trued up to nothing
            assert not R.unsettled(env.out) and spend(env)["stages"][-1]["not_started"] is True
    assert R.cmd_sets("u2", 0.5) == 0 and (env.out / "sets.u1.json").read_text() == before


def test_an_interrupted_sets_build_is_settled_when_its_time_has_passed(env):
    env.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        R.cmd_sets("u1", 0.5)
    env.interrupt = False
    n = len(env.spawned)
    with pytest.raises(SystemExit, match="may still be running; try again after"):
        R.cmd_sets("u1", 0.5)  # the sets belong to no tag: nothing to fetch, only its container's time to wait out
    assert len(env.spawned) == n and not (env.out / "sets.json").exists()
    age_reserve(env, "sets", 0.5 + 0.2)
    with pytest.raises(SystemExit, match="may still be running; try again after"):
        R.cmd_sets("u1", 0.5)  # its half hour and the grace alone are not enough: the queue slack is waited out too
    age_reserve(env, "sets", R.QUEUE_SLACK_H)
    assert R.cmd_sets("u1", 0.5) == 0
    st = spend(env)["stages"]
    assert [(x["stage"], x["kind"], x.get("late", False)) for x in st] == [
        ("sets", "reserve", False), ("sets", "true-up", True), ("sets", "reserve", False), ("sets", "true-up", False)]
    assert st[1]["usd"] == 0.0 and st[1]["metered_usd"] == st[0]["usd"]  # the interrupted build is charged in full


# ------------------------------------------------------------------------------------------------ --plumb


def test_the_rehearsal_runs_one_run_through_the_three_stages(env, capsys):
    with pytest.raises(SystemExit, match="build the sets first"):
        R.cmd_plumb("p1", None)
    assert R.cmd_sets("u1", 0.5) == 0
    for x in (env.spawned, env.options, env.apps, env.variants, env.logs):
        x.clear()
    assert R.main(["--plumb", "p1"]) == 0
    base, offline = "/runs/act1/u1", {"HF_HUB_OFFLINE": "1"}
    assert env.apps == [("modal_sft", True), ("modal_sft", True), ("modal_act1", True)]  # each stage through launch()
    assert env.options == [("finetune", {"timeout": int(0.6 * 3600), "env": offline}),
                           ("finetune", {"timeout": int(0.35 * 3600), "env": offline}), ("press", {"timeout": 1800})]
    docs, prac, press = [kw for _, kw in env.spawned]
    # arm N, seed 0: one epoch on the arm's documents, one epoch of practice from that adapter
    assert docs == dict(label="N", seed=0, lr=1e-4, rank=16, epochs=1, batch=16, micro=4, src_dir=f"{base}/sets/Ndocs",
                        out_dir=f"{base}/p1/docs/N_s0")
    assert prac == dict(label="N", seed=0, lr=1e-4, rank=16, epochs=1, batch=16, micro=4, src_dir=f"{base}/sets/practice",
                        out_dir=f"{base}/p1/practice/N_s0", init=f"{base}/p1/docs/N_s0/adapter")
    # then press from that adapter: six steps, read at steps 0, 3 and 6, on the first read pairs in sorted order
    assert press["init"] == f"{base}/p1/practice/N_s0/adapter" and (press["label"], press["tag"]) == ("N_s0", "p1")
    assert (press["seed"], press["stream"], press["steps"], press["read_steps"]) == (0, "gameable", 6, [3, 6])
    assert press["eval_names"] == sorted(env.inp["eval_oneoff"])[:4] == [r["name"] for r in press["pairs_rows"]]
    assert press["cfg"] == U.PRESS_CFG and press["root"] == "u1"
    # it exercised a round cut short, a read inside the round and a read at the end of training
    assert env.rec.events == ["read", "nll", "nll", "nll", "train", "step", "step", "step", "read", "step", "step", "step",
                              "read"]
    assert len(env.rec.trained[-1]["items"]) == 96 and [r["seed"] for r in env.rec.reads[-3:]] == [10_000, 10_003, 10_006]
    d = env.out / "p1"
    assert sorted(p.name for p in (d / "N_s0").iterdir()) == ["info.json", "log.jsonl", "read_s0.jsonl", "read_s3.jsonl",
                                                              "read_s6.jsonl"]
    cut = R.read_jsonl(d / "N_s0" / "log.jsonl")[-1]
    assert (cut["kind"], cut["steps"], cut["cut"], cut["items"]["n"]) == ("train", 6, True, 96)
    rec = json.loads((d / "plumb.json").read_text())
    assert rec["ok"] is True and rec["checks"] == {"docs_done": True, "practice_done": True, "init_lora_b_abs": True,
                                                   "steps": True, "read_s0": True, "read_s3": True, "read_s6": True,
                                                   "held_nll": True}
    assert rec["root"] == "u1" and rec["run"] == "N_s0" and rec["plan"] == R.PLUMB and rec["pairs"] == 3
    stages = [(x["stage"], x["kind"]) for x in spend(env)["stages"] if x["stage"].startswith("plumb-p1")]
    assert stages == [(f"plumb-p1-{st}", k) for st in R.STAGES for k in ("reserve", "true-up")]
    assert rec["usd"] == pytest.approx(sum(x["usd"] for x in spend(env)["stages"] if x["stage"].startswith("plumb-p1")), abs=1e-3)
    assert env.variants[0]["cmd"] == "plumb" and env.variants[0]["stages"] == list(R.STAGES)
    # seconds and costs only: no outcome of the run is printed or logged
    out = capsys.readouterr().out
    assert ("the rehearsal (N_s0: 1 epoch a fine-tune, 6 steps of phase 3 read at steps [0, 3, 6], 3 read pairs): ok; "
            "it cost $") in out
    assert not any(w in out + " ".join(env.logs) for w in ("released", "hacked", "'imp'"))
    # the command goes on from where it stopped: with every stage done, nothing is launched
    n = len(env.spawned)
    assert R.cmd_plumb("p1", None) == 0 and len(env.spawned) == n
    # a rehearsal's tag is its own: no dose check and no block under it, and no rehearsal under theirs
    with pytest.raises(SystemExit, match="is a rehearsal's tag"):
        R.cmd_dose("p1", 1.3, 0.75)
    with pytest.raises(SystemExit, match="is a rehearsal's tag"):
        block(env, 1, "docs", tag="p1")
    assert R.cmd_dose("d1", 1.3, 0.75) == 0
    with pytest.raises(SystemExit, match=r"holds \['dose.json', 'freeze.json'\]"):
        R.cmd_plumb("d1", None)
    assert block(env, 1, "docs") == 0  # the one-main-tag rule does not count the rehearsal's tag
    with pytest.raises(SystemExit, match=r"holds \['run.json'\]"):
        R.cmd_plumb("main", None)
    assert len([f for f in env.out.glob("*/run.json")]) == 1


def test_a_rehearsal_stage_that_fails_stops_it_and_the_command_goes_on_from_there(env, capsys):
    assert R.cmd_sets("u1", 0.5) == 0
    env.spawned.clear()
    env.fail = {("practice", "N_s0")}
    with pytest.raises(SystemExit, match="the rehearsal's practice stage failed"):
        R.cmd_plumb("p1", None)
    d = env.out / "p1"
    assert [name for name, _ in env.spawned] == ["finetune", "finetune"] and R.tries(d / "practice" / "N_s0") == 1
    assert json.loads((d / "plumb.json").read_text())["ok"] is None  # not finished: no verdict yet
    env.fail = {"N_s0"}  # practice now runs; the press run's first folder fails
    with pytest.raises(SystemExit, match="the rehearsal's press stage failed"):
        R.cmd_plumb("p1", None)
    assert [name for name, _ in env.spawned[2:]] == ["finetune", "press"]  # the documents stage was not run again
    env.spawned.clear()
    assert R.cmd_plumb("p1", 0.75) == 0  # --upper-h: the reserve for each stage still to run
    assert [(name, kw["label"]) for name, kw in env.spawned] == [("press", "N_s0_try2")]  # press never overwrites
    assert env.options[-1] == ("press", {"timeout": 2700}) and json.loads((d / "plumb.json").read_text())["ok"] is True
    # a rehearsal whose run does not show what it must is not ok
    info = json.loads((d / "N_s0" / "info.json").read_text())
    info["held_nll"]["E"]["nll_per_token"] = float("nan")
    (d / "N_s0" / "info.json").write_text(json.dumps(info))
    R.write_jsonl(d / "N_s0" / "read_s3.jsonl", R.read_jsonl(d / "N_s0" / "read_s3.jsonl")[:-1])  # a record short
    capsys.readouterr()
    assert R.cmd_plumb("p1", None) == 1
    out = capsys.readouterr().out
    rec = json.loads((d / "plumb.json").read_text())
    assert rec["ok"] is False and rec["checks"]["held_nll"] is False and rec["checks"]["read_s3"] is False
    assert rec["checks"]["read_s0"] is True and rec["checks"]["read_s6"] is True and rec["checks"]["steps"] is True
    assert "NOT OK" in out and "read_s3: FAILED" in out
    # nor is one whose own record says it trained other steps than the rehearsal's (a whole round, say)
    info["info"]["steps"] = 12
    (d / "N_s0" / "info.json").write_text(json.dumps(info))
    assert R.cmd_plumb("p1", None) == 1 and json.loads((d / "plumb.json").read_text())["checks"]["steps"] is False


def test_an_interrupted_rehearsal_press_is_fetched_and_settled_at_its_seconds(env):
    assert R.cmd_sets("u1", 0.5) == 0
    env.fail = {"N_s0"}  # the press stage fails once, so the rehearsal stops after documents and practice
    with pytest.raises(SystemExit, match="press stage failed"):
        R.cmd_plumb("p1", None)
    env.fail, env.interrupt = set(), True
    with pytest.raises(KeyboardInterrupt):
        R.cmd_plumb("p1", None)
    env.interrupt = False
    names = sorted(env.inp["eval_oneoff"])[:4]
    read = "".join(json.dumps(e) + "\n" for e in F.read_eps(names, 3)).encode()
    info = {"init_lora_b_abs": 2.5, "seconds": 300.0, "steps": 6, "read_steps": [0, 3, 6],
            "held_nll": {a: {"nll_per_token": 1.5} for a in U.ARMS}}
    vol = "act1/u1/p1/N_s0_try2"  # the second attempt's own folder
    env.volume.update({f"{vol}/info.json": json.dumps(info).encode(), f"{vol}/log.jsonl": b"",
                       **{f"{vol}/read_s{k}.jsonl": read for k in (0, 3, 6)}})
    assert R.cmd_fetch("p1") == 0
    tu = spend(env)["stages"][-1]
    assert (tu["stage"], tu["late"], tu["unfinished"]) == ("plumb-p1-press", True, [])
    assert tu["metered_usd"] == pytest.approx((300 + 120) / 3600 * 2.55, abs=1e-3)  # the press run's recorded seconds
    n = len(env.spawned)
    assert R.cmd_plumb("p1", None) == 0 and len(env.spawned) == n  # every stage is in: the checks, and no launch
    assert json.loads((env.out / "p1" / "plumb.json").read_text())["ok"] is True
