"""The GPU app on a laptop, under a stand-in `modal` package: a job of phases (the view-first step) through work() on
a tiny random model and through main()'s laptop side, with the results folder and the spend files in a temporary
directory. Nothing here touches Modal, a GPU, the real model or the lab's spend records. Claude (Fable 5.2, an AI
model) wrote this file on 19 Nov 2026."""

import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from test_overlap import DOCS_A, DOCS_B, PRACTICE, documents, make_tiny  # noqa: E402  (the tiny model and the made-up texts)

LOADED = ("h4_modal", "modal_overlap")


class Stand:
    """The modal surface modal_overlap.py and h4_modal.py touch: the image is built from nothing, a function runs in
    this process (map calls it once a job), a volume commits nothing and counts how often it was asked to."""

    def __init__(self):
        stand = self
        self.commits = 0
        m = types.ModuleType("modal")

        class Image:
            @staticmethod
            def debian_slim(**k):
                return Image()

            def __getattr__(self, name):
                return lambda *a, **k: self

        class Func:
            def __init__(self, f):
                self.f, self.__name__ = f, f.__name__

            def __call__(self, *a, **k):
                return self.f(*a, **k)

            def map(self, jobs, order_outputs=True, return_exceptions=False):
                for job in jobs:
                    try:
                        yield self.f(job)
                    except Exception as e:  # noqa: BLE001  (as Modal hands a failed container back)
                        if not return_exceptions:
                            raise
                        yield e

        class App:
            def __init__(self, name, image=None, **k):
                self.name, self.image = name, image

            def function(self, **k):
                return lambda f: Func(f)

            def local_entrypoint(self, **k):
                return lambda f: f

        class Volume:
            @staticmethod
            def from_name(name, **k):
                return Volume()

            def read_only(self):
                return self

            def commit(self):
                stand.commits += 1

        m.Image, m.App, m.Volume, m.is_local = Image, App, Volume, (lambda: True)
        self.module = m


@pytest.fixture
def app(tmp_path, monkeypatch):
    """(the app's module loaded afresh under the stand-in, the stand-in). Results, spend files and the adapters'
    volume are folders of tmp_path; a model that is loaded is a new tiny one."""
    pytest.importorskip("peft")
    for key, value in (("ANATTA_RESULTS_ROOT", tmp_path / "results"), ("ANATTA_SPEND_DIR", tmp_path / "spend"), ("ANATTA_DAILY_CAP_USD", "500")):
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv("OVERLAP_SET", raising=False)
    stand = Stand()
    monkeypatch.setitem(sys.modules, "modal", stand.module)
    for name in LOADED:
        sys.modules.pop(name, None)
    import modal_overlap as MO

    import anatta.models
    from anatta import spend

    assert tmp_path in MO.step_dir().parents and tmp_path in spend.default_spend_dir().parents  # nothing of the lab's is written to
    monkeypatch.setitem(MO.MODELS, "tiny", {"path": "a-tiny-model", "layer": 2, "batch": 8})
    monkeypatch.setattr(MO, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(anatta.models, "load", lambda path: make_tiny())
    yield MO, stand
    for name in LOADED:
        sys.modules.pop(name, None)


ASKS = [{"name": "ask", "ids": ["a", "b"], "max_new": 4, "temperature": 0.0, "chats": [[{"role": "user", "content": p}] for p, _ in PRACTICE[:2]]}]
TEXTS = {"a": documents(DOCS_A, 4, 1), "b": documents(DOCS_B, 4, 1)}


def arm_job(name: str = "view_I_s0") -> dict:
    """An arm as run_view.py writes it: documents, then the practice in a fresh adapter on top, each saved."""
    return {"name": name, "model": "tiny", "train": None, "sets": [], "lora": {"seed": 0, "rank": 8, "alpha": 32, "dropout": 0.2},
            "phases": [{"name": "docs", "loss": "text", "data": documents(DOCS_A, 12, 0), "steps": 4, "lr": 1e-2, "batch": 4, "max_tokens": 64,
                        "save": f"overlap/tiny/{name}_d"},
                       {"name": "practice", "loss": "sft", "data": PRACTICE, "steps": 5, "lr": 5e-3, "batch": 2, "stack": True,
                        "save": f"overlap/tiny/{name}_p"}],
            "look_sets": ASKS, "held_texts": TEXTS, "held_max_tokens": 64}


def test_a_job_of_phases_runs_in_a_container_and_its_saved_adapters_answer_as_the_arm_did(app, tmp_path):
    MO, stand = app
    job = arm_job()
    r = MO.work(job)
    assert {"name", "model", "replies", "looks", "log", "held", "lora", "info", "seconds", "load_seconds", "train_seconds", "phases", "held_texts"} <= set(r)
    assert r["replies"] == {} and list(r["looks"]) == ["docs", "practice"] and all(len(v["ask"]) == 2 for v in r["looks"].values())
    assert list(r["held"]) == ["before", "docs", "practice"] and [p["name"] for p in r["phases"]] == ["docs", "practice"]
    assert r["lora"] == job["lora"] | {"adapters": ["default", "practice"]} and len(r["log"]) == 9 and {x["phase"] for x in r["log"]} == {"docs", "practice"}
    # each adapter by itself on the volume, its files at the folder that was named; two commits
    runs = tmp_path / "runs" / "overlap" / "tiny"
    assert all((runs / d / f).exists() for d in ("view_I_s0_d", "view_I_s0_p") for f in ("adapter_config.json", "adapter_model.safetensors"))
    assert stand.commits == 2
    # no document and no practice reply goes back
    blob = json.dumps(r)
    assert not any(d in blob for d in job["phases"][0]["data"] + TEXTS["a"] + TEXTS["b"]) and not any(reply in blob for _, reply in PRACTICE)
    # asked later from the saved adapters: the pair answers as the arm did after the practice, the first alone as after the documents
    both = MO.work({"name": "later", "model": "tiny", "train": None, "adapters": ["overlap/tiny/view_I_s0_d", "overlap/tiny/view_I_s0_p"], "sets": ASKS})
    assert both["replies"] == r["looks"]["practice"] and both["lora"] == {"loaded": ["overlap/tiny/view_I_s0_d", "overlap/tiny/view_I_s0_p"]}
    assert "phases" not in both and both["log"] == [] and both["held"] == {}
    first = MO.work({"name": "later", "model": "tiny", "train": None, "adapters": ["overlap/tiny/view_I_s0_d"], "sets": ASKS})
    assert first["replies"] == r["looks"]["docs"]
    with pytest.raises(FileNotFoundError):
        MO.work({"name": "later", "model": "tiny", "train": None, "adapters": ["overlap/tiny/nowhere"], "sets": ASKS})
    with pytest.raises(ValueError):
        MO.work(job | {"adapter": "overlap/tiny/view_I_s0_d"})  # a job of phases names no adapter of the older kind


def test_the_laptop_side_keeps_each_phase_without_its_data_and_leaves_older_jobs_as_they_were(app, tmp_path, capsys):
    MO, _ = app
    old = {"name": "plain_s0", "model": "tiny", "sets": ASKS,
           "train": {"loss": "sft", "data": PRACTICE, "seed": 0, "steps": 3, "lr": 1e-2, "batch": 2, "rank": 8, "alpha": 32, "dropout": 0.2,
                     "save": "overlap/tiny/plain_s0"}}
    jobs = [{"name": "view_base", "model": "tiny", "train": None, "sets": ASKS},
            {"name": "view_base_held", "model": "tiny", "train": None, "sets": ASKS, "held_texts": TEXTS, "held_max_tokens": 64},
            arm_job(), old, {"name": "elsewhere", "model": "m7", "train": None, "sets": ASKS}]
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps(jobs))
    MO.main(stage="jobs", models="tiny", jobs=str(f), est=30)
    said = capsys.readouterr().out
    out = tmp_path / "results" / "explore" / "2026-11-19" / "attractor-o1" / "gpu"
    assert sorted(p.name for p in (out / "tiny").iterdir()) == ["plain_s0.json", "view_I_s0.json", "view_base.json", "view_base_held.json"]
    arm = json.loads((out / "tiny" / "view_I_s0.json").read_text())
    # each phase: its settings as the job gave them and what the container recorded of it, and never its data
    docs, practice = arm["phases"]
    assert (docs["name"], docs["loss"], docs["lr"], docs["batch"], docs["max_tokens"], docs["save"]) == ("docs", "text", 1e-2, 4, 64, "overlap/tiny/view_I_s0_d")
    assert (docs["steps"], docs["documents"], docs["adapter"]) == (4, 12, "default") and len(docs["data_sha256"]) == 64 and "data" not in docs
    assert (practice["stack"], practice["steps"], practice["pairs"], practice["adapter"], practice["on_top_of"]) == (True, 5, 4, "practice", ["default"])
    assert "data" not in practice and arm["train"] == {} and arm["sets"] == {} and arm["look_sets"] == {"ask": ["a", "b"]} and arm["cost_usd"] > 0
    text = (out / "tiny" / "view_I_s0.json").read_text()
    assert not any(d in text for d in arm_job()["phases"][0]["data"] + TEXTS["a"] + TEXTS["b"]) and not any(reply in text for _, reply in PRACTICE)
    assert "docs: 4 steps, loss " in said and "practice: 5 steps, loss " in said
    # the unchanged model measured: no phase, the held texts' losses before anything; the plain ask and the older training job as before
    held = json.loads((out / "tiny" / "view_base_held.json").read_text())
    assert held["phases"] == [] and set(held["held"]["before"]) == {"a", "b"} and held["held_texts"]["a"]["documents"] == 4 and len(held["replies"]["ask"]) == 2
    base, plain = (json.loads((out / "tiny" / f"{n}.json").read_text()) for n in ("view_base", "plain_s0"))
    assert "phases" not in base and "held_texts" not in base and base["held"] == {} and base["lora"] == {} and len(base["replies"]["ask"]) == 2
    kept = {k: v for k, v in plain["train"].items() if k != "held"}  # (in one process the container's own "held" default shows; on Modal it does not)
    assert "phases" not in plain and kept == {k: v for k, v in old["train"].items() if k not in ("data", "held")} and len(plain["log"]) == 3
    assert plain["lora"]["pairs"] == 4 and plain["held"] == {"before": {}, "after": {}} and (tmp_path / "runs" / "overlap" / "tiny" / "plain_s0" / "adapter_config.json").exists()
    assert "plain_s0: " in said and "3 steps, loss " in said
    # every container is booked in the step's own log, and a second run finds nothing left to do
    rows = [json.loads(x) for x in (out / "spend.jsonl").read_text().splitlines()]
    assert sorted(r["name"] for r in rows) == ["plain_s0", "view_I_s0", "view_base", "view_base_held"] and all(r["kind"] == "jobs" for r in rows)
    MO.main(stage="jobs", models="tiny", jobs=str(f), est=30)
    assert "nothing to do" in capsys.readouterr().out and len((out / "spend.jsonl").read_text().splitlines()) == 4
