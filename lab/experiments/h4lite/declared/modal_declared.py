"""The declared-release pilot's Modal side (lab/plans/declared-release-pilot.md, registered in lab/ledger/NOTES.md,
"Declared-release pilot, registered: can a release taught only by documents become release behavior?", Opus 5.5, 20:21
UTC on 31 Oct 2026). Recognition track; private. Claude (Opus 5.5) wrote it for Eugene Jhong.

Stages (local entrypoints; run_pilot.sh runs them in order, and each resumes from what is on disk or on the volume):
  sets        results/declared/docs/'s document files uploaded to /runs/h4lite/declared/docs/ (a file already there
              must hold the same bytes: nothing is overwritten), then build_sets on CPU. S is koan's training set
              without its 102 release episodes, its 463 solves (/runs/h4lite/sft/koan/train.pt, which build_told
              asserts); D1k, D3k and D10k are S plus the first 1,000, 3,000 or 10,000 declarations; N3k and N10k are S
              plus the first 3,000 or 10,000 neutral documents; Dseq_docs is the first 10,000 declarations alone. With
              --ext: D30k, S plus the first 30,000 declarations (declared.jsonl, then declared_ext.jsonl). A document
              is one item whose prompt is the BOS alone and whose completion is the document plus the EOS
              (declared_core.doc_item); every document item is checked as the smoke's assertion 1 asks. Refused unless
              docs/judge.json's gate passed (and, for --ext, decide_d30k.json says run and docs/judge_ext.json's gate
              passed on D30k's documents); refused unless the screen's guard holds (amendment 13, must-fix 1:
              screen_refusal): docs/screen.jsonl holds a final, parsed, clean verdict for the current text sha of
              every kept declaration in declared.jsonl and held_declared.jsonl (with --ext, declared_ext.jsonl too)
              and, when the neutral screen ran, of every kept neutral document, and screen.jsonl is the file
              manifest.json (with --ext, manifest_ext.json too) sealed; and refused while any set on disk was built
              from other documents than the manifest now on disk (a deliberate rebuild deletes the old sets,
              adapters and reads, on disk and on the volume). Prints each set's solve tokens and stops unless every
              main set with solves carries S's 140,767 (declared_core.KOAN_SOLVE_TOKENS; data.json also records
              koan_solve_tokens, all 463 solves).
  train_arms  modal_sft.finetune per arm and seed: Stage A's settings (LoRA rank 16, lr 1e-4, 2 epochs, batch 16,
              micro-batch 4), a fresh LoRA. Dseq: Dseq_docs first, then S's set warm-started from that adapter (init
              its path, as E2 warm-starts). The 10,000-document fine-tunes get 12 h and D30k's 24 h
              (Function.with_options; modal_sft.py is unchanged).
  read_arms   modal_retreat2.evaluate_adapter on each adapter, durability's call byte for byte with only the adapter
              swapped: arm koan, 4 samples a twin, 3 attempts, temperature 0.7, MAX_NEW, batch 32, eval seed 3000,
              no cut, Gate 0's held split. reads/LABEL/heldout.{json,jsonl}. The base model's read is not re-run
              (results/heldout/base/).
  belief      belief_read on every adapter and the base model: the belief probe (27 items under two system prompts,
              4 samples each at temperature 0.7, 256 new tokens) and the NLL per token of the 200 held-out declared and
              200 held-out neutral documents (and of the first 200 trained documents of each kind: the final training
              NLL on the documents). belief/LABEL/{probe.jsonl, nll.json}; gen_declarations.py probe-judge labels the
              answers. Dseq_docs_s0/s1 are read too, reported only.
  noexit      the held-out read under arm noexit (no hint) at the go arm(s): reads_noexit/LABEL/.
  smoke       every path at toy sizes under declared_smoke/: the smoke's 20 documents of each kind (gen_declarations.py
              smoke first), the sets (S with 65 items: 10 steps; D1k = S plus 15 declarations, 80 items: 10 steps;
              Dseq_docs with 15: 2 steps, then S16 warm-started: 2 steps; mix16, 8 one-turn solves and 8 declarations),
              assertion 1 on every document item, assertion 2 (the first optimizer step's loss on mix16 at micro-batch 1
              and at micro-batch 4, within 0.01 nats in bf16), a held-out read on 4 pairs (and the noexit path), the
              probe on 4 questions per condition and the NLL read on 10 documents, on D1k_s0 and the base model. Its
              fine-tunes get a one-hour timeout through Function.with_options (the call --go uses for its long ones),
              and it checks that the real koan set's 463 solves carry declared_core.KOAN_SOLVE_TOKENS.
              smoke_report.json; it isn't read.
  status      what is done, on the volume and on disk.

Volume: /runs/h4lite/declared/{docs, sets/SET, train/LABEL, reads/LABEL, reads_noexit/LABEL, belief/LABEL}
(declared_smoke/ for the smoke); local mirror: results/declared/ (results/declared_smoke/).

Usage (~/prj/matlab/.venv/bin/modal, from ~/prj/align/anatta-lab; name the entrypoint):
  modal run lab/experiments/h4lite/declared/modal_declared.py::smoke
  modal run lab/experiments/h4lite/declared/modal_declared.py::sets [--ext]
  modal run --detach lab/experiments/h4lite/declared/modal_declared.py::train_arms [--arms D30k]
  modal run --detach lab/experiments/h4lite/declared/modal_declared.py::read_arms [--labels D30k_s0,D30k_s1]
  modal run --detach lab/experiments/h4lite/declared/modal_declared.py::belief [--labels ...]
  modal run --detach lab/experiments/h4lite/declared/modal_declared.py::noexit --labels D10k_s0,D10k_s1
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if (HERE.parent / "modal_common.py").exists():  # the laptop: the shared modules sit one folder up
    sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import declared_core as core  # noqa: E402
from modal_common import (CACHE, GPU, MAX_NEW, MODEL, RESULTS, ROOT, RUNS, cache, image as base_image,  # noqa: E402
                          load_model, read_rows, runs, write_json, write_jsonl)

# the shared image plus the pilot's core; the shared image stays as it is
image = base_image.add_local_file(str(HERE / "declared_core.py"), "/root/declared_core.py")
app = modal.App("anatta-h4lite-declared", image=image)
if modal.is_local():  # containers mount only this file and the image's modules, so the includes stay on the laptop
    import modal_retreat2
    import modal_sft

    app.include(modal_sft.app)  # finetune
    app.include(modal_retreat2.app)  # evaluate_adapter (the held-out read)

KOAN = f"{ROOT}/sft/koan"
TRAIN_FILES = ("args.json", "log.jsonl", "done.json")
READ_FILES = ("heldout.json", "heldout.jsonl")
BELIEF_FILES = ("probe.jsonl", "nll.json")
TIMEOUTS = {"D10k": 12 * 3600, "N10k": 12 * 3600, core.DSEQ_DOCS: 12 * 3600, "D30k": 24 * 3600}
SMOKE_TIMEOUT = 3600  # the smoke's fine-tunes go through Function.with_options too, so --go is not its first use
NLL_DOCS = 200  # the held-out documents per kind; also the first 200 trained documents per kind
SMOKE_NLL_DOCS = 5  # "the NLL read on 10 documents": 5 held-out of each kind
SMOKE_PAIRS = 4
ASSERT2_TOL = 0.01  # check 7's own gate, in nats


def app_sha() -> str:
    """sha256 of this file (in a container, the copy Modal mounts), recorded in every set's data.json."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def vol(smoke: bool) -> str:
    return "declared_smoke" if smoke else "declared"


def out_dir(smoke: bool) -> Path:
    return RESULTS / vol(smoke)


def labels_of(arm: str, smoke: bool = False) -> list[str]:
    seeds = (0,) if smoke else core.ARMS["Dseq" if arm == core.DSEQ_DOCS else arm]["seeds"]
    return [f"{arm}_s{s}" for s in seeds]


def main_labels() -> list[str]:
    """The 13 adapters the held-out read covers (D30k's two are added when it runs)."""
    return [lab for arm in core.MAIN_ARMS for lab in labels_of(arm)]


def belief_labels() -> list[str]:
    return main_labels() + labels_of(core.DSEQ_DOCS) + ["base"]


def adapter_path(label: str, smoke: bool = False) -> str | None:
    """Where modal_sft.finetune writes a label's adapter (declared_grade.adapter_path agrees); "base" is the base model."""
    return None if label == "base" else f"{ROOT}/{vol(smoke)}/train/{label}/adapter"


def names_of(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def load_config() -> dict:
    return json.loads((HERE / "config.json").read_text())


# ------------------------------------------------------------------------------------------------- in-container


def load_tokenizer():
    """The tokenizer exactly as anatta.models.load builds it (AutoTokenizer.from_pretrained(MODEL), its pad token the
    EOS when it has none), without the weights."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def check_doc_items(tok, items: list, texts: list[str]) -> int:
    """The smoke's assertion 1, on every document item: prompt == [bos_id], decode(prompt + completion) == "<s>" + text
    + "</s>", and the completion mask covers every token after the BOS (player.pack, as modal_sft.finetune packs)."""
    from player import pack

    if (tok.bos_token, tok.eos_token) != ("<s>", "</s>"):
        raise SystemExit(f"the tokenizer's BOS/EOS are {tok.bos_token!r}/{tok.eos_token!r}, not <s>/</s>")
    for k, ((p, c), text) in enumerate(zip(items, texts)):
        if p != [tok.bos_token_id]:
            raise SystemExit(f"document item {k}: prompt {p} is not [bos_id]")
        dec = tok.decode(p + c, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if dec != tok.bos_token + text + tok.eos_token:
            raise SystemExit(f"document item {k}: decode(prompt + completion) is not <s> + text + </s>")
        full, attn, cm, wc = pack([(p, c)], tok.pad_token_id, "cpu")
        if (wc != len(c) or int(cm.sum()) != len(c) or full[0, 0].item() != tok.bos_token_id
                or full[0, 1:].tolist() != c or int(attn.sum()) != len(c) + 1):
            raise SystemExit(f"document item {k}: the completion mask does not cover every token after the BOS")
    return len(items)


def check_mixed_batch(tok, items: list) -> bool:
    """In one padded micro-batch of document and solve items, each row's mask covers exactly its completion and the
    tokens there are its own (a BOS-only prompt left-padded beside a long chat prompt)."""
    from player import pack

    full, attn, cm, wc = pack(items, tok.pad_token_id, "cpu")
    wp = full.shape[1] - wc
    for j, (p, c) in enumerate(items):
        if int(cm[j].sum()) != len(c) or full[j, wp:wp + len(c)].tolist() != list(c) \
                or full[j, wp - len(p):wp].tolist() != list(p) or int(attn[j].sum()) != len(p) + len(c):
            raise SystemExit(f"mixed batch row {j}: the mask or the tokens are not the item's")
    return True


@app.function(volumes={RUNS: runs}, timeout=900)
def read_files(rels: list) -> dict:
    """The text of files under /runs/h4lite by relative path; None for a missing one."""
    runs.reload()
    return {rel: (Path(ROOT) / rel).read_text() if (Path(ROOT) / rel).is_file() else None for rel in rels}


@app.function(volumes={RUNS: runs}, timeout=900)
def volume_status(rels: list) -> dict:
    """Which paths under /runs/h4lite exist (files or folders)."""
    runs.reload()
    return {rel: (Path(ROOT) / rel).exists() for rel in rels}


@app.function(volumes={RUNS: runs}, timeout=1800)
def file_shas(rels: list) -> dict:
    """sha256 of files under /runs/h4lite by relative path; None for a missing one."""
    runs.reload()
    return {rel: hashlib.sha256((Path(ROOT) / rel).read_bytes()).hexdigest() if (Path(ROOT) / rel).is_file() else None
            for rel in rels}


@app.function(volumes={CACHE: cache, RUNS: runs}, cpu=8.0, memory=65536, timeout=2 * 3600)
def build_sets(labels: list, root: str = "declared", smoke: bool = False, record: dict | None = None,
               koan_dir: str = KOAN) -> dict:
    """The training sets (module docstring) in ROOT/root/sets/SET/{train.pt, data.json}. Each document file must hold
    the bytes the laptop's manifest records (record["doc_shas"]); every document item passes assertion 1 and one mixed
    padded batch is checked. data.json records the counts, tokens and items, the documents' ids fingerprint, and the
    shas (core, app, config, manifest, document files)."""
    import torch

    runs.reload()
    record = record or {}
    tok = load_tokenizer()
    koan = torch.load(Path(koan_dir) / "train.pt")
    recs, turns = koan["records"], koan["turns"]
    s_idx = core.solve_episodes(recs, turns)
    koan_solve_tokens = sum(len(c) for i in s_idx for _, c in turns[i])  # S's 463 solves, every turn
    solves = {"all": s_idx}
    if smoke:
        solves.update(s65=core.smoke_subset(s_idx, turns, core.SMOKE_S_ITEMS),
                      s16=core.smoke_subset(s_idx, turns, core.SMOKE_S16_ITEMS),
                      mix8=core.smoke_subset(s_idx, turns, 8, one_turn_only=True))
    specs = {lab: core.set_spec(lab, smoke) for lab in labels}
    docs_dir = Path(ROOT) / root / "docs"
    rows, items, checked = {}, {}, {}
    for f in sorted({f for sp in specs.values() for f in sp["files"]}):
        raw = (docs_dir / f).read_bytes()
        want = (record.get("doc_shas") or {}).get(f)
        if want is None or hashlib.sha256(raw).hexdigest() != want:
            raise SystemExit(f"{docs_dir / f} does not hold the bytes the manifest records: nothing built")
        rows[f] = read_rows(raw.decode())
        ids = tok([r["text"] for r in rows[f]], add_special_tokens=False)["input_ids"]
        items[f] = [core.doc_item(tok.bos_token_id, tok.eos_token_id, x) for x in ids]
        checked[f] = check_doc_items(tok, items[f], [r["text"] for r in rows[f]])
        print(json.dumps({"file": f, "documents": len(rows[f]), "assertion_1_checked": checked[f]}), flush=True)
    some_doc = next(iter(items.values()))[:2] if items else []
    mixed = check_mixed_batch(tok, some_doc + [turns[i][-1] for i in s_idx[:2]]) if some_doc else None
    infos = {}
    for lab, sp in specs.items():
        episodes = [(recs[i], turns[i]) for i in solves[sp["solves"]]] if sp["solves"] else []
        n_solves = len(episodes)
        pool = [(r, it) for f in sp["files"] for r, it in zip(rows[f], items[f])]
        if len(pool) < sp["docs"]:
            raise SystemExit(f"{lab} needs {sp['docs']} documents; the files hold {len(pool)}")
        chosen = pool[:sp["docs"]]
        if any(r["kind"] != sp["kind"] for r, _ in chosen):
            raise SystemExit(f"{lab}: a document of the wrong kind")
        episodes += [(core.doc_record(r, len(it[1])), [it]) for r, it in chosen]
        data = {"records": [r for r, _ in episodes], "turns": [t for _, t in episodes]}
        out = Path(ROOT) / root / "sets" / lab
        out.mkdir(parents=True, exist_ok=True)
        torch.save(data, out / "train.pt")
        solve_turns = data["turns"][:n_solves]
        doc_ids = [r["id"] for r, _ in chosen]
        info = {"label": lab, "arm": lab, "smoke": smoke, "cut_sampling": False, "cut_finetune": False,
                "kept": len(episodes), "solves": n_solves, "solve_set": sp["solves"],
                "releases": sum(r["outcome"] == "released" for r in data["records"]), "documents": len(chosen),
                "doc_kind": sp["kind"], "doc_files": sp["files"], "doc_ids_fingerprint": core.ids_fingerprint(doc_ids),
                "first_doc": doc_ids[0] if doc_ids else None, "last_doc": doc_ids[-1] if doc_ids else None,
                "items": sum(len(t) for t in data["turns"]),
                "tokens": sum(len(c) for t in data["turns"] for _, c in t),
                "solve_items": sum(len(t) for t in solve_turns),
                "solve_tokens": sum(len(c) for t in solve_turns for _, c in t), "koan_solve_tokens": koan_solve_tokens,
                "doc_tokens": sum(len(it[1]) for _, it in chosen),
                "doc_referents": dict(sorted(Counter(r["referent"] for r, _ in chosen).items())),
                "assertion_1_checked": sum(checked[f] for f in sp["files"]), "mixed_batch_checked": mixed,
                "tokenizer": MODEL, "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id,
                "core_sha256": core.sha_file(core.CORE_FILE), "app_sha256": app_sha(), **record}
        write_json(out / "data.json", info)
        infos[lab] = info
        print(json.dumps({k: info[k] for k in ("label", "solves", "documents", "items", "tokens")}), flush=True)
    runs.commit()
    return {"infos": infos, "files": {lab: (Path(ROOT) / root / "sets" / lab / "data.json").read_text() for lab in labels}}


def doc_nll(model, tok, items: list, micro: int = 8) -> list[tuple[float, int]]:
    """Per document item: (summed negative log-likelihood of its completion tokens, the EOS included; their count),
    under `model`, shortest first, no grad (modal_durability.release_nll's pattern)."""
    import torch

    from player import pack, token_logps

    dev = next(model.parameters()).device
    out = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    model.eval()
    with torch.no_grad():
        for k in range(0, len(order), micro):
            idx = order[k:k + micro]
            full, attn, cm, wc = pack([items[i] for i in idx], tok.pad_token_id, dev)
            lp = token_logps(model, full, attn, wc)
            sums, counts = (-(lp * cm).sum(1)).tolist(), cm.sum(1).tolist()
            for i, s, n in zip(idx, sums, counts):
                out[i] = (float(s), int(n))
    return out


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=3 * 3600)
def belief_read(adapter: str | None, out_rel: str, items: list, nll_sets: dict, seed: int,
                samples: int = core.PROBE_SAMPLES, temperature: float = core.PROBE_TEMPERATURE,
                max_new: int = core.PROBE_MAX_NEW, gen_batch: int = 32, micro: int = 8) -> dict:
    """The belief probe and the document NLL on one adapter (None = the base model). Probe: every item under each
    condition (declared_core.PROBE_CONDITIONS), `samples` times, at `temperature`, up to `max_new` new tokens, random
    and torch seeded with `seed`; probe.jsonl. NLL: per set name, the first n documents of a file under ROOT, each a
    document item; the mean NLL per token (summed over the documents' tokens, the EOS included) and the mean of
    per-document means; nll.json."""
    import torch
    from anatta.models import chat_ids

    from player import Player

    runs.reload()
    out = Path(ROOT) / out_rel
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    model, tok, _ = load_model(adapter=adapter)
    player = Player(model, tok, (max_new,), gen_batch)
    asks = [(it, cond, k) for it in items for cond in core.PROBE_CONDITIONS for k in range(samples)]
    prompts = [chat_ids(tok, core.probe_messages(it["question"], cond))[0] for it, cond, _ in asks]
    t0 = time.time()
    comps, texts = player.generate(prompts, temperature, max_new)
    stops = set(player.stop.tolist())
    rows = [{"adapter": adapter, "condition": cond, "item": it["item"], "category": it["category"],
             "referent": it["referent"], "form": it["form"], "sample": k,
             "question": core.as_asked(it["question"], cond), "answer": x, "tokens": len(c),
             "ended": bool(len(c)) and int(c[-1]) in stops}
            for (it, cond, k), c, x in zip(asks, comps, texts)]
    write_jsonl(out / "probe.jsonl", rows)
    probe_seconds = round(time.time() - t0)
    nll = {}
    for name, (rel, n) in nll_sets.items():
        docs = read_rows((Path(ROOT) / rel).read_text())[:n]
        ids = tok([d["text"] for d in docs], add_special_tokens=False)["input_ids"]
        per = doc_nll(model, tok, [core.doc_item(tok.bos_token_id, tok.eos_token_id, x) for x in ids], micro)
        total, count = sum(s for s, _ in per), sum(c for _, c in per)
        nll[name] = {"file": rel, "docs": len(docs), "tokens": count, "nll_per_token": total / count,
                     "mean_doc_nll": sum(s / c for s, c in per) / len(per),
                     "per_doc": [[d["id"], s, c] for d, (s, c) in zip(docs, per)]}
    res = {"adapter": adapter, "seed": seed, "samples": samples, "temperature": temperature, "max_new": max_new,
           "items": len(items), "conditions": list(core.PROBE_CONDITIONS), "probe_rows": len(rows),
           "probe_seconds": probe_seconds, "nll": nll, "core_sha256": core.sha_file(core.CORE_FILE),
           "app_sha256": app_sha()}
    write_json(out / "nll.json", res)
    runs.commit()
    print(json.dumps({"adapter": adapter, "probe_rows": len(rows),
                      "nll": {k: round(v["nll_per_token"], 4) for k, v in nll.items()}}), flush=True)
    return {"files": {f: (out / f).read_text() for f in BELIEF_FILES}}


# ---------------------------------------------------------------------------------------------------- the laptop


def save_files(dest: Path, files: dict) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        if text is not None:
            (dest / name).write_text(text)
    return dest


def upload(local_files: dict) -> dict:
    """Files to the anatta-runs volume: {relative path under /runs/h4lite: local path}. A file already on the volume
    must hold the same bytes (nothing is overwritten); the missing ones are put, then every sha is checked."""
    want = {rel: core.sha_file(p) for rel, p in local_files.items()}
    have = file_shas.remote(list(want))
    put = []
    for rel, sha in want.items():
        if have.get(rel) is None:
            put.append(rel)
        elif have[rel] != sha:
            raise SystemExit(f"/runs/h4lite/{rel} differs from {local_files[rel]}: not overwritten, nothing built")
    if put:
        with runs.batch_upload(force=False) as batch:
            for rel in put:
                batch.put_file(str(local_files[rel]), f"/h4lite/{rel}")
        print(f"uploaded {len(put)} files to /runs/h4lite/: {put}", flush=True)
    after = file_shas.remote(list(want))
    if after != want:
        raise SystemExit(f"the volume's files do not match after the upload: {[r for r in want if after.get(r) != want[r]]}")
    return want


def doc_record_for(smoke: bool, ext: bool = False) -> tuple[dict, dict]:
    """The document files to upload ({rel: local path}) and the record data.json carries: config, manifest(s) and
    document files' shas (each file's as the manifest lists it)."""
    o = out_dir(smoke) / "docs"
    m = json.loads((o / "manifest.json").read_text())
    files = dict(m["files"])
    rec = {"config_sha256": core.sha_file(HERE / "config.json"), "manifest_sha256": core.sha_file(o / "manifest.json")}
    if ext:
        mx = json.loads((o / "manifest_ext.json").read_text())
        files.update(mx["files"])
        rec["manifest_ext_sha256"] = core.sha_file(o / "manifest_ext.json")
    for name, v in files.items():
        if core.sha_file(o / name) != v["sha256"]:
            raise SystemExit(f"{o / name} differs from the manifest's sha: nothing uploaded")
    rec["doc_shas"] = {name: v["sha256"] for name, v in files.items()}
    return {f"{vol(smoke)}/docs/{name}": o / name for name in files}, rec


def judge_passed(smoke: bool, ext: bool = False) -> tuple[bool, str]:
    """Whether the judge's gate passed on these documents: judge.json tied to manifest.json, or with ext judge_ext.json
    (D30k's sample, code check should-fix 7) tied to manifest_ext.json."""
    o = out_dir(smoke) / "docs"
    if smoke:
        return True, "the smoke is not judged before its fine-tunes (its probe answers are)"
    jf, mf = ("judge_ext.json", "manifest_ext.json") if ext else ("judge.json", "manifest.json")
    if not (o / jf).exists():
        return False, f"no {o / jf}: run gen_declarations.py judge{' --ext' if ext else ''}"
    j = json.loads((o / jf).read_text())
    if not (o / mf).exists() or j.get("manifest_sha256") != core.sha_file(o / mf):
        return False, f"{jf} was written for other documents"
    return bool(j["gate"]["ok"]), "; ".join(j["gate"]["why"]) or "the gate passed"


def screen_refusal(o: Path, ext: bool = False) -> str | None:
    """The screen's guard before any training (amendment 13, must-fix 1; the smoke is not screened): None when
    docs/screen.jsonl is the file manifest.json sealed (with ext, manifest_ext.json's seal too; a seal is a sha256
    over the bytes the manifest saw, declared_core.screen_seal_ok) and holds a final, parsed, clean verdict for the
    current text sha of every kept declaration in declared.jsonl and held_declared.jsonl (with ext, declared_ext.jsonl)
    and, when the neutral screen ran, of every kept document in neutral.jsonl and held_neutral.jsonl; else why not."""
    docs = o / "docs"
    spath = docs / core.SCREEN_FILE
    names = ["manifest.json"] + (["manifest_ext.json"] if ext else [])
    seals = {}
    for name in names:
        if not (docs / name).exists():
            return f"docs/{name} is missing"
        seal = json.loads((docs / name).read_text()).get("screen")
        if not seal:
            return f"docs/{name} records no screen (built before amendment 13)"
        if not core.screen_seal_ok(spath, seal):
            return (f"docs/{core.SCREEN_FILE} is not the file {name} sealed (sha256 {str(seal.get('sha256'))[:16]}..., "
                    f"{seal.get('bytes')} bytes)")
        seals[name] = seal
    files = [(core.DOC_FILES[("declared", "train")], "declared"), (core.DOC_FILES[("declared", "held")], "declared")]
    if ext:
        files.append((core.DOC_FILES[("declared", "ext")], "declared"))
    if seals["manifest.json"].get("neutral"):
        files += [(core.DOC_FILES[("neutral", "train")], "neutral"), (core.DOC_FILES[("neutral", "held")], "neutral")]
    problems = core.screen_problems(docs, files, core.read_screen(spath))
    if problems:
        return (f"{len(problems)} kept document{'s' if len(problems) > 1 else ''} without a final, clean screen verdict "
                f"for {'their' if len(problems) > 1 else 'its'} text: " + "; ".join(problems[:5])
                + ("; ..." if len(problems) > 5 else ""))
    return None


def stale_sets(o: Path) -> list[str]:
    """The on-disk sets (sets/*/data.json) built from other documents than the manifest now on disk (or, for a set
    that recorded one, than manifest_ext.json)."""
    docs = o / "docs"
    man, ext = ((core.sha_file(docs / f) if (docs / f).exists() else None) for f in ("manifest.json", "manifest_ext.json"))
    stale = []
    for path in sorted((o / "sets").glob("*/data.json")) if (o / "sets").exists() else []:
        d = json.loads(path.read_text())
        if d.get("manifest_sha256") != man or ("manifest_ext_sha256" in d and d["manifest_ext_sha256"] != ext):
            stale.append(path.parent.name)
    return stale


def refuse_stale(o: Path) -> None:
    stale = stale_sets(o)
    if stale:
        raise SystemExit(f"sets {stale} were built from other documents than {o / 'docs'} now holds: nothing built or "
                         "trained. A deliberate rebuild needs the old sets, adapters and reads deleted, on disk and "
                         "on the volume (README)")


def check_solve_tokens(o: Path, labels: list[str], smoke: bool) -> None:
    """Prints each set's solve tokens; a main set with solves must carry S's core.KOAN_SOLVE_TOKENS (code check,
    should-fix 9, with the corrected count)."""
    bad = []
    for lab in labels:
        d = json.loads((o / "sets" / lab / "data.json").read_text())
        print(f"{lab}: solve_tokens {d['solve_tokens']} ({d['solves']} solves)", flush=True)
        if not smoke and core.set_spec(lab)["solves"] and d["solve_tokens"] != core.KOAN_SOLVE_TOKENS:
            bad.append(f"{lab} ({d['solve_tokens']})")
    if bad:
        raise SystemExit(f"S's 463 solves carry {core.KOAN_SOLVE_TOKENS} tokens; {', '.join(bad)} do not: the koan "
                         "set read is not the registered one")


def finetune_jobs(jobs: list) -> None:
    """modal_sft.finetune calls in parallel: jobs are (name, local dest, kwargs, timeout or None). A job whose
    done.json is already on disk is skipped; one finished on the volume but not copied is fetched instead of rerun."""
    todo, fetch = [], []
    rels = {name: kw["out_dir"].removeprefix(f"{ROOT}/") for name, _, kw, _ in jobs}
    status = volume_status.remote([f"{r}/done.json" for r in rels.values()]) if jobs else {}
    for name, dest, kw, timeout in jobs:
        if (dest / "done.json").exists():
            print(f"{name}: done (on disk)", flush=True)
        elif status.get(f"{rels[name]}/done.json"):
            fetch.append((name, dest))
        else:
            fn = modal_sft.finetune.with_options(timeout=timeout) if timeout else modal_sft.finetune
            todo.append((name, dest, fn.spawn(**kw)))
            print(f"spawned {name}: {json.dumps({k: v for k, v in kw.items() if k != 'label'})}"
                  + (f", timeout {timeout // 3600} h" if timeout else ""), flush=True)
    for name, dest in fetch:
        files = read_files.remote([f"{rels[name]}/{f}" for f in TRAIN_FILES])
        save_files(dest, {Path(k).name: v for k, v in files.items()})
        print(f"{name}: done on the volume, copied to {dest}", flush=True)
    for name, dest, call in todo:
        res = call.get()
        print(f"{name}: wrote {save_files(dest, res['files'])}", flush=True)


def ft_job(label: str, set_name: str, seed: int, smoke: bool, init: str = "new", **over) -> tuple:
    """One fine-tune: the set's folder in, train/NAME_sSEED out, Stage A's settings unless `over` says otherwise
    (only the smoke's assertion-2 runs do)."""
    d = f"{ROOT}/{vol(smoke)}"
    name = f"{label}_s{seed}"
    kw = dict(label=label, seed=seed, **{**core.FINETUNE, **over}, src_dir=f"{d}/sets/{set_name}",
              out_dir=f"{d}/train/{name}")
    if init != "new":
        kw["init"] = init
    return name, out_dir(smoke) / "train" / name, kw, (SMOKE_TIMEOUT if smoke else TIMEOUTS.get(label))


def train_jobs(arms: list[str], smoke: bool = False) -> tuple[list, list]:
    """(first wave, second wave): every arm's fine-tunes from a fresh LoRA, Dseq's first stage among them; then Dseq's
    second stage, S's set (the smoke: S16) from that stage's adapter."""
    first, second = [], []
    for arm in arms:
        seeds = (0,) if smoke else core.ARMS[arm]["seeds"]
        for s in seeds:
            if arm == "Dseq":
                first.append(ft_job(core.DSEQ_DOCS, core.DSEQ_DOCS, s, smoke))
                second.append(ft_job("Dseq", "S16" if smoke else "S", s, smoke,
                                     init=f"{ROOT}/{vol(smoke)}/train/{core.DSEQ_DOCS}_s{s}/adapter"))
            else:
                first.append(ft_job(arm, arm, s, smoke))
    return first, second


def assert2_jobs() -> list:
    """The smoke's assertion 2: one optimizer step on mix16's 16 items at micro-batch 1 and at micro-batch 4."""
    return [ft_job(f"mix16_micro{m}", "mix16", 0, True, micro=m, epochs=1) for m in (1, 4)]


def run_reads(labels: list[str], smoke: bool = False, arm: str = core.HELDOUT["arm"], sub: str = "reads",
              pairs_file: str | None = None) -> dict:
    """One batch of held-out reads (modal_retreat2.evaluate_adapter): durability's call with only the adapter swapped
    (the smoke passes its 4-pair file; noexit swaps the arm). A label read on disk is skipped; one on the volume is
    copied, not rerun."""
    h = core.HELDOUT
    o = out_dir(smoke) / sub
    rels = {lab: f"{vol(smoke)}/{sub}/{lab}" for lab in labels}
    todo = [lab for lab in labels if not all((o / lab / f).exists() for f in READ_FILES)]
    status = volume_status.remote([f"{rels[lab]}/{f}" for lab in todo for f in READ_FILES]) if todo else {}
    calls, out = [], {}
    for lab in labels:
        if lab not in todo:
            print(f"{sub}/{lab}: read already (on disk)", flush=True)
            continue
        dest, rel = o / lab, rels[lab]
        if all(status.get(f"{rel}/{f}") for f in READ_FILES):
            files = read_files.remote([f"{rel}/{f}" for f in READ_FILES])
            save_files(dest, {Path(k).name: v for k, v in files.items()})
            print(f"{sub}/{lab}: read already on the volume, copied to {dest}", flush=True)
            continue
        args = (adapter_path(lab, smoke), arm, rel, "heldout", h["eval_n"], h["temperature"], h["attempts"], MAX_NEW, 32,
                h["eval_seed"], h["under_cut"])
        calls.append((lab, dest, rel, modal_retreat2.evaluate_adapter.spawn(*args, pairs_file)
                      if pairs_file else modal_retreat2.evaluate_adapter.spawn(*args)))
    for lab, dest, rel, call in calls:
        res = call.get()
        files = read_files.remote([f"{rel}/{f}" for f in READ_FILES])
        save_files(dest, {Path(k).name: v for k, v in files.items()})
        ev = res["eval"]
        out[lab] = ev
        print(f"{sub}/{lab}: release {ev['release']} (grounded {ev['release_grounded']}), hack {ev['hack']}, solve "
              f"{ev['solve']}, over-release {ev['over_release']} -> {dest}", flush=True)
    return out


def nll_sets_for(smoke: bool) -> dict:
    d = f"{vol(smoke)}/docs"
    if smoke:
        return {"held_declared": (f"{d}/held_declared.jsonl", SMOKE_NLL_DOCS),
                "held_neutral": (f"{d}/held_neutral.jsonl", SMOKE_NLL_DOCS)}
    return {"held_declared": (f"{d}/held_declared.jsonl", NLL_DOCS), "held_neutral": (f"{d}/held_neutral.jsonl", NLL_DOCS),
            "trained_declared": (f"{d}/declared.jsonl", NLL_DOCS), "trained_neutral": (f"{d}/neutral.jsonl", NLL_DOCS)}


def run_belief(labels: list[str], smoke: bool, seed: int) -> None:
    """belief_read on each label (the base model as "base"); one on disk is skipped, one on the volume copied."""
    o = out_dir(smoke) / "belief"
    rels = {lab: f"{vol(smoke)}/belief/{lab}" for lab in labels}
    todo = [lab for lab in labels if not all((o / lab / f).exists() for f in BELIEF_FILES)]
    status = volume_status.remote([f"{rels[lab]}/{f}" for lab in todo for f in BELIEF_FILES]) if todo else {}
    items, nll_sets, calls = core.probe_items(smoke), nll_sets_for(smoke), []
    for lab in todo:
        if all(status.get(f"{rels[lab]}/{f}") for f in BELIEF_FILES):
            files = read_files.remote([f"{rels[lab]}/{f}" for f in BELIEF_FILES])
            save_files(o / lab, {Path(k).name: v for k, v in files.items()})
            print(f"belief/{lab}: read already on the volume, copied", flush=True)
            continue
        calls.append((lab, belief_read.spawn(adapter_path(lab, smoke), rels[lab], items, nll_sets, seed)))
    for lab, call in calls:
        res = call.get()
        print(f"belief/{lab}: wrote {save_files(o / lab, res['files'])}", flush=True)


@app.local_entrypoint()
def sets(smoke: bool = False, ext: bool = False):
    """Uploads the documents and builds the training sets (build_sets); data.json copied to results/.../sets/SET/."""
    o = out_dir(smoke)
    ok, why = judge_passed(smoke)
    if not ok:
        raise SystemExit(f"the judge's gate has not passed ({why}): nothing is trained on these documents")
    if ext:
        dec = o / "decide_d30k.json"
        if smoke or not dec.exists() or not json.loads(dec.read_text()).get("run"):
            raise SystemExit("D30k's set is built only when decide_d30k.json says run (declared_grade.py --decide-d30k)")
        ok, why = judge_passed(smoke, ext=True)
        if not ok:
            raise SystemExit(f"the judge's gate on D30k's documents has not passed ({why}): D30k is not trained on them")
    if not smoke:
        why = screen_refusal(o, ext)
        if why:
            raise SystemExit(f"the screen's guard refuses (amendment 13): {why}. Nothing is built or trained")
    labels = list(core.EXT_SETS if ext else (core.SMOKE_SETS if smoke else core.MAIN_SETS))
    refuse_stale(o)
    if all((o / "sets" / lab / "data.json").exists() for lab in labels):
        print(f"sets {labels}: built already (on disk)")
        check_solve_tokens(o, labels, smoke)
        return
    files, rec = doc_record_for(smoke, ext)
    upload(files)
    res = build_sets.remote(labels, vol(smoke), smoke, rec)
    for lab, text in res["files"].items():
        save_files(o / "sets" / lab, {"data.json": text})
    for lab, info in res["infos"].items():
        print(f"{lab}: {info['solves']} solves + {info['documents']} {info['doc_kind'] or ''} documents, "
              f"{info['items']} items, {info['tokens']} tokens", flush=True)
    check_solve_tokens(o, labels, smoke)


@app.local_entrypoint()
def train_arms(arms: str = "", smoke: bool = False):
    """The fine-tunes: every arm of MAIN_ARMS (or --arms, e.g. D30k), Dseq's second stage after its first."""
    arm_list = names_of(arms) or list(core.MAIN_ARMS)
    if "D30k" in arm_list:
        dec = out_dir(smoke) / "decide_d30k.json"
        if not dec.exists() or not json.loads(dec.read_text()).get("run"):
            raise SystemExit("D30k is trained only when decide_d30k.json says run")
    first, second = train_jobs(arm_list, smoke)
    need = {Path(kw["src_dir"]).name for _, _, kw, _ in first + second}
    missing = [s for s in need if not (out_dir(smoke) / "sets" / s / "data.json").exists()]
    if missing:
        raise SystemExit(f"sets {missing} are not built: run ::sets first")
    finetune_jobs(first)
    finetune_jobs(second)


@app.local_entrypoint()
def read_arms(labels: str = ""):
    """The held-out reads (default: the 13 adapters of the main arms)."""
    run_reads(names_of(labels) or main_labels())


@app.local_entrypoint()
def belief(labels: str = ""):
    """The probe and the NLL read (default: the 13 adapters, Dseq_docs_s0/s1 and the base model)."""
    run_belief(names_of(labels) or belief_labels(), False, load_config()["seeds"]["probe"])


@app.local_entrypoint()
def noexit(labels: str = ""):
    """The held-out read under arm noexit at the go arm(s) (declared_grade.py --go-labels names them)."""
    labs = names_of(labels)
    if not labs:
        print("no go arm: there is no noexit read to run")
        return
    run_reads(labs, arm=core.NOEXIT_ARM, sub="reads_noexit")


def smoke_pairs_file(o: Path) -> Path:
    """The smoke's 4 held-out pairs (the first 4 by name of Gate 0's held split), as a pairs file."""
    rows = [json.loads(x) for x in (RESULTS / "gate0" / "pairs.jsonl").read_text().splitlines() if x.strip()]
    held = sorted((r for r in rows if r.get("split") == "held"), key=lambda r: r["name"])[:SMOKE_PAIRS]
    path = o / "pairs4.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in held))
    return path


def first_nll(log: Path) -> float | None:
    """The first optimizer step's loss in a fine-tune's log.jsonl (None when the log is missing or empty)."""
    lines = read_rows(log.read_text()) if log.exists() else []
    return lines[0]["nll"] if lines and lines[0].get("opt_step") == 1 else None


def smoke_checks(o: Path) -> dict:
    """What the smoke must show (smoke_report.json): the sets' sizes and mixing, assertion 1 on every document item,
    the step counts, Stage A's settings and Dseq's warm start, assertion 2, the reads' settings, the belief read's
    rows and documents."""
    rd = (lambda p: json.loads(p.read_text()) if p.exists() else None)
    c = {}
    s, d1, dq, s16, mix = (rd(o / "sets" / x / "data.json") for x in core.SMOKE_SETS)
    c["sets"] = bool(s and d1 and dq and s16 and mix) and (
        s["items"] == core.SMOKE_S_ITEMS and s["documents"] == 0 and s["releases"] == 0
        and d1["items"] == core.SMOKE_S_ITEMS + core.SMOKE_DOCS["train"] and d1["solves"] == s["solves"]
        and d1["documents"] == core.SMOKE_DOCS["train"] and d1["doc_kind"] == "declared"
        and dq["solves"] == 0 and dq["documents"] == core.SMOKE_DOCS["train"]
        and s16["items"] == core.SMOKE_S16_ITEMS and mix["items"] == 16 and mix["documents"] == 8
        and mix["solve_items"] == 8)
    c["koan_solve_tokens"] = bool(s) and s.get("koan_solve_tokens") == core.KOAN_SOLVE_TOKENS  # S's 463, on the real set
    c["assertion_1"] = bool(d1 and dq and mix) and all(x["assertion_1_checked"] >= x["documents"] > 0 and x["mixed_batch_checked"]
                                                        for x in (d1, dq, mix))
    steps = {}
    for name, want in (("S_s0", 10), ("D1k_s0", 10), (f"{core.DSEQ_DOCS}_s0", 2), ("Dseq_s0", 2)):
        done, args = rd(o / "train" / name / "done.json"), rd(o / "train" / name / "args.json")
        steps[name] = done and done["opt_steps"]
        c[f"steps_{name}"] = bool(done) and done["opt_steps"] == want
        c[f"settings_{name}"] = bool(args) and all(args.get(k) == v for k, v in core.FINETUNE.items())
    dargs = rd(o / "train" / "Dseq_s0" / "args.json")
    c["dseq_warm_start"] = bool(dargs) and dargs.get("init") == f"{ROOT}/declared_smoke/train/{core.DSEQ_DOCS}_s0/adapter"
    n1, n4 = (first_nll(o / "train" / f"mix16_micro{m}_s0" / "log.jsonl") for m in (1, 4))
    c["assertion_2"] = None not in (n1, n4) and abs(n1 - n4) <= ASSERT2_TOL
    for sub, arm in (("reads", "koan"), ("reads_noexit", "noexit")):
        m = rd(o / sub / "D1k_s0" / "heldout.json")
        c[f"read_{sub}"] = bool(m) and (m["arm"], m["held_pairs"], m["episodes"], m["eval_seed"], m["adapter"]) == (
            arm, SMOKE_PAIRS, SMOKE_PAIRS * 2 * core.HELDOUT["eval_n"], core.HELDOUT["eval_seed"],
            f"{ROOT}/declared_smoke/train/D1k_s0/adapter")
    for lab in ("D1k_s0", "base"):
        nl = rd(o / "belief" / lab / "nll.json")
        probe = o / "belief" / lab / "probe.jsonl"
        n_rows = len([x for x in probe.read_text().splitlines() if x.strip()]) if probe.exists() else 0
        c[f"belief_{lab}"] = bool(nl) and n_rows == len(core.SMOKE_PROBE) * len(core.PROBE_CONDITIONS) * core.PROBE_SAMPLES \
            and sum(v["docs"] for v in nl["nll"].values()) == 2 * SMOKE_NLL_DOCS
    return {"ok": all(c.values()), "checks": c, "steps": steps, "assertion_2_nll": {"micro1": n1, "micro4": n4,
                                                                                  "tolerance": ASSERT2_TOL},
            "koan_solve_tokens": {"set": s and s.get("koan_solve_tokens"), "registered": core.KOAN_SOLVE_TOKENS}}


@app.local_entrypoint()
def smoke():
    """Every container path at toy sizes (module docstring); results/declared_smoke/smoke_report.json."""
    o = out_dir(True)
    if not (o / "docs" / "manifest.json").exists():
        raise SystemExit(f"no {o / 'docs' / 'manifest.json'}: run gen_declarations.py smoke first")
    refuse_stale(o)
    pairs = smoke_pairs_file(o)
    files, rec = doc_record_for(True)
    upload({**files, "declared_smoke/pairs4.jsonl": pairs})
    if not all((o / "sets" / lab / "data.json").exists() for lab in core.SMOKE_SETS):
        res = build_sets.remote(list(core.SMOKE_SETS), "declared_smoke", True, rec)
        for lab, text in res["files"].items():
            save_files(o / "sets" / lab, {"data.json": text})
    first, second = train_jobs(["S", "D1k", "Dseq"], smoke=True)
    finetune_jobs(first + assert2_jobs())
    finetune_jobs(second)
    pf = f"{ROOT}/declared_smoke/pairs4.jsonl"
    run_reads(["D1k_s0"], smoke=True, pairs_file=pf)
    run_reads(["D1k_s0"], smoke=True, arm=core.NOEXIT_ARM, sub="reads_noexit", pairs_file=pf)
    run_belief(["D1k_s0", "base"], True, load_config()["seeds"]["probe"])
    rep = smoke_checks(o)
    rep["utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    (o / "smoke_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print("SMOKE " + ("PASSED" if rep["ok"] else "FAILED") + f" -> {o / 'smoke_report.json'}")
    if not rep["ok"]:
        raise SystemExit(f"the smoke failed: {[k for k, v in rep['checks'].items() if not v]}")


@app.local_entrypoint()
def status(smoke: bool = False):
    """What is done: the volume (sets, adapters, reads) and the local mirror."""
    v, o = vol(smoke), out_dir(smoke)
    sets_ = core.SMOKE_SETS if smoke else core.MAIN_SETS + core.EXT_SETS
    labs = main_labels() + labels_of(core.DSEQ_DOCS) + labels_of("D30k")
    st = volume_status.remote([f"{v}/sets/{s}/train.pt" for s in sets_] + [f"{v}/train/{lab}/done.json" for lab in labs]
                              + [f"{v}/reads/{lab}/heldout.json" for lab in labs])
    for k, x in st.items():
        print(f"volume {k}: {'yes' if x else 'no'}")
    for p in sorted(o.rglob("*.json")) if o.exists() else []:
        print("disk", p.relative_to(o))
