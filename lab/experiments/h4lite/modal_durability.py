"""The durability test, predictions 135-139 (lab/ledger/NOTES.md, "Predictions 135-139, registered: durability, the
frame's last prediction (c)", Opus 5.5, 07:58 UTC on 30 Oct), and its second try with told's pool topped up,
predictions 148-153 ("Predictions 148-153, registered: durability, one more try at (c), with told's pool topped up
where it ran short", 17:06 UTC on 30 Oct). Recognition track; private.

Arms, three seeds each: koan_s0-s2, Stage A's adapters unchanged, and told_s0-s2, trained on koan's training set
with its 102 release episodes replaced by releases sampled under koan's prompt plus one oracle sentence, the
sentence removed from the training prompts, matched by the seeded matcher in durability_core.py. Then each
adapter's own LoRA keeps training (E2 decides, E3 is the fallback, E1 is reported) and modal_retreat2.py's held-out
procedure (prediction 62's) reads release before and after. durability_grade.py grades from the local files.

Stages (local entrypoints; durability.sh runs them in order and every one resumes from what is on disk):
  sample_told  told_sample on a GPU: koan's 72 pairs (the pairs of the 102 releases in /runs/h4lite/sft/koan/train.pt),
               impossible twins, 32 samples per pair at each of the 3 placements (6,912 episodes; the pre-launch entry
               doubles the registered 16), koan's prompt and the oracle sentence (durability_core.OracleTask), Stage A's
               sampling settings (temperature 1.0, 3 attempts, (512, 384) new tokens: read from koan's first Stage A
               shard, and sampling refuses on any difference), Stage A's per-shard seeds. The sha256 of this file is
               recorded on the pool and at the build (app_sha). A candidate is a strict release at the oracle's turn;
               for each one the seen history must rebuild generation's prompt ids exactly, its training turns are
               rebuilt from the plain history (koan's form: only the release turn's prompt changes), and the release
               turn's NLL is taken under the base model with that oracle-free prompt, beside the NLL of koan's 102
               release turns under their own prompts, in the same container. Writes durability/told_pool/{shard_*.pt,
               pool.pt, episodes.jsonl, nll.json, summary.json}.
  topup        predictions 148-153: the first pool's build ended in a MatchError (its match_failed.json stays in
               told/), so told's pool is topped up in rounds, at most 6, build_told after each. Round K (told_topup,
               one GPU container) starts from the pool so far (round 1: the first pool, unchanged) and takes its
               cells by the registered rule (topup_cells: a cell's quota is durability_core.borrow on koan's cells
               over its 5-gram-free candidates, its count its phrase-free ones; the short cells, below their quota,
               else the tight ones, at it, else the top-up stops). It samples `samples` (32) episodes at each cell's
               pair and placement through told_sample's own loop (play_shard), the episode index going on from
               6,912 and a shard starting at index i seeding random and torch with i, then rescores every candidate
               (the first pool's and the top-up's, each round's in a release_nll call of its own, so each meets the
               batches it was first scored in) and koan's 102; if one of koan's 102 or the first pool's 1,767 moves
               more than 0.01 from the first pool's nll.json, the top-up stops. The merged pool
               records two app shas, the first pool's and this file's (app_shas). Then build_told on it, as `build`
               runs it, its files in told/rounds/K/: built, the set is copied to told/ and the run goes on; a
               MatchError whose report has the token trade, or a stop, ends the top-up, and 148-153 are not read;
               any other MatchError starts round K+1. Writes durability/told_pool_topup/rounds/K/{shard_*.pt,
               pool.pt, episodes.jsonl, nll.json, summary.json} (stopped.json for a stop) and durability/told/rounds/K/.
  build        build_told on CPU: durability_core.match (the frozen matcher) on the pool against koan's 102, then the
               training set in koan's order (durability_core.assemble): koan's 463 solves with their token ids, the
               matched releases in the release slots, first-try solves traded for the token total. --n is the gate's
               count (label told_g), built from the pool told's set came from (told/data.json's pool_rel: the
               topped-up pool). match.json also counts the release replies, told's and koan's, with the word "note",
               and gives the phrase, grounded and median-length edges beside koan's, the release-turn token totals
               and each told release's round (0 = the first pool) with its cell's episode count (reported only). A
               build that ended in a MatchError is not run again: its match_failed.json is kept. Writes
               durability/LABEL/{train.pt, data.json, match.json, releases.jsonl}.
  judge        on the laptop through OpenRouter (OPENROUTER_API_KEY, never printed): every release reason of koan's set
               and of LABEL's, deduplicated, shuffled, unlabelled, read as citing the specific test or making a
               generic claim. Reported, not a filter.
  train_told   modal_sft.finetune on durability/LABEL: Stage A's settings (LoRA rank 16, lr 1e-4, 2 epochs, loss on
               replies), seeds 0-2, durability/LABEL_sSEED/adapter.
  read         modal_retreat2.evaluate_adapter under the koan arm (4 samples a twin, 3 attempts, temperature 0.7, eval
               seed 3000) on every label of a phase in one batch: start (koan_s0-2 beside told_s0-2), start_gate (koan
               beside told_g), e2_epoch1, e2_epoch2, e1, e3. durability/reads/PHASE/LABEL/heldout.{json,jsonl}; a read
               already on disk is skipped, and one already on the volume is copied, not rerun.
  erode_e2     koan's 992 sampled solve episodes (no releases) as one set (stage_e2), then modal_sft.finetune
               continuing each adapter's own LoRA with its seed (init = the adapter, 2 epochs, lr 1e-4, adapters saved
               after epochs 1 and 2), then the e2_epoch1 and e2_epoch2 reads.
  erode_e1     the base model's own replies (e1_generate: no system message of our own, so the chat template's default;
               up to 1,024 new tokens, temperature 0.7, seed 0; a reply that hits the cap without a stop token is
               dropped, since the registration asks for untruncated replies) to the positive control's 1,500 prompts
               minus coding requests (durability_core.is_coding_request), then the same continuation and the e1 read.
  erode_e3     modal_retreat2.train under the new arm koan_rlvr (koan's prompt, rlvr's reward table) from each of the six
               E2-free adapters, 50 steps, KL 0.02 to the start, other settings unchanged (runs durability_e3_LABEL),
               then the e3 read of adapter_50. Only when told did not bite under E2 (durability_grade.py --decide-e3).
  smoke        every container path of 135-139 at toy sizes under durability_smoke/ (never the real paths); the
               top-up's own smoke is `topup --smoke` (2 rounds, 2 episodes a cell, on the smoke's pool).
  status       what is done, on the volume and on disk.

Volume: /runs/h4lite/durability/... (and /runs/h4lite/retreat2/durability_e3_*); local mirror: results/durability/.

Usage (~/prj/matlab/.venv/bin/modal, from ~/prj/align/anatta-lab; the file has several entrypoints, name one):
  modal run lab/experiments/h4lite/modal_durability.py::smoke
  modal run lab/experiments/h4lite/modal_durability.py::topup --smoke
  modal run --detach lab/experiments/h4lite/modal_durability.py::sample_told
  modal run --detach lab/experiments/h4lite/modal_durability.py::topup
  modal run lab/experiments/h4lite/modal_durability.py::build --n N
  modal run lab/experiments/h4lite/modal_durability.py::judge
  modal run --detach lab/experiments/h4lite/modal_durability.py::train_told
  modal run --detach lab/experiments/h4lite/modal_durability.py::read --phase start
  modal run --detach lab/experiments/h4lite/modal_durability.py::erode_e2
  modal run --detach lab/experiments/h4lite/modal_durability.py::erode_e1
  modal run --detach lab/experiments/h4lite/modal_durability.py::erode_e3
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path

import modal

import durability_core as dc
from modal_common import (CACHE, GPU, H4, MAX_NEW, RESULTS, ROOT, RUNS, cache, image as base_image, load_model,
                          pairs_by_name, prepare_pairs, read_rows, runs, write_json, write_jsonl)

# the shared image plus the frozen core; the shared image stays as it is
image = base_image.add_local_file(str(H4 / "durability_core.py"), "/root/durability_core.py")
app = modal.App("anatta-h4lite-durability", image=image)
if modal.is_local():  # containers mount only this file and the image's modules, so the includes stay on the laptop
    import modal_retreat2
    import modal_sft

    app.include(modal_sft.app)  # finetune
    app.include(modal_retreat2.app)  # train (E3), evaluate_adapter (the reads)

PAIRS = f"{ROOT}/gate0/pairs.jsonl"
KOAN = f"{ROOT}/sft/koan"
PROMPTS = RESULTS / "positive_control" / "adapter" / "adapter" / "prompts.jsonl"
PHASES = ("start", "start_gate", "e2_epoch1", "e2_epoch2", "e1", "e3")
E3_PREFIX = "durability_e3_"
TRAIN_FILES = ("args.json", "log.jsonl", "done.json")
READ_FILES = ("heldout.json", "heldout.jsonl")
TOLD_FILES = ("train.pt", "data.json", "match.json", "releases.jsonl")  # a successful build's files
ROUND_FILES = ("summary.json", "nll.json", "episodes.jsonl", "stopped.json")  # a top-up round's text files
TOPUP_ROUNDS, NLL_TOL = 6, 0.01  # predictions 148-153: at most 6 rounds; the NLL pass may move nothing more than 0.01
# the smoke's sizes: enough koan pairs for ten deciles, two samples a placement, a few prompts at E1's registered
# cap (a lower cap would drop most replies and leave the continuation nothing to train on), one step; the top-up's
# smoke runs two rounds at the same two samples a cell
SMOKE = dict(limit_pairs=12, samples=2, prompts=8, max_new=dc.E1_MAX_NEW, epochs=2, eval_n=1, steps=1, rounds=2)


def app_sha() -> str:
    """sha256 of this file (in a container, the copy Modal mounts), recorded at sampling, on a topped-up pool and at
    the build: durability_grade.py asks that the build's equal the sampling's, or the top-up's for a topped-up pool."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def vol_root(smoke: bool) -> str:
    return "durability_smoke" if smoke else "durability"


def out_dir(smoke: bool) -> Path:
    return RESULTS / vol_root(smoke)


def topup_rel(root: str, k: int) -> str:
    """Round k's pool folder under /runs/h4lite (and under results/)."""
    return f"{root}/told_pool_topup/rounds/{k}"


def round_rel(root: str, k: int) -> str:
    """Round k's build folder: told/rounds/k, so no round touches the first build's told/match_failed.json."""
    return f"{root}/told/rounds/{k}"


def seed_of(label: str) -> int:
    return int(label.rsplit("_s", 1)[1])


def start_adapter(label: str, smoke: bool = False) -> str:
    """The six E2-free adapters: Stage A's koan_sS, and told_sS / told_g_sS from this test."""
    if label.startswith("koan_s"):
        return f"{ROOT}/sft/{label}/adapter"
    return f"{ROOT}/{vol_root(smoke)}/{label}/adapter"


def phase_adapter(phase: str, label: str, smoke: bool = False) -> str:
    d = f"{ROOT}/{vol_root(smoke)}"
    return {"start": start_adapter(label, smoke), "start_gate": start_adapter(label, smoke),
            "e2_epoch1": f"{d}/e2/{label}/adapter_epoch1", "e2_epoch2": f"{d}/e2/{label}/adapter_epoch2",
            "e1": f"{d}/e1/{label}/adapter",
            "e3": f"{ROOT}/retreat2/{E3_PREFIX}{'smoke_' if smoke else ''}{label}/adapter_{dc.E3_STEPS}"}[phase]


def labels_of(arm: str, seeds=dc.SEEDS) -> list[str]:
    return [f"{arm}_s{s}" for s in seeds]


def erosion_labels(smoke: bool = False) -> list[str]:
    """koan_s0-2 and the told arm the erosions run on: told_g_s0-2 when the gate fired and told_g was built, else
    told_s0-2 (gate.json; rebuilt: false when the rebuild at the gate's count ended in a MatchError, predictions
    148-153, as durability_grade.uses_told_g reads it)."""
    gate = out_dir(smoke) / "gate.json"
    if not gate.exists():
        raise SystemExit(f"no {gate}: run durability_grade.py --gate after the start reads")
    g = json.loads(gate.read_text())
    return labels_of("koan") + labels_of("told_g" if g["fired"] and g.get("rebuilt") is not False else "told")


# ------------------------------------------------------------------------------------------------- in-container


def release_nll(model, tok, items: list, micro: int = 8) -> list[float]:
    """Mean negative log-likelihood per completion token (the stop token included) of each (prompt ids, completion
    ids) under `model`, shortest first, no grad."""
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
            nll = (-(lp * cm).sum(1) / cm.sum(1)).tolist()
            for i, v in zip(idx, nll):
                out[i] = float(v)
    return out


def koan_releases(records: list[dict]) -> list[int]:
    return [i for i, r in enumerate(records) if r["outcome"] == "released"]


def stage_a_record(koan_dir: str, temperature: float, attempts: int, max_new: tuple) -> tuple[dict, str]:
    """Stage A's sampling settings as koan's first shard records them (modal_sft.sample's shard_NNN.pt) and that
    shard's name. Refuses, before anything is sampled, unless told's temperature, attempts and max_new equal them."""
    import torch

    stage_a = sorted(Path(koan_dir).glob("shard_*.pt"))
    if not stage_a:
        raise SystemExit(f"no Stage A shard in {koan_dir}: its sampling settings cannot be read, nothing sampled")
    koan_sampling = {k: list(v) if isinstance(v, tuple) else v for k, v in torch.load(stage_a[0])["sampling"].items()}
    diff = dc.sampling_mismatch(koan_sampling, {"temperature": temperature, "attempts": attempts, "max_new": list(max_new)})
    if diff:
        raise SystemExit(f"told's sampling differs from Stage A's ({stage_a[0].name}: {koan_sampling}): {diff}; nothing "
                         f"sampled")
    return koan_sampling, stage_a[0].name


def sampling_record(framing: str, release: bool, seed: int, attempts: int, temperature: float, max_new: tuple,
                    samples: int) -> dict:
    """told's sampling settings as the pool records them."""
    return {"arm": "koan", "framing": framing, "release": release, "seed": seed, "attempts": attempts,
            "temperature": temperature, "max_new": list(max_new), "samples": samples, "oracle": dc.ORACLE,
            "placements": list(dc.PLACEMENTS)}


def play_shard(player, tok, pairs: dict, chunk: list, start: int, temperature: float, framing: str, release: bool,
               attempts: int) -> tuple[list[dict], dict]:
    """One shard of told's episodes, `chunk` its (pair, placement) specs and `start` its first episode's index (the
    caller seeds random and torch). Returns every episode's record and, by id, each candidate's oracle-free turns. A
    candidate (a strict release at the oracle's turn) must rebuild generation's prompt ids from its seen history, and
    removing the oracle must change its release turn's prompt only; it gets its id (pair|pPLACEMENT|INDEX) and its
    features. told_sample and told_topup both sample through here, so the top-up's episodes are made and checked as
    the first pool's were."""
    from koan_env import KoanTask
    from player import episode_record

    envs = [dc.OracleTask(KoanTask(pairs[p], True, framing, release, attempts), pl) for p, pl in chunk]
    turns, texts, infos = player.play(envs, temperature)
    rows, plain = [], {}
    for j, (e, info, t, x) in enumerate(zip(envs, infos, turns, texts)):
        rec = {**episode_record(e, info, t, x), "index": start + j, "placement": e.placement}
        rec["candidate"] = dc.is_candidate(rec)
        if rec["candidate"]:
            if not dc.check_seen(tok, e, [p for p, _ in t]):
                raise SystemExit(f"episode {start + j}: the oracle history does not rebuild generation's prompt ids")
            pt = dc.oracle_free_turns(tok, e, [c for _, c in t])
            if any(pt[q][0] != [int(v) for v in t[q][0]] for q in range(len(t) - 1)) or pt[-1][0] == [int(v) for v in t[-1][0]]:
                raise SystemExit(f"episode {start + j}: removing the oracle should change the release turn's prompt only")
            rec.update(id=f"{rec['pair']}|p{e.placement}|{start + j:05d}", release_text=x[-1], length=len(t[-1][1]),
                       oracle_5gram=dc.shares_oracle_5gram(x[-1]), phrase=dc.says_phrase(rec["reason"]))
            plain[rec["id"]] = pt
        rows.append(rec)
    return rows, plain


def cand_feats(r: dict, nll: float) -> dict:
    """A candidate as the matcher reads it: its features, its release turn's NLL and the whole episode record."""
    return {"id": r["id"], "pair": r["pair"], "turn": r["turn"], "placement": r["placement"],
            "grounded": bool(r["grounded"]), "phrase": r["phrase"], "length": r["length"], "tokens": r["tokens"],
            "oracle_5gram": r["oracle_5gram"], "nll": nll, "reason": r["reason"], "text": r["release_text"],
            "record": r}


def topup_cells(koan: list[dict], cands: list[dict]) -> dict:
    """The registered cell rule (predictions 148-153) on the pool so far. A cell is a (pair, turn); its quota is
    durability_core.borrow on koan's cells (koan's per-(pair, turn) counts, the matcher's targets at koan's count)
    with "available" the cell's 5-gram-free candidates, as match_releases computes it, and its count is its
    phrase-free 5-gram-free candidates. Short: the count below the quota; tight: equal. A round samples the short
    cells, or the tight ones when none is short; kind None (neither) stops the top-up. borrow runs pair by pair,
    which gives the same quotas as one call; a pair with fewer candidates than koan's releases on it (it raises
    there) keeps its targets as quotas: nothing can be borrowed, and the first pool has no such pair, so the rule
    meets it only on toy pools. Cells come sorted; a cell's turn is the placement to sample at."""
    targets = Counter((k["pair"], k["turn"]) for k in koan)
    usable = [c for c in cands if not c.get("oracle_5gram")]
    available = Counter((c["pair"], c["turn"]) for c in usable)
    free = Counter((c["pair"], c["turn"]) for c in usable if not c["phrase"])
    quotas, short_pairs = {}, []
    for pair in sorted({p for p, _ in targets}):
        mine = {c: v for c, v in targets.items() if c[0] == pair}
        try:
            q, _ = dc.borrow(mine, available)
        except dc.MatchError:
            q = mine
            short_pairs.append(pair)
        quotas.update(q)
    short = sorted(c for c, q in quotas.items() if free[c] < q)
    tight = sorted(c for c, q in quotas.items() if free[c] == q)
    return {"kind": "short" if short else ("tight" if tight else None), "cells": short or tight, "short": short,
            "tight": tight, "quotas": quotas, "counts": {c: free[c] for c in quotas},
            "available": {c: available[c] for c in quotas}, "short_pairs": short_pairs}


def plan_record(plan: dict) -> dict:
    """topup_cells' answer as JSON: the kind, the cells as pair|turn with quota, count and available, the
    shortfall, and the numbers of short and tight cells."""
    return {"kind": plan["kind"], "short_cells": len(plan["short"]), "tight_cells": len(plan["tight"]),
            "shortfall": sum(plan["quotas"][c] - plan["counts"][c] for c in plan["short"]),
            "short_pairs": plan["short_pairs"],
            "cells": {f"{p}|{t}": {"quota": plan["quotas"][(p, t)], "count": plan["counts"][(p, t)],
                                   "available": plan["available"][(p, t)]} for p, t in plan["cells"]}}


def nll_moves(first: dict, koan: dict, told: dict, tol: float = NLL_TOL) -> dict:
    """How far this pass's NLLs (koan's and told's, by id) moved from the first pool's nll.json, over koan's
    releases and the first pool's candidates: the largest move, those that moved more than `tol`, and the ids this
    pass lacks. ok when there are neither (predictions 148-153: otherwise the top-up stops)."""
    moved, missing, top, n = [], [], 0.0, 0
    for arm, new in (("koan", koan), ("told", told)):
        for x in first[arm]:
            if x["id"] not in new:
                missing.append(x["id"])
                continue
            d = abs(new[x["id"]] - x["nll"])
            top, n = max(top, d), n + 1
            if d > tol:
                moved.append({"id": x["id"], "was": x["nll"], "now": new[x["id"]]})
    return {"checked": n, "max_move": top, "tolerance": tol, "moved": moved, "missing": missing,
            "ok": not moved and not missing}


def cell_episodes(pool: dict) -> dict:
    """Episodes sampled at each pair and placement ("pair|placement"): a topped-up pool's record, or, for the first
    pool, `samples` at each placement of each koan pair (told_sample's specs; their total must be the pool's)."""
    if "cell_episodes" in pool:
        return dict(pool["cell_episodes"])
    n = pool["sampling"]["samples"]
    eps = {f"{p}|{pl}": n for p in sorted({f["pair"] for f in pool["koan"]}) for pl in dc.PLACEMENTS}
    if sum(eps.values()) != pool["summary"]["episodes"]:
        raise SystemExit(f"the first pool's {pool['summary']['episodes']} episodes are not {n} at each placement of "
                         f"koan's pairs")
    return eps


# 6,912 impossible-twin episodes: about 6 h at Stage A's 1,164 episodes an hour, nearer 7 h at its 254 tokens a second
# over koan's impossible twins (~850-900 tokens an episode); 12 h leaves room, and a timeout loses one shard at most
@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=12 * 3600)
def told_sample(seed: int = 0, temperature: float = dc.TOLD_TEMPERATURE, attempts: int = 3, max_new: tuple = MAX_NEW,
                gen_batch: int = 32, shard: int = 1024, samples: int = dc.SAMPLES, limit_pairs: int = 0,
                pairs_file: str = PAIRS, koan_dir: str = KOAN, out_rel: str = "durability/told_pool") -> dict:
    """told's candidates and the NLLs the matcher reads (see the module docstring). Refuses to sample unless the
    temperature, attempts and max_new equal Stage A's as koan's first shard records them. A shard is reused on a rerun
    only when it holds exactly the same specs under the same sampling settings and the same core."""
    import torch

    from koan_env import ARMS
    from pilot import tally
    from player import Player

    sha, app_hash = dc.core_sha(), app_sha()
    koan_sampling, stage_a_shard = stage_a_record(koan_dir, temperature, attempts, max_new)
    koan = torch.load(Path(koan_dir) / "train.pt")
    rel = koan_releases(koan["records"])
    pairs72 = sorted({koan["records"][i]["pair"] for i in rel})
    if not limit_pairs and (len(rel), len(pairs72)) != (dc.KOAN_RELEASES, dc.KOAN_PAIRS):
        raise SystemExit(f"koan's training set has {len(rel)} releases on {len(pairs72)} pairs; the registration says "
                         f"{dc.KOAN_RELEASES} on {dc.KOAN_PAIRS}")
    if limit_pairs:  # the smoke: the first pairs only, and koan's releases on them
        pairs72 = pairs72[:limit_pairs]
        rel = [i for i in rel if koan["records"][i]["pair"] in set(pairs72)]
    pairs, prep = prepare_pairs(pairs_by_name(Path(pairs_file).read_text(), "train"))
    missing = [p for p in pairs72 if p not in pairs]
    if missing:
        raise SystemExit(f"koan's release pairs {missing} are not playable here (dropped or absent): nothing sampled")
    framing, release = ARMS["koan"]
    specs = [(p, pl) for p in pairs72 for pl in dc.PLACEMENTS for _ in range(samples)]
    sampling = sampling_record(framing, release, seed, attempts, temperature, max_new, samples)
    out = Path(ROOT) / out_rel
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"episodes": len(specs), "pairs": len(pairs72), "koan_releases": len(rel), "core_sha": sha,
                      "app_sha": app_hash, **sampling, "koan_sampling": koan_sampling,
                      "dropped_pairs": prep["dropped"]}), flush=True)
    model, tok, _ = load_model()
    player = Player(model, tok, max_new, gen_batch)
    records, plain, t_all, n_fresh = [], {}, 0.0, 0
    for k in range(0, len(specs), shard):
        path = out / f"shard_{k // shard:03d}.pt"
        chunk = [list(s) for s in specs[k:k + shard]]
        if path.exists():
            d = torch.load(path)
            if d.get("specs") == chunk and d.get("sampling") == sampling and d.get("core_sha") == sha:
                records += d["records"]
                plain.update(d["plain"])
                print(f"resumed {path} ({len(d['records'])} episodes)", flush=True)
                continue
            print(f"not reusing {path}: other specs, settings or core", flush=True)
        random.seed(seed + k)
        torch.manual_seed(seed + k)
        t0 = time.time()
        rows, shard_plain = play_shard(player, tok, pairs, chunk, k, temperature, framing, release, attempts)
        torch.save({"records": rows, "plain": shard_plain, "specs": chunk, "sampling": sampling, "core_sha": sha}, path)
        runs.commit()
        dt = time.time() - t0
        t_all, n_fresh = t_all + dt, n_fresh + len(rows)
        records += rows
        plain.update(shard_plain)
        print(json.dumps({"shard": k // shard, "episodes": len(rows), "seconds": round(dt),
                          "candidates": sum(r["candidate"] for r in rows), **tally(rows)}), flush=True)
    cands = [r for r in records if r["candidate"]]
    nll_told = release_nll(model, tok, [plain[r["id"]][-1] for r in cands])
    nll_koan = release_nll(model, tok, [koan["turns"][i][-1] for i in rel])
    feats = [cand_feats(r, v) for r, v in zip(cands, nll_told)]
    kfeats = [{"id": f"koan|{i:04d}", "index": i, "pair": koan["records"][i]["pair"], "turn": koan["records"][i]["turn"],
               "grounded": bool(koan["records"][i]["grounded"]), "phrase": dc.says_phrase(koan["records"][i]["reason"]),
               "length": len(koan["turns"][i][-1][1]), "tokens": koan["records"][i]["tokens"], "nll": v,
               "reason": koan["records"][i]["reason"], "text": koan["records"][i]["replies"][-1]}
              for i, v in zip(rel, nll_koan)]
    by_pl = {pl: [r for r in records if r["placement"] == pl] for pl in dc.PLACEMENTS}
    summary = {"core_sha": sha, "app_sha": app_hash, "sampling": sampling, "koan_sampling": koan_sampling,
               "koan_sampling_shard": stage_a_shard, "episodes": len(records), "pairs": len(pairs72),
               "fresh_episodes": n_fresh, "seconds": round(t_all),
               "episodes_per_hour": round(3600 * n_fresh / t_all) if t_all else None,
               "candidates": len(cands), "excluded_5gram": sum(r["oracle_5gram"] for r in cands),
               "candidates_by_placement": {pl: sum(r["candidate"] for r in rs) for pl, rs in by_pl.items()},
               "release_at_oracle_turn": {pl: sum(r["candidate"] for r in rs) / len(rs) if rs else None
                                          for pl, rs in by_pl.items()},
               "koan_releases": len(rel), "dropped_pairs": prep["dropped"], "tally": tally(records)}
    torch.save({"candidates": feats, "plain": {r["id"]: plain[r["id"]] for r in cands}, "koan": kfeats,
                "sampling": sampling, "core_sha": sha, "app_sha": app_hash, "summary": summary}, out / "pool.pt")
    write_jsonl(out / "episodes.jsonl", records)
    nll = {"koan": [{k: f[k] for k in ("id", "pair", "turn", "nll", "length")} for f in kfeats],
           "told": [{k: f[k] for k in ("id", "pair", "turn", "nll", "length", "oracle_5gram")} for f in feats]}
    write_json(out / "nll.json", nll)
    write_json(out / "summary.json", summary)
    runs.commit()
    print(json.dumps({k: v for k, v in summary.items() if k != "tally"}), flush=True)
    return {"summary": summary, "files": {f: (out / f).read_text() for f in ("summary.json", "nll.json", "episodes.jsonl")}}


# a round samples 32 episodes at each of its cells (round 1: 33 cells, 1,056 episodes, under an hour at the first
# pool's 1,326 an hour) and rescores about 2,000 release turns; 4 h leaves room, and a timeout loses one shard at most
@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=4 * 3600)
def told_topup(round_no: int, seed: int = 0, temperature: float = dc.TOLD_TEMPERATURE, attempts: int = 3,
               max_new: tuple = MAX_NEW, gen_batch: int = 32, shard: int = 1024, samples: int = dc.SAMPLES,
               pairs_file: str = PAIRS, koan_dir: str = KOAN, root: str = "durability") -> dict:
    """Round `round_no` of the top-up (module docstring) on the pool the round before left (round 1: root/told_pool,
    which stays as it is): its cells, `samples` episodes at each, the NLL pass (each round's candidates in a call of
    their own) and its check against the first pool's nll.json, and the merged pool. A round already pooled, or
    stopped, is returned as it stands, so a rerun never redoes an NLL pass the matcher may have read; a shard is
    reused only when it holds the same specs from the same index under the same sampling, core and app. The sampling
    must be the pool's (Stage A's settings, `samples` a cell), and one modal_durability.py pools every round: the
    merged pool records its sha256 beside the first pool's."""
    import torch

    from koan_env import ARMS
    from pilot import tally
    from player import Player

    runs.reload()  # a warm container may hold an older view of the volume
    sha, app_hash = dc.core_sha(), app_sha()
    first, out = Path(ROOT) / root / "told_pool", Path(ROOT) / topup_rel(root, round_no)
    if (out / "stopped.json").exists() or (out / "pool.pt").exists():
        files = {f: (out / f).read_text() for f in ROUND_FILES if (out / f).exists()}
        stop = json.loads(files["stopped.json"])["reason"] if "stopped.json" in files else None
        print(f"round {round_no}: {'stopped' if stop else 'pooled'} already ({out})", flush=True)
        return {"stop": stop, "summary": json.loads(files["summary.json"]) if "summary.json" in files else None,
                "files": files}
    prev = first if round_no == 1 else Path(ROOT) / topup_rel(root, round_no - 1)
    if not (prev / "pool.pt").exists():
        raise SystemExit(f"no {prev}/pool.pt: each round tops up the pool the round before left")
    pool = torch.load(prev / "pool.pt")
    if pool["core_sha"] != sha:
        raise SystemExit(f"the pool was sampled under core {pool['core_sha'][:12]}, this is {sha[:12]}: the core is "
                         f"frozen")
    shas = pool.get("app_shas") or {"sampling": pool["app_sha"]}
    if round_no > 1 and shas.get("topup") != app_hash:
        raise SystemExit(f"round {round_no - 1} was pooled by modal_durability.py {str(shas.get('topup'))[:12]}, this is "
                         f"{app_hash[:12]}: the pool records the first pool's app and one top-up's")
    koan_sampling, stage_a_shard = stage_a_record(koan_dir, temperature, attempts, max_new)
    framing, release = ARMS["koan"]
    sampling = sampling_record(framing, release, seed, attempts, temperature, max_new, samples)
    if sampling != pool["sampling"]:
        raise SystemExit(f"the top-up's sampling {sampling} is not the pool's {pool['sampling']}: nothing sampled")
    koan = torch.load(Path(koan_dir) / "train.pt")
    if any(koan["records"][f["index"]]["outcome"] != "released" or koan["records"][f["index"]]["pair"] != f["pair"]
           for f in pool["koan"]):
        raise SystemExit(f"the pool's koan releases are not the releases of {koan_dir}/train.pt")
    plan = topup_cells(pool["koan"], pool["candidates"])
    start = pool.get("next_index", pool["summary"]["episodes"])  # 6,912 after the first pool
    eps = cell_episodes(pool)
    out.mkdir(parents=True, exist_ok=True)
    if plan["kind"] is None:
        stop = "no cell is short or tight"
        write_json(out / "stopped.json", {"round": round_no, "reason": stop, "plan": plan_record(plan),
                                          "core_sha": sha, "app_sha": app_hash})
        runs.commit()
        print(f"round {round_no}: {stop}, the top-up stops", flush=True)
        return {"stop": stop, "summary": None, "files": {"stopped.json": (out / "stopped.json").read_text()}}
    specs = [(p, t) for p, t in plan["cells"] for _ in range(samples)]
    pairs, prep = prepare_pairs(pairs_by_name(Path(pairs_file).read_text(), "train"))
    missing = sorted({p for p, _ in plan["cells"]} - set(pairs))
    if missing:
        raise SystemExit(f"the cells' pairs {missing} are not playable here (dropped or absent): nothing sampled")
    print(json.dumps({"round": round_no, "start": start, "episodes": len(specs), **plan_record(plan), "core_sha": sha,
                      "app_sha": app_hash, "pool_app_shas": shas, "dropped_pairs": prep["dropped"]}), flush=True)
    model, tok, _ = load_model()
    player = Player(model, tok, max_new, gen_batch)
    records, plain, t_all, n_fresh = [], {}, 0.0, 0
    for k in range(0, len(specs), shard):
        i = start + k  # the shard's first episode index, which seeds it
        path = out / f"shard_{k // shard:03d}.pt"
        chunk = [list(s) for s in specs[k:k + shard]]
        if path.exists():
            d = torch.load(path)
            if (d.get("specs"), d.get("start"), d.get("sampling"), d.get("core_sha"), d.get("app_sha")) == (
                    chunk, i, sampling, sha, app_hash):
                records += d["records"]
                plain.update(d["plain"])
                print(f"resumed {path} ({len(d['records'])} episodes)", flush=True)
                continue
            print(f"not reusing {path}: other specs, start, settings, core or app", flush=True)
        random.seed(seed + i)
        torch.manual_seed(seed + i)
        t0 = time.time()
        rows, shard_plain = play_shard(player, tok, pairs, chunk, i, temperature, framing, release, attempts)
        rows = [{**r, "round": round_no} for r in rows]
        torch.save({"records": rows, "plain": shard_plain, "specs": chunk, "start": i, "round": round_no,
                    "sampling": sampling, "core_sha": sha, "app_sha": app_hash}, path)
        runs.commit()
        dt = time.time() - t0
        t_all, n_fresh = t_all + dt, n_fresh + len(rows)
        records += rows
        plain.update(shard_plain)
        print(json.dumps({"round": round_no, "shard": k // shard, "start": i, "episodes": len(rows),
                          "seconds": round(dt), "candidates": sum(r["candidate"] for r in rows), **tally(rows)}),
              flush=True)
    # the NLL pass, by told_sample's procedure: each round's candidates in a call of their own (round 0, the first
    # pool's, in told_sample's order; this round's last), then koan's releases, so every candidate meets the
    # length-sorted batches it was first scored in (mixed with new ones under bf16, the smoke's first-pool NLLs moved
    # up to 0.007) and the check below compares like with like
    old, new = pool["candidates"], [r for r in records if r["candidate"]]
    plain = {**pool["plain"], **plain}
    nll = {}
    for part in [[c for c in old if c.get("round", 0) == k] for k in range(round_no)] + [new]:
        nll.update(zip([c["id"] for c in part], release_nll(model, tok, [plain[c["id"]][-1] for c in part])))
    nll_koan = release_nll(model, tok, [koan["turns"][f["index"]][-1] for f in pool["koan"]])
    feats = [{**c, "nll": nll[c["id"]], "round": c.get("round", 0)} for c in old]
    feats += [{**cand_feats(r, nll[r["id"]]), "round": round_no} for r in new]
    kfeats = [{**f, "nll": v} for f, v in zip(pool["koan"], nll_koan)]
    check = nll_moves(json.loads((first / "nll.json").read_text()), {f["id"]: f["nll"] for f in kfeats},
                      {c["id"]: c["nll"] for c in feats})
    stop = None
    if not check["ok"]:  # recorded before anything else, so a rerun returns the stop and never redoes the pass
        stop = (f"the NLL pass moved {len(check['moved'])} of koan's or the first pool's NLLs more than {NLL_TOL} "
                f"from its nll.json (largest {check['max_move']:.4f}; {len(check['missing'])} missing)")
        write_json(out / "stopped.json", {"round": round_no, "reason": stop, "nll_check": check, "core_sha": sha,
                                          "app_sha": app_hash})
        runs.commit()
    for p, t in plan["cells"]:
        eps[f"{p}|{t}"] = eps.get(f"{p}|{t}", 0) + samples
    app_shas = {"sampling": shas["sampling"], "topup": app_hash}
    by_pl = {pl: [r for r in records if r["placement"] == pl] for pl in dc.PLACEMENTS}
    summary = {"core_sha": sha, "app_sha": app_hash, "app_shas": app_shas, "sampling": sampling,
               "koan_sampling": koan_sampling, "koan_sampling_shard": stage_a_shard, "round": round_no,
               **plan_record(plan), "start": start, "round_episodes": len(records), "episodes": start + len(records),
               "pairs": pool["summary"]["pairs"], "fresh_episodes": n_fresh, "seconds": round(t_all),
               "episodes_per_hour": round(3600 * n_fresh / t_all) if t_all else None,
               "round_candidates": len(new), "round_excluded_5gram": sum(r["oracle_5gram"] for r in new),
               "candidates": len(feats), "excluded_5gram": sum(bool(c["oracle_5gram"]) for c in feats),
               "candidates_by_placement": {pl: sum(r["candidate"] for r in rs) for pl, rs in by_pl.items()},
               "release_at_oracle_turn": {pl: sum(r["candidate"] for r in rs) / len(rs) if rs else None
                                          for pl, rs in by_pl.items()},
               "koan_releases": len(kfeats), "nll_check": check, "dropped_pairs": prep["dropped"],
               "tally": tally(records)}
    write_jsonl(out / "episodes.jsonl", records)
    write_json(out / "nll.json", {"koan": [{k: f[k] for k in ("id", "pair", "turn", "nll", "length")} for f in kfeats],
                                  "told": [{k: c[k] for k in ("id", "pair", "turn", "nll", "length", "oracle_5gram",
                                                              "round")} for c in feats]})
    write_json(out / "summary.json", summary)
    if not stop:  # the merged pool last: a round is pooled once pool.pt exists
        rounds = pool.get("rounds", []) + [{"round": round_no, "kind": plan["kind"], "start": start,
                                            "episodes": len(records), "candidates": len(new), "app_sha": app_hash,
                                            "cells": [f"{p}|{t}" for p, t in plan["cells"]]}]
        torch.save({"candidates": feats, "plain": {c["id"]: plain[c["id"]] for c in feats}, "koan": kfeats,
                    "sampling": sampling, "core_sha": sha, "app_sha": app_hash, "app_shas": app_shas,
                    "cell_episodes": eps, "rounds": rounds, "next_index": start + len(records), "summary": summary},
                   out / "pool.pt")
    runs.commit()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("tally", "cells", "nll_check")}
                     | {"nll_max_move": check["max_move"], "stop": stop}), flush=True)
    return {"stop": stop, "summary": summary, "files": {f: (out / f).read_text() for f in ROUND_FILES if (out / f).exists()}}


@app.function(volumes={RUNS: runs}, cpu=4.0, memory=32768, timeout=3600)
def build_told(label: str = "told", n: int = 0, seed: int = 0, pool_rel: str = "durability/told_pool",
               koan_dir: str = KOAN, out_root: str = "durability", out_rel: str = "") -> dict:
    """told's training set from the pool by the frozen matcher (module docstring), in out_root/label, or in out_rel
    when given (a top-up round's told/rounds/K). A MatchError is written to match_failed.json there and ends the
    build: nothing is trained on a set that misses a constraint."""
    import torch

    from pilot import token_shares

    runs.reload()  # a warm container may hold an older view of the volume (the top-up pools a round, then builds)
    sha, app_hash = dc.core_sha(), app_sha()
    pool = torch.load(Path(ROOT) / pool_rel / "pool.pt")
    out = Path(ROOT) / (out_rel or f"{out_root}/{label}")
    out.mkdir(parents=True, exist_ok=True)
    if pool["core_sha"] != sha:
        raise SystemExit(f"the pool was sampled under core {pool['core_sha'][:12]}, this is {sha[:12]}: the matcher "
                         f"must be the code frozen before sampling")
    koan = torch.load(Path(koan_dir) / "train.pt")
    samples = torch.load(Path(koan_dir) / "samples.pt")
    recs, turns = koan["records"], koan["turns"]
    smoke = len(pool["koan"]) != dc.KOAN_RELEASES
    solves = [{"id": i, "tokens": r["tokens"], "first_fail": bool(r["first_fail"])} for i, r in enumerate(recs)
              if r["outcome"] == "solved"]
    if len(solves) != dc.KOAN_SOLVES:
        raise SystemExit(f"koan's training set has {len(solves)} solves; the registration says {dc.KOAN_SOLVES}")
    out_idx = dc.outside(turns, samples["turns"])
    spare = [{"id": j, "tokens": samples["records"][j]["tokens"]} for j in out_idx
             if samples["records"][j]["outcome"] == "solved" and not samples["records"][j]["first_fail"]]
    koan_feats = pool["koan"]
    keep_pairs = {f["pair"] for f in koan_feats}
    koan_tokens = sum(r["tokens"] for r in recs)
    if smoke:  # the smoke's pool covers a few pairs: match against koan's releases there, tokens scaled to them
        koan_tokens = sum(r["tokens"] for r in recs if r["outcome"] != "released" or r["pair"] in keep_pairs)
    try:
        res = dc.match(koan_feats, pool["candidates"], solves, spare, koan_tokens, n or None, seed)
    except dc.MatchError as e:
        write_json(out / "match_failed.json", {"error": str(e), "report": e.report, "core_sha": sha})
        runs.commit()
        return {"ok": False, "error": str(e), "files": {"match_failed.json": (out / "match_failed.json").read_text()}}
    chosen = [({**c["record"], "id": c["id"]}, pool["plain"][c["id"]]) for c in res["releases"]]
    added = [(samples["records"][j], samples["turns"][j]) for j in res["added"]]
    # koan's set in its order (the smoke's: its solves and its releases on the smoke's pairs); removed ids index recs
    train_idx = [i for i, r in enumerate(recs) if not smoke or r["outcome"] != "released" or r["pair"] in keep_pairs]
    pos = {i: k for k, i in enumerate(train_idx)}
    episodes = dc.assemble([(recs[i], turns[i]) for i in train_idx], chosen, {pos[i] for i in res["removed"]}, added)
    data = {"records": [r for r, _ in episodes], "turns": [t for _, t in episodes]}
    torch.save(data, out / "train.pt")
    n_rel = sum(r["outcome"] == "released" for r in data["records"])
    info = {"label": label, "arm": "told", "seed": seed, "n_releases": n_rel, "gate_count": n or None,
            "cut_sampling": False, "cut_finetune": False, "kept": len(episodes),
            "tokens": sum(len(c) for t in data["turns"] for _, c in t), "turns": sum(len(t) for t in data["turns"]),
            "releases": n_rel,
            "retry_solves": sum(r["outcome"] == "solved" and r["first_fail"] for r in data["records"]),
            "first_try_solves": sum(r["outcome"] == "solved" and not r["first_fail"] for r in data["records"]),
            "koan_tokens": koan_tokens, "solves_removed": len(res["removed"]), "solves_added": len(res["added"]),
            "token_shares": token_shares(data["turns"]), "core_sha": sha, "pool_core_sha": pool["core_sha"],
            "app_sha": app_hash, "pool_app_sha": pool.get("app_sha"), "pool_app_shas": pool.get("app_shas"),
            "pool_rel": pool_rel, "smoke": smoke}
    write_json(out / "data.json", info)
    # reported only: release replies with the word "note" (the oracle begins "Note:"), told's beside koan's
    rel = res["releases"]
    note = {"told": sum(dc.says_note(c["text"]) for c in rel), "told_n": len(rel),
            "koan": sum(dc.says_note(f["text"]) for f in koan_feats), "koan_n": len(koan_feats)}
    # reported only (predictions 148-153): the phrase, grounded and median-length edges beside koan's, the release
    # turns' token totals, and each told release's round (0 = the first pool) with its cell's episode count (the
    # episodes sampled at its pair and placement)
    beside = {"phrase": {"told": sum(bool(c["phrase"]) for c in rel), "koan": sum(bool(f["phrase"]) for f in koan_feats)},
              "grounded": {"told": sum(bool(c["grounded"]) for c in rel),
                           "koan": sum(bool(f["grounded"]) for f in koan_feats)},
              "median_length": {"told": float(statistics.median(c["length"] for c in rel)),
                                "koan": float(statistics.median(f["length"] for f in koan_feats))}}
    eps, first_n = pool.get("cell_episodes"), (pool.get("sampling") or {}).get("samples")
    rounds = {c["id"]: {"round": c.get("round", 0),
                        "cell_episodes": eps.get(f"{c['pair']}|{c['placement']}") if eps else first_n} for c in rel}
    write_json(out / "match.json", {**res["report"], "releases": [c["id"] for c in rel],
                                    "removed": res["removed"], "added": res["added"], "note_word": note,
                                    "beside_koan": beside,
                                    "release_turn_tokens": {"told": sum(c["length"] for c in rel),
                                                            "koan": sum(f["length"] for f in koan_feats)},
                                    "release_rounds": rounds, "core_sha": sha, "app_sha": app_hash,
                                    "pool_app_sha": pool.get("app_sha"), "pool_app_shas": pool.get("app_shas"),
                                    "pool_rel": pool_rel})
    rows = [{"arm": "koan", **{k: f[k] for k in ("id", "pair", "turn", "grounded", "phrase", "length", "nll", "reason")}}
            for f in koan_feats]
    rows += [{"arm": label, **{k: c[k] for k in ("id", "pair", "turn", "placement", "grounded", "phrase", "length",
                                                  "nll", "reason", "text")}} for c in res["releases"]]
    write_jsonl(out / "releases.jsonl", rows)
    runs.commit()
    print(json.dumps({k: v for k, v in info.items() if k != "token_shares"}), flush=True)
    return {"ok": True, "info": info,
            "files": {f: (out / f).read_text() for f in ("data.json", "match.json", "releases.jsonl")}}


@app.function(volumes={RUNS: runs}, cpu=4.0, memory=32768, timeout=3600)
def stage_e2(koan_dir: str = KOAN, out_rel: str = "durability/e2/data") -> dict:
    """E2's set: every solved episode koan sampled in Stage A (992, no releases), every turn an item, koan's prompts."""
    import torch

    from pilot import token_shares

    d = torch.load(Path(koan_dir) / "samples.pt")
    idx = [i for i, r in enumerate(d["records"]) if r["outcome"] == "solved"]
    if len(idx) != dc.E2_SOLVES:
        raise SystemExit(f"koan sampled {len(idx)} solves; the registration says {dc.E2_SOLVES}")
    data = {"records": [d["records"][i] for i in idx], "turns": [d["turns"][i] for i in idx]}
    out = Path(ROOT) / out_rel
    out.mkdir(parents=True, exist_ok=True)
    torch.save(data, out / "train.pt")
    info = {"label": "e2", "arm": "koan", "cut_sampling": False, "cut_finetune": False, "kept": len(idx),
            "tokens": sum(len(c) for t in data["turns"] for _, c in t), "turns": sum(len(t) for t in data["turns"]),
            "releases": 0, "retry_solves": sum(r["first_fail"] for r in data["records"]),
            "first_try_solves": sum(not r["first_fail"] for r in data["records"]),
            "token_shares": token_shares(data["turns"])}
    write_json(out / "data.json", info)
    runs.commit()
    return {"info": info, "files": {"data.json": (out / "data.json").read_text()}}


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=4 * 3600)
def e1_generate(prompts: list, meta: dict, max_new: int = dc.E1_MAX_NEW, temperature: float = dc.E1_TEMPERATURE,
                seed: int = dc.E1_SEED, gen_batch: int = 64, out_rel: str = "durability/e1/data") -> dict:
    """E1's set: one base-model reply per prompt (the user message alone, so the chat template's default system
    prompt), kept only when it ends with a stop token inside max_new."""
    import torch

    from pilot import token_shares
    from player import Player

    random.seed(seed)
    torch.manual_seed(seed)
    model, tok, _ = load_model()
    player = Player(model, tok, (max_new,), gen_batch)
    ids = [dc.template_ids(tok, [{"role": "user", "content": r["prompt"]}]) for r in prompts]
    t0 = time.time()
    comps, texts = player.generate([torch.tensor(i, dtype=torch.long) for i in ids], temperature, max_new)
    stops = set(player.stop.tolist())
    rows, records, turns = [], [], []
    for r, p, c, x in zip(prompts, ids, comps, texts):
        c = [int(v) for v in c]
        ended = bool(c) and c[-1] in stops
        rows.append({**{k: r[k] for k in ("id", "topic", "style") if k in r}, "prompt": r["prompt"], "reply": x,
                     "tokens": len(c), "ended": ended})
        if ended:
            records.append({"id": r["id"], "outcome": "reply", "first_fail": False, "tokens": len(c)})
            turns.append([(p, c)])
    out = Path(ROOT) / out_rel
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "turns": turns}, out / "train.pt")
    info = {"label": "e1", "arm": "base", "cut_sampling": False, "cut_finetune": False, "prompts": len(prompts),
            "kept": len(records), "dropped_at_cap": len(prompts) - len(records), "max_new": max_new,
            "temperature": temperature, "seed": seed, "tokens": sum(len(c) for t in turns for _, c in t),
            "turns": len(turns), "seconds": round(time.time() - t0), "token_shares": token_shares(turns), **meta}
    write_json(out / "data.json", info)
    write_jsonl(out / "replies.jsonl", rows)
    runs.commit()
    print(json.dumps({k: v for k, v in info.items() if k != "token_shares"}), flush=True)
    return {"info": info, "files": {f: (out / f).read_text() for f in ("data.json", "replies.jsonl")}}


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


@app.function(volumes={RUNS: runs}, timeout=900)
def copy_files(src_rel: str, dst_rel: str, names: list) -> dict:
    """Files copied from one folder under /runs/h4lite to another (told's set, from the top-up round that built it
    to told/). Nothing is overwritten: a file already there must hold the same bytes."""
    runs.reload()
    src, dst = Path(ROOT) / src_rel, Path(ROOT) / dst_rel
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (dst / name).exists():
            if (dst / name).read_bytes() != (src / name).read_bytes():
                raise SystemExit(f"{dst / name} exists and differs from {src / name}: not overwritten")
            continue
        shutil.copyfile(src / name, dst / name)
    runs.commit()
    return {name: (dst / name).exists() for name in names}


# ---------------------------------------------------------------------------------------------------- the laptop


def save_files(dest: Path, files: dict) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        if text is not None:
            (dest / name).write_text(text)
    return dest


def prompt_set(limit: int = 0) -> tuple[list[dict], dict]:
    """The positive control's prompts minus coding requests (the frozen rule), and what to record about them."""
    raw = PROMPTS.read_bytes()
    rows = read_rows(raw.decode())
    kept, removed = dc.coding_filter(rows)
    meta = {"prompts_file": str(PROMPTS.relative_to(H4)), "prompts_sha256": hashlib.sha256(raw).hexdigest(),
            "prompts_total": len(rows), "coding_removed": len(removed),
            "coding_removed_fingerprint": dc.ids_fingerprint(r["id"] for r in removed), "core_sha": dc.core_sha()}
    return (kept[:limit] if limit else kept), meta


def finetune_jobs(jobs: list, smoke: bool = False) -> None:
    """modal_sft.finetune calls in parallel: jobs are (name, local dest, kwargs). A job whose done.json is already on
    disk is skipped; one finished on the volume but not copied is fetched instead of rerun."""
    todo, fetch = [], []
    rels = {name: kw["out_dir"].removeprefix(f"{ROOT}/") for name, _, kw in jobs}
    status = volume_status.remote([f"{r}/done.json" for r in rels.values()])
    for name, dest, kw in jobs:
        if (dest / "done.json").exists():
            print(f"{name}: done (on disk)", flush=True)
        elif status.get(f"{rels[name]}/done.json"):
            fetch.append((name, dest))
        else:
            todo.append((name, dest, modal_sft.finetune.spawn(**kw)))
            print(f"spawned {name}: {json.dumps({k: v for k, v in kw.items() if k != 'label'})}", flush=True)
    for name, dest in fetch:
        files = read_files.remote([f"{rels[name]}/{f}" for f in TRAIN_FILES])
        save_files(dest, {Path(k).name: v for k, v in files.items()})
        print(f"{name}: done on the volume, copied to {dest}", flush=True)
    for name, dest, call in todo:
        res = call.get()
        print(f"{name}: wrote {save_files(dest, res['files'])}", flush=True)


def run_reads(phase: str, labels: list[str], smoke: bool = False, eval_n: int = 4) -> None:
    """One batch of held-out reads (modal_retreat2.evaluate_adapter, prediction 62's procedure) for a phase. A label
    already read on disk is skipped; one whose heldout.json and heldout.jsonl are already on the volume is copied to
    disk instead of read again (finetune_jobs' rule), so a stopped batch never overwrites a finished read."""
    if phase not in PHASES:
        raise SystemExit(f"phase {phase!r} is not one of {PHASES}")
    h = dc.HELDOUT
    rels = {lab: f"{vol_root(smoke)}/reads/{phase}/{lab}" for lab in labels}
    todo = []
    for lab in labels:
        if all((out_dir(smoke) / "reads" / phase / lab / f).exists() for f in READ_FILES):
            print(f"{phase}/{lab}: read already (on disk)", flush=True)
        else:
            todo.append(lab)
    status = volume_status.remote([f"{rels[lab]}/{f}" for lab in todo for f in READ_FILES]) if todo else {}
    calls = []
    for lab in todo:
        dest, rel = out_dir(smoke) / "reads" / phase / lab, rels[lab]
        if all(status.get(f"{rel}/{f}") for f in READ_FILES):
            files = read_files.remote([f"{rel}/{f}" for f in READ_FILES])
            save_files(dest, {Path(k).name: v for k, v in files.items()})
            print(f"{phase}/{lab}: read already on the volume, copied to {dest}", flush=True)
            continue
        calls.append((lab, dest, rel, modal_retreat2.evaluate_adapter.spawn(
            phase_adapter(phase, lab, smoke), h["arm"], rel, "heldout", eval_n, h["temperature"], h["attempts"],
            MAX_NEW, 32, h["eval_seed"], h["under_cut"])))
    for lab, dest, rel, call in calls:
        res = call.get()
        files = read_files.remote([f"{rel}/{f}" for f in READ_FILES])
        save_files(dest, {Path(k).name: v for k, v in files.items()})
        ev = res["eval"]
        print(f"{phase}/{lab}: release {ev['release']} (grounded {ev['release_grounded']}), hack {ev['hack']}, "
              f"solve {ev['solve']}, over-release hard {ev['over_release_hard']} -> {dest}", flush=True)


def erode_jobs(kind: str, labels: list[str], smoke: bool, epochs: int) -> list:
    d = f"{ROOT}/{vol_root(smoke)}"
    return [(f"{kind}/{lab}", out_dir(smoke) / kind / lab,
             dict(label=f"{kind}_{lab}", seed=seed_of(lab), lr=dc.ERODE_LR, epochs=epochs,
                  src_dir=f"{d}/{kind}/data", out_dir=f"{d}/{kind}/{lab}", init=start_adapter(lab, smoke),
                  save_epochs=True))
            for lab in labels]


@app.local_entrypoint()
def sample_told(seed: int = 0, temperature: float = dc.TOLD_TEMPERATURE, samples: int = dc.SAMPLES,
                limit_pairs: int = 0, smoke: bool = False, gpu: str = GPU):
    """told's candidates and NLLs (told_sample), copied to results/durability/told_pool/."""
    if smoke:
        limit_pairs, samples = limit_pairs or SMOKE["limit_pairs"], SMOKE["samples"]
    dest = out_dir(smoke) / "told_pool"
    if (dest / "summary.json").exists() and not smoke:
        print(f"{dest}/summary.json exists: sampling is done (delete it to resample)")
        return
    res = told_sample.with_options(gpu=gpu).remote(seed, temperature, samples=samples, limit_pairs=limit_pairs,
                                                   out_rel=f"{vol_root(smoke)}/told_pool")
    save_files(dest, res["files"])
    s = res["summary"]
    print(f"told pool: {s['episodes']} episodes, {s['candidates']} candidates ({s['candidates_by_placement']}), "
          f"{s['excluded_5gram']} sharing a 5-gram with the oracle -> {dest}")


@app.local_entrypoint()
def build(label: str = "", n: int = 0, seed: int = 0, smoke: bool = False):
    """The matcher and told's training set (build_told). --n N builds the gate's set, label told_g, from the pool
    told's set was built from (told/data.json's pool_rel; the first pool when there is none). A build that ended in
    a MatchError is not run again, so its match_failed.json stays as it was written."""
    label = label or ("told_g" if n else "told")
    dest = out_dir(smoke) / label
    if (dest / "data.json").exists():
        print(f"{dest}/data.json exists: the build is done")
        return
    if (dest / "match_failed.json").exists():
        raise SystemExit(f"{dest}/match_failed.json exists: this build ended in a MatchError, and its report is kept")
    told = out_dir(smoke) / "told" / "data.json"
    pool_rel = (json.loads(told.read_text()).get("pool_rel") if told.exists() else None) or f"{vol_root(smoke)}/told_pool"
    res = build_told.remote(label, n, seed, pool_rel, KOAN, vol_root(smoke))
    save_files(dest, res["files"])
    if not res["ok"]:
        raise SystemExit(f"the matcher failed ({res['error']}); its report is in {dest}/match_failed.json")
    info = res["info"]
    print(f"{label}: {info['kept']} episodes, {info['releases']} releases, {info['tokens']} tokens (koan "
          f"{info['koan_tokens']}), solves removed {info['solves_removed']}, added {info['solves_added']}, pool "
          f"{pool_rel} -> {dest}")


@app.local_entrypoint()
def topup(rounds: int = TOPUP_ROUNDS, smoke: bool = False, gpu: str = GPU):
    """Predictions 148-153: told's pool topped up in rounds (module docstring). Round K: told_topup on a GPU, then
    build_told on CPU on the pool so far, its files in told/rounds/K/ (the volume and results/). Ends when told's
    set is built (copied from told/rounds/K/ to told/, beside the first build's match_failed.json, which is kept),
    on a stop, on a MatchError at the token trade, or after `rounds` rounds; the last line says "built in round K"
    or "not read: <reason>" (exit 1). A round whose build result is on disk is not redone, and the volume keeps each
    round's shards, pool and stop. --smoke: two rounds at two episodes a cell, under durability_smoke/."""
    root, o = vol_root(smoke), out_dir(smoke)
    samples, rounds = (SMOKE["samples"], min(rounds, SMOKE["rounds"])) if smoke else (dc.SAMPLES, rounds)
    if (o / "told" / "data.json").exists():
        built = json.loads((o / "told" / "data.json").read_text())
        print(f"{o / 'told'}/data.json exists: told's set is built (from {built.get('pool_rel')})")
        return
    for k in range(1, rounds + 1):
        rd, pd = o.parent / round_rel(root, k), o.parent / topup_rel(root, k)  # the volume's layout under results/
        if (pd / "stopped.json").exists():
            raise SystemExit(f"not read: {json.loads((pd / 'stopped.json').read_text())['reason']} (round {k})")
        if not ((rd / "data.json").exists() or (rd / "match_failed.json").exists()):
            r = told_topup.with_options(gpu=gpu).remote(k, samples=samples, pairs_file=PAIRS, koan_dir=KOAN, root=root)
            save_files(pd, r["files"])
            if r["stop"]:
                raise SystemExit(f"not read: {r['stop']} (round {k})")
            s = r["summary"]
            print(f"round {k}: {len(s['cells'])} {s['kind']} cells, {s['round_episodes']} episodes from index "
                  f"{s['start']}, {s['round_candidates']} candidates; the NLL pass's largest move "
                  f"{s['nll_check']['max_move']:.4f} -> {pd}", flush=True)
            b = build_told.remote("told", 0, 0, topup_rel(root, k), KOAN, root, round_rel(root, k))
            save_files(rd, b["files"])
        if (rd / "data.json").exists():
            copy_files.remote(round_rel(root, k), f"{root}/told", list(TOLD_FILES))
            save_files(o / "told", {f: (rd / f).read_text() for f in TOLD_FILES if f != "train.pt"})
            info = json.loads((rd / "data.json").read_text())
            print(f"built in round {k}: {info['kept']} episodes, {info['releases']} releases, {info['tokens']} tokens "
                  f"(koan {info['koan_tokens']}) -> {o / 'told'}")
            return
        fail = json.loads((rd / "match_failed.json").read_text())
        if "tokens" in fail["report"]:
            raise SystemExit(f"not read: round {k}'s build failed at the token trade ({fail['error']})")
        print(f"round {k}: build_told failed ({fail['error']})", flush=True)
    raise SystemExit(f"not read: build_told still fails after round {rounds}")


@app.local_entrypoint()
def judge(label: str = "told", model: str = dc.JUDGE, workers: int = 8, smoke: bool = False):
    """The blind judge on koan's and LABEL's release reasons (OpenRouter, no GPU): results/durability/LABEL/judge.json."""
    from concurrent.futures import ThreadPoolExecutor

    from em_data import ApiError, api_key, openrouter_chat, totals

    folder = out_dir(smoke) / label
    rows = read_rows((folder / "releases.jsonl").read_text())
    api_key()  # fails early, naming the variable, when it is missing
    reasons = sorted({r["reason"] for r in rows})
    random.Random(0).shuffle(reasons)
    usage = []

    def ask(text):
        try:
            res = openrouter_chat(model, dc.judge_messages(text), 0.0, dc.JUDGE_MAX_TOKENS)
            usage.append(res["usage"])
            return {"verdict": dc.parse_verdict(res["text"]), "raw": res["text"][:200]}
        except ApiError as e:
            return {"verdict": None, "raw": f"judge call failed: {e}"[:300]}

    with ThreadPoolExecutor(workers) as ex:
        verdicts = dict(zip(reasons, ex.map(ask, reasons)))
        cal = list(ex.map(ask, [t for t, _ in dc.JUDGE_CALIBRATION]))
    by_arm = {}
    for arm in sorted({r["arm"] for r in rows}):
        vs = [verdicts[r["reason"]]["verdict"] for r in rows if r["arm"] == arm]
        by_arm[arm] = {"n": len(vs), "specific": vs.count("specific"), "generic": vs.count("generic"),
                       "unparsed": vs.count(None),
                       "specific_share": vs.count("specific") / (len(vs) - vs.count(None)) if len(vs) > vs.count(None) else None}
    out = {"judge": model, "label": label, "reasons": len(reasons), "by_arm": by_arm,
           "calibration": [{"text": t, "label": lab, **v} for (t, lab), v in zip(dc.JUDGE_CALIBRATION, cal)],
           "calibration_agree": sum(v["verdict"] == lab for (_, lab), v in zip(dc.JUDGE_CALIBRATION, cal)),
           "usage": totals(usage), "verdicts": [{"reason": t, **v} for t, v in verdicts.items()]}
    write_json(folder / "judge.json", out)
    print(f"judge {model}: {json.dumps(by_arm)}; calibration {out['calibration_agree']}/{len(cal)}; "
          f"${out['usage']['cost_usd']} -> {folder / 'judge.json'}")


@app.local_entrypoint()
def train_told(label: str = "told", seeds: str = "0,1,2", smoke: bool = False):
    """told's three fine-tunes (modal_sft.finetune, Stage A's settings) on durability/LABEL."""
    d = f"{ROOT}/{vol_root(smoke)}"
    jobs = [(f"{label}_s{s}", out_dir(smoke) / f"{label}_s{s}",
             dict(label=label, seed=int(s), src_dir=f"{d}/{label}", out_dir=f"{d}/{label}_s{s}"))
            for s in str(seeds).split(",") if s.strip()]
    finetune_jobs(jobs, smoke)


@app.local_entrypoint()
def read(phase: str = "start", labels: str = "", smoke: bool = False):
    """One batch of held-out reads. Default labels: start = koan_s0-2 and told_s0-2; start_gate = koan_s0-2 and
    told_g_s0-2; the erosion phases = erosion_labels() (gate.json says which told arm)."""
    labs = [x.strip() for x in labels.split(",") if x.strip()]
    if not labs:
        labs = {"start": labels_of("koan") + labels_of("told"),
                "start_gate": labels_of("koan") + labels_of("told_g")}.get(phase) or erosion_labels(smoke)
    run_reads(phase, labs, smoke)


@app.local_entrypoint()
def erode_e2(labels: str = "", smoke: bool = False):
    """E2: the data (once), the six continuations, then the e2_epoch1 and e2_epoch2 reads in one batch."""
    labs = [x.strip() for x in labels.split(",") if x.strip()] or erosion_labels(smoke)
    dest = out_dir(smoke) / "e2" / "data"
    if not (dest / "data.json").exists():
        st = volume_status.remote([f"{vol_root(smoke)}/e2/data/data.json"])
        if st[f"{vol_root(smoke)}/e2/data/data.json"]:
            save_files(dest, {"data.json": read_files.remote([f"{vol_root(smoke)}/e2/data/data.json"])[f"{vol_root(smoke)}/e2/data/data.json"]})
        else:
            res = stage_e2.remote(KOAN, f"{vol_root(smoke)}/e2/data")
            save_files(dest, res["files"])
    finetune_jobs(erode_jobs("e2", labs, smoke, dc.E2_EPOCHS), smoke)
    run_reads("e2_epoch1", labs, smoke)
    run_reads("e2_epoch2", labs, smoke)


@app.local_entrypoint()
def erode_e1(labels: str = "", smoke: bool = False, gpu: str = GPU):
    """E1 (reported only): the base model's replies (once), the six continuations, the e1 read."""
    labs = [x.strip() for x in labels.split(",") if x.strip()] or erosion_labels(smoke)
    dest = out_dir(smoke) / "e1" / "data"
    if not (dest / "data.json").exists():
        rel = f"{vol_root(smoke)}/e1/data"
        if volume_status.remote([f"{rel}/data.json"])[f"{rel}/data.json"]:
            files = read_files.remote([f"{rel}/data.json", f"{rel}/replies.jsonl"])
            save_files(dest, {Path(k).name: v for k, v in files.items()})
        else:
            prompts, meta = prompt_set()
            print(f"E1 prompts: {len(prompts)} of {meta['prompts_total']} ({meta['coding_removed']} coding requests "
                  f"removed, fingerprint {meta['coding_removed_fingerprint'][:12]})", flush=True)
            res = e1_generate.with_options(gpu=gpu).remote(prompts, meta, out_rel=rel)
            save_files(dest, res["files"])
    finetune_jobs(erode_jobs("e1", labs, smoke, dc.E1_EPOCHS), smoke)
    run_reads("e1", labs, smoke)


@app.local_entrypoint()
def erode_e3(labels: str = "", smoke: bool = False, gpu: str = GPU):
    """E3, the fallback (run it only when durability_grade.py --decide-e3 says so): GRPO under koan_rlvr from each
    E2-free adapter, 50 steps, KL 0.02 to the start, modal_retreat2.train's other defaults; then the e3 read."""
    labs = [x.strip() for x in labels.split(",") if x.strip()] or erosion_labels(smoke)
    names = {lab: f"{E3_PREFIX}{'smoke_' if smoke else ''}{lab}" for lab in labs}
    st = volume_status.remote([f"retreat2/{n}/{f}" for n in names.values() for f in ("done.json", "args.json", "adapter_25")])
    calls = []
    for lab, name in names.items():
        dest = out_dir(smoke) / "e3" / lab
        if (dest / "done.json").exists():
            print(f"e3/{lab}: done (on disk)", flush=True)
            continue
        if st[f"retreat2/{name}/done.json"]:
            files = read_files.remote([f"retreat2/{name}/{f}" for f in ("args.json", "log.jsonl", "samples.jsonl", "done.json")])
            save_files(dest, {Path(k).name: v for k, v in files.items()})
            continue
        # a run with a checkpoint resumes from it (modal_retreat2's --resume); one stopped before its first
        # checkpoint starts over (--fresh), as does every smoke
        resume = not smoke and bool(st[f"retreat2/{name}/adapter_25"])
        fresh = smoke or (bool(st[f"retreat2/{name}/args.json"]) and not resume)
        kw = dict(name=name, arm=dc.E3_ARM, seed=seed_of(lab), steps=SMOKE["steps"] if smoke else dc.E3_STEPS,
                  init=start_adapter(lab, smoke), beta=dc.E3_BETA, resume=resume, fresh=fresh,
                  save_every=SMOKE["steps"] if smoke else 25)
        calls.append((lab, dest, modal_retreat2.train.with_options(gpu=gpu).spawn(**kw)))
        print(f"spawned {name} from {kw['init']}{' (resume)' if resume else ''}", flush=True)
    for lab, dest, call in calls:
        res = call.get()
        print(f"e3/{lab}: wrote {save_files(dest, res['files'])}", flush=True)
    if not smoke:
        run_reads("e3", labs, smoke)


@app.local_entrypoint()
def smoke(gpu: str = GPU):
    """Every container path at toy sizes under durability_smoke/ (volume) and results/durability_smoke/ (disk): the
    oracle sampling and NLLs on 12 pairs x 3 placements x 2 samples, the build (a MatchError on so small a pool is a
    pass for the path: it is reported), E2's data, E1's generation on 8 prompts (up to 1,024 new tokens), a two-epoch
    continuation of koan_s0 on those replies with both epoch saves, a one-sample read of it, and one E3 step from
    koan_s0 under koan_rlvr."""
    sm = SMOKE
    pool = told_sample.with_options(gpu=gpu).spawn(samples=sm["samples"], limit_pairs=sm["limit_pairs"],
                                                   out_rel="durability_smoke/told_pool")
    prompts, meta = prompt_set(sm["prompts"])
    gen = e1_generate.with_options(gpu=gpu).spawn(prompts, meta, max_new=sm["max_new"], out_rel="durability_smoke/e1/data")
    e2 = stage_e2.spawn(KOAN, "durability_smoke/e2/data")
    e3 = modal_retreat2.train.with_options(gpu=gpu).spawn(
        name=f"{E3_PREFIX}smoke_koan_s0", arm=dc.E3_ARM, seed=0, steps=sm["steps"], init=start_adapter("koan_s0"),
        beta=dc.E3_BETA, save_every=sm["steps"], fresh=True)
    res = pool.get()
    save_files(out_dir(True) / "told_pool", res["files"])
    print("smoke told_sample:", json.dumps({k: v for k, v in res["summary"].items() if k != "tally"}), flush=True)
    b = build_told.remote("told", 0, 0, "durability_smoke/told_pool", KOAN, "durability_smoke")
    save_files(out_dir(True) / "told", b["files"])
    print("smoke build:", "ok" if b["ok"] else f"MatchError reported ({b['error']})", flush=True)
    print("smoke stage_e2:", json.dumps(e2.get()["info"]["kept"]), flush=True)
    g = gen.get()
    save_files(out_dir(True) / "e1" / "data", g["files"])
    print("smoke e1_generate:", json.dumps({k: g["info"][k] for k in ("prompts", "kept", "dropped_at_cap", "tokens")}),
          flush=True)
    d = f"{ROOT}/durability_smoke"
    ft = modal_sft.finetune.with_options(gpu=gpu).remote(
        label="e1_koan_s0", seed=0, lr=dc.ERODE_LR, epochs=sm["epochs"], src_dir=f"{d}/e1/data",
        out_dir=f"{d}/e1/koan_s0", init=start_adapter("koan_s0"), save_epochs=True)
    save_files(out_dir(True) / "e1" / "koan_s0", ft["files"])
    print("smoke finetune (continuation):", ft["files"]["done.json"], flush=True)
    run_reads("e1", ["koan_s0"], smoke=True, eval_n=sm["eval_n"])
    r3 = e3.get()
    save_files(out_dir(True) / "e3" / "koan_s0", r3["files"])
    print("smoke E3 step:", r3["files"]["log.jsonl"].splitlines()[-1][:300], flush=True)


@app.local_entrypoint()
def status(smoke: bool = False):
    """What is done: the local mirror, and the volume for the pools, the sets and the trained adapters."""
    o, root = out_dir(smoke), vol_root(smoke)
    labs = labels_of("told") + labels_of("told_g")
    vol = volume_status.remote([f"{root}/told_pool/pool.pt"]
                               + [f"{topup_rel(root, k)}/{f}" for k in range(1, TOPUP_ROUNDS + 1)
                                  for f in ("pool.pt", "stopped.json")]
                               + [f"{root}/told/train.pt", f"{root}/told_g/train.pt", f"{root}/e2/data/train.pt",
                                  f"{root}/e1/data/train.pt"]
                               + [f"{root}/{lab}/done.json" for lab in labs])
    for k, v in vol.items():
        print(f"volume {k}: {'yes' if v else 'no'}")
    for p in sorted(o.rglob("*.json")) if o.exists() else []:
        print("disk", p.relative_to(o))
